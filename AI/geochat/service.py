"""GeoChat agent: the VQA and grounding services, backed by GeoChat-7B.

Speaks the team contract the controller expects (see controller/README.md):

    POST /vqa     {image_id, question}  -> {answer, confidence}
    POST /ground  {image_id, phrase}    -> {objects: [{id, label, geometry, confidence}]}

wrapped as {"status": "success", "result": {...}}. Every geometry is a GeoJSON
Polygon in EPSG:4326. GeoChat answers in pixel space (boxes on a 0-100 grid
with a rotation angle); the conversion to lon/lat happens here, using the
GeoTIFF's own georeferencing, and pixel coordinates never leave this file.

Two backends:
  stub     canned answers, no GPU -- for running the service on a laptop and
           proving the contract, the geometry conversion and the controller
           wiring before anyone touches a model.
  geochat  the real model, mirrors geochat/eval/batch_geochat_grounding.py
           from the GeoChat repo. Needs a GPU and the repo installed.

Run:  python service.py --backend stub                       # ports 8001 (vqa) + 8002 (ground)
      python service.py --backend geochat --model-path /kaggle/working/geochat-7B --load-4bit
      python service.py ... --port 8000                     # one port for both, then set
                                                             # SATQUERY_VQA_URL / SATQUERY_GROUND_URL
Images come from controller/images/manifest.json (or --images DIR).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

DEFAULT_IMAGES = Path(__file__).resolve().parents[2] / "controller" / "images"
PORTS = {"vqa": 8001, "ground": 8002}  # the controller's defaults, see controller/tools.py

# GeoChat produces no confidence. The contract needs one in [0, 1] and the
# controller takes the minimum across a chain and abstains below 0.5, so these
# are a stated policy, not a measurement: a fixed value per tool, overridable.
VQA_CONFIDENCE = float(os.getenv("GEOCHAT_VQA_CONFIDENCE", "0.70"))
GROUND_CONFIDENCE = float(os.getenv("GEOCHAT_GROUND_CONFIDENCE", "0.60"))

# Prompts, from the repo's evaluation scripts. [refer] asks for the location of
# a phrase; the model answers with <p>label</p> {<x0><y0><x1><y1>|<angle>}.
REFER_PROMPT = "[refer] Give me the location of <p> {phrase} </p>"
GRID = 100.0  # GeoChat box coordinates are on a 0-100 grid over the image


# --- parsing GeoChat's answer -------------------------------------------------

_BRACED = re.compile(r"\{([^{}]*)\}")
_LABELLED = re.compile(r"<p>\s*(.*?)\s*</p>\s*\{([^{}]*)\}", re.DOTALL)


def parse_boxes(text: str) -> list[dict[str, Any]]:
    """Every box in a GeoChat answer as {label, x0, y0, x1, y1, angle} on the
    0-100 grid. Tolerant of missing angles, reversed corners, out-of-range
    values (clamped) and prose around the boxes. A box with no area is
    dropped. Labels come from the <p>..</p> tag before each box when present."""
    labelled = {m.group(2): m.group(1) for m in _LABELLED.finditer(text)}
    boxes = []
    for m in _BRACED.finditer(text):
        ints = [int(v) for v in re.findall(r"-?\d+", m.group(1))]
        if len(ints) < 4:
            continue
        x0, y0, x1, y1 = ints[:4]
        angle = ints[4] if len(ints) > 4 else 0
        x0, x1 = sorted((max(0, min(GRID, x0)), max(0, min(GRID, x1))))
        y0, y1 = sorted((max(0, min(GRID, y0)), max(0, min(GRID, y1))))
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            continue
        boxes.append({"label": labelled.get(m.group(1), "").strip(),
                      "x0": x0, "y0": y0, "x1": x1, "y1": y1, "angle": angle})
    return boxes


# --- images and georeferencing -----------------------------------------------

class ImageStore:
    """The chips in controller/images, keyed by image_id via manifest.json."""

    def __init__(self, images_dir: Path = DEFAULT_IMAGES):
        self.dir = Path(images_dir)
        manifest = self.dir / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"no manifest at {manifest}; run controller/fetch_chips.py first")
        self.manifest: dict[str, dict] = json.loads(manifest.read_text(encoding="utf-8"))

    def __contains__(self, image_id: str) -> bool:
        return image_id in self.manifest

    def path(self, image_id: str) -> Path:
        return self.dir / self.manifest[image_id]["file"]

    def open(self, image_id: str) -> Image.Image:
        # The preview PNG is the same pixels as the GeoTIFF, without rasterio
        # in the read path; the GeoTIFF is only opened for its georeferencing.
        preview = self.dir / self.manifest[image_id].get("preview", "")
        if preview.is_file():
            return Image.open(preview).convert("RGB")
        import rasterio
        with rasterio.open(self.path(image_id)) as src:
            return Image.fromarray(src.read([1, 2, 3]).transpose(1, 2, 0)).convert("RGB")

    def pixels_to_lonlat(self, image_id: str, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """(col, row) pixel corners -> (lon, lat), via the GeoTIFF's affine
        transform and CRS. This is the only place pixel space meets the map."""
        import rasterio
        from rasterio.warp import transform as warp_transform

        with rasterio.open(self.path(image_id)) as src:
            xs, ys = zip(*(src.transform * (col, row) for col, row in points))
            lons, lats = warp_transform(src.crs, "EPSG:4326", list(xs), list(ys))
        return list(zip(lons, lats))


def box_to_pixel_corners(box: dict, width: int, height: int) -> list[tuple[float, float]]:
    """Grid box -> four pixel corners, rotated about the box centre by the
    model's angle (degrees, image axes with y down), then clamped to the image
    so a rotated corner can never poke outside the footprint."""
    sx, sy = width / GRID, height / GRID
    x0, y0, x1, y1 = box["x0"] * sx, box["y0"] * sy, box["x1"] * sx, box["y1"] * sy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    theta = math.radians(box.get("angle", 0) or 0)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    corners = []
    for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        dx, dy = x - cx, y - cy
        rx, ry = cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t
        corners.append((max(0.0, min(float(width), rx)), max(0.0, min(float(height), ry))))
    return corners


def box_to_geojson(store: ImageStore, image_id: str, box: dict, width: int, height: int) -> dict:
    ring = store.pixels_to_lonlat(image_id, box_to_pixel_corners(box, width, height))
    ring = [[round(lon, 7), round(lat, 7)] for lon, lat in ring]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


# --- backends -----------------------------------------------------------------

class Backend(Protocol):
    name: str

    def infer(self, image: Image.Image, prompt: str) -> str: ...


class StubBackend:
    """Canned GeoChat-shaped answers so everything except the model can be
    exercised on a laptop. Two boxes per grounding call, one of them rotated."""
    name = "stub"

    def infer(self, image: Image.Image, prompt: str) -> str:
        if prompt.startswith("[refer]"):
            phrase = prompt.split("<p>")[1].split("</p>")[0].strip() if "<p>" in prompt else "object"
            return (f"<p>{phrase}</p> {{<10><12><30><28>|<0>}} and "
                    f"<p>{phrase}</p> {{<55><50><72><66>|<15>}}")
        return "The image shows a dense built-up area with roads, buildings and a river."


class GeoChatBackend:
    """GeoChat-7B through the repo's own LLaVA-style API. Mirrors
    geochat/eval/batch_geochat_grounding.py: llava_v1 template, 504 px
    preprocessing, greedy decoding. Written against the repo, not yet run on
    a GPU here -- the first Kaggle run is where any drift shows up."""
    name = "geochat"

    def __init__(self, model_path: str, model_base: str | None = None, *, load_4bit: bool = True,
                 load_8bit: bool = False, device: str = "cuda", conv_mode: str = "llava_v1",
                 max_new_tokens: int = 256):
        import torch
        from geochat.mm_utils import get_model_name_from_path
        from geochat.model.builder import load_pretrained_model

        self.torch = torch
        self.device, self.conv_mode, self.max_new_tokens = device, conv_mode, max_new_tokens
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, model_base, get_model_name_from_path(model_path), load_8bit, load_4bit, device=device)
        self._lock = threading.Lock()  # one generation at a time on one GPU

    def infer(self, image: Image.Image, prompt: str) -> str:
        from geochat.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                                       DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
        from geochat.conversation import SeparatorStyle, conv_templates
        from geochat.mm_utils import tokenizer_image_token

        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(self.device)
        pixels = self.image_processor.preprocess(
            image, return_tensors="pt", crop_size={"height": 504, "width": 504},
            size={"shortest_edge": 504})["pixel_values"]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        with self._lock, self.torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=pixels.half().to(self.device), do_sample=False,
                num_beams=1, max_new_tokens=self.max_new_tokens, length_penalty=2.0, use_cache=True)
        # The repo decodes from the prompt length; newer transformers already
        # strip the prompt, in which case the slice is empty and we take it all.
        new_tokens = output_ids[:, input_ids.shape[1]:]
        text = self.tokenizer.batch_decode(new_tokens if new_tokens.shape[1] else output_ids,
                                           skip_special_tokens=True)[0].strip()
        if stop_str and text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()
        return text


# --- the two tools ------------------------------------------------------------

def vqa(store: ImageStore, backend: Backend, args: dict) -> dict:
    image = store.open(args["image_id"])
    answer = backend.infer(image, str(args["question"])).strip() or "No answer."
    return {"answer": answer, "confidence": VQA_CONFIDENCE}


def ground(store: ImageStore, backend: Backend, args: dict) -> dict:
    image_id, phrase = args["image_id"], str(args["phrase"])
    image = store.open(image_id)
    text = backend.infer(image, REFER_PROMPT.format(phrase=phrase))
    objects = []
    for i, box in enumerate(parse_boxes(text), start=1):
        objects.append({
            "id": f"obj_{i:03d}",
            "label": box["label"] or phrase,
            "geometry": box_to_geojson(store, image_id, box, image.width, image.height),
            "confidence": GROUND_CONFIDENCE,
        })
    return {"objects": objects, "raw": text}


TOOLS = {"vqa": (vqa, ("image_id", "question")), "ground": (ground, ("image_id", "phrase"))}


def respond(store: ImageStore, backend: Backend, tool: str, args: Any) -> tuple[int, dict]:
    """Pure function behind the HTTP handler so tests need no socket."""
    if tool not in TOOLS:
        return 404, {"status": "error", "error": f"unknown tool {tool!r}; this agent serves {sorted(TOOLS)}"}
    fn, required = TOOLS[tool]
    if not isinstance(args, dict):
        return 400, {"status": "error", "error": "body must be a JSON object"}
    missing = [k for k in required if k not in args]
    if missing:
        return 400, {"status": "error", "error": f"{tool}: missing {missing}"}
    if args["image_id"] not in store:
        return 404, {"status": "error", "error": f"{tool}: image {args['image_id']!r} is not loaded"}
    try:
        return 200, {"status": "success", "result": fn(store, backend, args)}
    except Exception as exc:  # noqa: BLE001 - the controller wants an envelope, not a traceback
        return 500, {"status": "error", "error": f"{tool}: {type(exc).__name__}: {exc}"}


# --- HTTP ---------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    store: ImageStore
    backend: Backend

    def _send(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("", "/health"):
            self._send(200, {"status": "success", "backend": self.backend.name,
                             "tools": sorted(TOOLS), "images": sorted(self.store.manifest)})
        else:
            self._send(404, {"status": "error", "error": "POST /vqa or POST /ground"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        try:
            args = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"status": "error", "error": "body is not JSON"})
            return
        status, body = respond(self.store, self.backend, self.path.strip("/"), args)
        self._send(status, body)

    def log_message(self, fmt, *args) -> None:
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


def serve(store: ImageStore, backend: Backend, ports: list[int]) -> None:
    Handler.store, Handler.backend = store, backend
    servers = [ThreadingHTTPServer(("0.0.0.0", p), Handler) for p in ports]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"geochat agent [{backend.name}] serving /vqa and /ground on "
          + ", ".join(f"http://0.0.0.0:{p}/" for p in ports) + f"  ({len(store.manifest)} images; Ctrl+C to stop)",
          flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            srv.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser(description="GeoChat VQA + grounding agent")
    ap.add_argument("--backend", choices=["stub", "geochat"], default="stub")
    ap.add_argument("--model-path", help="GeoChat-7B directory (geochat backend)")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--load-4bit", action="store_true", help="fits one 16 GB T4")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES), help="folder holding manifest.json")
    ap.add_argument("--port", type=int, help="serve both tools on one port instead of 8001/8002")
    ns = ap.parse_args()

    store = ImageStore(Path(ns.images))
    if ns.backend == "geochat":
        if not ns.model_path:
            ap.error("--model-path is required with --backend geochat")
        backend: Backend = GeoChatBackend(ns.model_path, ns.model_base, load_4bit=ns.load_4bit,
                                          load_8bit=ns.load_8bit, device=ns.device)
    else:
        backend = StubBackend()
    serve(store, backend, [ns.port] if ns.port else sorted(set(PORTS.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
