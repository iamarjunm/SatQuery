"""GeoChat agent: the VQA, grounding and cross-modal services, backed by GeoChat-7B.

Speaks the team contract the controller expects (see controller/README.md):

    POST /vqa          {image_id, question}                       -> {answer, confidence}
    POST /ground       {image_id, phrase}                          -> {objects: [{id, label, geometry, confidence}]}
    POST /cross_modal  {optical_image_id, sar_image_id, phrase}    -> {regions: [{label, geometry, confidence}], summary}

wrapped as {"status": "success", "result": {...}}. Every geometry is a GeoJSON
Polygon in EPSG:4326. GeoChat answers in pixel space (boxes on a 0-100 grid
with a rotation angle); the conversion to lon/lat happens here, using the
GeoTIFF's own georeferencing, and pixel coordinates never leave this file.

cross_modal runs grounding independently on the optical and SAR image of a
co-registered pair for the same phrase, then merges: a detection confirmed by
both sensors is reported once with a confidence boost; a detection seen in
only one sensor is still reported at its own confidence -- often the actual
point of combining them (cloud hides something from optical that SAR still
sees, or vice versa), not a case to discard.

Three backends:
  stub     canned answers, no GPU -- for running the service on a laptop and
           proving the contract, the geometry conversion and the controller
           wiring before anyone touches a model.
  gemini   Gemini vision over the OpenAI-compatible endpoint. No GPU, no
           install, real answers on the free tier. Rotates across models when
           one hits its daily quota. The prototype's brain until GeoChat is up.
  geochat  the real model, mirrors geochat/eval/batch_geochat_grounding.py
           from the GeoChat repo. Needs a GPU and the repo installed.

Every non-stub answer is cached on disk keyed by tool + arguments, so a
rehearsed demo query never spends a second call.

Run:  python service.py --backend stub                       # ports 8001 (vqa) + 8002 (ground)
      python service.py --backend gemini                     # needs GEMINI_API_KEY (or the
                                                             # controller's SATQUERY_LLM_FALLBACK_API_KEY)
      python service.py --backend geochat --model-path /kaggle/working/geochat-7B --load-4bit
      python service.py ... --port 8000                     # one port for both, then set
                                                             # SATQUERY_VQA_URL / SATQUERY_GROUND_URL
Images come from controller/images/manifest.json (or --images DIR).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGES = HERE.parents[1] / "controller" / "images"
DEFAULT_CACHE = HERE / "cache"
PORTS = {"vqa": 8001, "ground": 8002, "cross_modal": 8004}  # the controller's defaults, see controller/tools.py


def _load_dotenv(*paths: Path) -> None:
    """KEY=value lines into os.environ without overriding what is already set.
    The agent's own .env first, then the controller's, so one Gemini key in
    controller/.env serves both the planner fallback and this agent."""
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(HERE / ".env", HERE.parents[1] / "controller" / ".env")

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
        # GeoChat writes integers; the Gemini backend writes decimals so a
        # 1000-grid box is not rounded to whole percent.
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))]
        if len(nums) < 4:
            continue
        x0, y0, x1, y1 = nums[:4]
        angle = nums[4] if len(nums) > 4 else 0
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


class GeminiBackend:
    """Gemini vision through Google's OpenAI-compatible endpoint. The
    prototype's brain: no GPU, no install, real answers about the real chips.

    Grounding: Gemini is asked for boxes as JSON on its documented 0-1000
    grid ([ymin, xmin, ymax, xmax]); they are rewritten into GeoChat's text
    format so ground() and parse_boxes() do not know which model answered.

    Quota: the free tier allows a small number of requests a day per model,
    and each model has its own allowance. A quota or not-found error puts
    that model on cooldown and the next one in the list is tried, so a demo
    survives one model running dry. Every model dry raises, and the service
    turns that into an error envelope."""
    name = "gemini"

    DEFAULT_MODELS = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
                      "gemini-flash-lite-latest", "gemini-3.1-flash-lite")
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    COOLDOWN_S = 60 * 60  # a rate-limited model is skipped for an hour
    MAX_SIDE = 1024       # resize before sending; enough for a 5 km chip, fewer tokens

    VQA_PREAMBLE = ("This is a Sentinel-2 true-colour satellite image at 10 m per pixel, "
                    "about 5 km across. Answer the question in one or two plain sentences, "
                    "naming only what is visible. Question: ")
    GROUND_PROMPT = (
        "This is a Sentinel-2 true-colour satellite image at 10 m per pixel. Find every "
        "instance of: {phrase}. Return JSON only, in the form "
        '{{"objects": [{{"label": "<short label>", "box_2d": [ymin, xmin, ymax, xmax]}}]}} '
        "with box_2d on a 0-1000 grid over the image. If there are none, return "
        '{{"objects": []}}. Tight boxes, one per distinct instance, at most 40.')

    def __init__(self, api_key: str | None = None, models: tuple[str, ...] | None = None,
                 base_url: str | None = None):
        from openai import OpenAI

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("SATQUERY_LLM_FALLBACK_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY (or SATQUERY_LLM_FALLBACK_API_KEY in controller/.env)")
        env_models = os.getenv("GEMINI_MODELS")
        self.models = tuple(models or (tuple(m.strip() for m in env_models.split(",") if m.strip())
                                       if env_models else self.DEFAULT_MODELS))
        self.client = OpenAI(base_url=base_url or self.BASE_URL, api_key=key, timeout=90, max_retries=0)
        self._cooldown_until: dict[str, float] = {}
        self.last_model: str | None = None

    # -- transport ---------------------------------------------------------------
    @staticmethod
    def _image_part(image: Image.Image) -> dict:
        image = image.convert("RGB")
        image.thumbnail((GeminiBackend.MAX_SIDE, GeminiBackend.MAX_SIDE))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}

    @staticmethod
    def _is_quota_or_missing(exc: Exception) -> bool:
        text = f"{type(exc).__name__}: {exc}".lower()
        return any(s in text for s in ("429", "quota", "rate limit", "ratelimit", "resource_exhausted",
                                       "404", "not found", "notfound"))

    def _available(self) -> list[str]:
        now = time.time()
        return [m for m in self.models if self._cooldown_until.get(m, 0) <= now]

    def _complete(self, image: Image.Image, prompt: str, *, json_mode: bool = False) -> str:
        """One completion against the first model that is not on cooldown."""
        failures = []
        for model in self._available():
            kwargs: dict[str, Any] = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = self.client.chat.completions.create(
                    model=model, temperature=0,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt},
                                                           self._image_part(image)]}],
                    **kwargs)
            except Exception as exc:  # noqa: BLE001 - classify, then move on or re-raise
                if self._is_quota_or_missing(exc):
                    self._cooldown_until[model] = time.time() + self.COOLDOWN_S
                    failures.append(f"{model}: {' '.join(str(exc).split())[:100]}")
                    continue
                raise
            self.last_model = model
            return response.choices[0].message.content or ""
        raise RuntimeError("every Gemini model is rate-limited or unavailable: " + " | ".join(failures)
                           if failures else "every Gemini model is on cooldown; try again later")

    # -- answers ------------------------------------------------------------------
    @staticmethod
    def _boxes_from_json(text: str) -> list[tuple[str, float, float, float, float]]:
        """Gemini's {"objects": [{"label", "box_2d": [ymin, xmin, ymax, xmax]}]} on a
        0-1000 grid -> (label, x0, y0, x1, y1) on the 0-100 grid GeoChat uses."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        items = data.get("objects", data) if isinstance(data, dict) else data
        out = []
        for item in items if isinstance(items, list) else []:
            box = item.get("box_2d") if isinstance(item, dict) else None
            if not (isinstance(box, list) and len(box) == 4):
                continue
            try:
                ymin, xmin, ymax, xmax = (float(v) for v in box)
            except (TypeError, ValueError):
                continue
            out.append((str(item.get("label", "")), xmin / 10, ymin / 10, xmax / 10, ymax / 10))
        return out

    def infer(self, image: Image.Image, prompt: str) -> str:
        if prompt.startswith("[refer]"):
            phrase = prompt.split("<p>")[1].split("</p>")[0].strip() if "<p>" in prompt else prompt
            raw = self._complete(image, self.GROUND_PROMPT.format(phrase=phrase), json_mode=True)
            return " ".join(f"<p>{label or phrase}</p> {{<{x0:.1f}><{y0:.1f}><{x1:.1f}><{y1:.1f}>|<0>}}"
                            for label, x0, y0, x1, y1 in self._boxes_from_json(raw))
        return self._complete(image, self.VQA_PREAMBLE + prompt)


class ResultCache:
    """One JSON file per (backend, tool, args). Warm it with the demo queries
    the night before and the demo never waits on a model, per the plan doc.
    Commit the folder if the whole team should share the warmed answers."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, backend: str, tool: str, args: dict) -> Path:
        key = hashlib.sha256(json.dumps([backend, tool, args], sort_keys=True).encode()).hexdigest()[:24]
        return self.dir / f"{tool}-{args.get('image_id', 'x')}-{key}.json"

    def get(self, backend: str, tool: str, args: dict) -> dict | None:
        path = self._path(backend, tool, args)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["result"]

    def put(self, backend: str, tool: str, args: dict, result: dict, model: str | None) -> None:
        payload = {"backend": backend, "model": model, "tool": tool, "args": args,
                   "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "result": result}
        self._path(backend, tool, args).write_text(json.dumps(payload, indent=1), encoding="utf-8")


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


# A detection in one sensor merges with one in the other when their boxes
# overlap by at least this much (intersection-over-union). Below this they are
# reported as separate, single-sensor detections rather than forced together.
CROSS_MODAL_IOU_THRESHOLD = 0.3

# Added to the higher of the two individual confidences when both sensors
# independently detect the same region -- real agreement between two
# different measurements, not a guess -- capped at 1.0.
CROSS_MODAL_AGREEMENT_BONUS = 0.15


def _iou(a, b) -> float:
    if not a.intersects(b):
        return 0.0
    intersection = a.intersection(b).area
    union = a.union(b).area
    return intersection / union if union > 0 else 0.0


def cross_modal(store: ImageStore, backend: Backend, args: dict) -> dict:
    from shapely.geometry import shape

    optical_id, sar_id = args["optical_image_id"], args["sar_image_id"]
    phrase = str(args["phrase"])

    optical = ground(store, backend, {"image_id": optical_id, "phrase": phrase})["objects"]
    sar = ground(store, backend, {"image_id": sar_id, "phrase": phrase})["objects"]
    optical_shapes = [shape(r["geometry"]) for r in optical]
    sar_shapes = [shape(r["geometry"]) for r in sar]

    regions = []
    matched_sar: set[int] = set()
    both_count = 0

    for opt_r, opt_shape in zip(optical, optical_shapes):
        best_i, best_iou = None, 0.0
        for i, sar_shape in enumerate(sar_shapes):
            if i in matched_sar:
                continue
            iou = _iou(opt_shape, sar_shape)
            if iou > best_iou:
                best_i, best_iou = i, iou

        if best_i is not None and best_iou >= CROSS_MODAL_IOU_THRESHOLD:
            matched_sar.add(best_i)
            sar_r = sar[best_i]
            both_count += 1
            confidence = round(min(1.0, max(opt_r["confidence"], sar_r["confidence"]) + CROSS_MODAL_AGREEMENT_BONUS), 3)
            # The higher-confidence sensor's own box is kept as the reported
            # geometry rather than a synthetic union/intersection shape, which
            # can self-intersect or balloon in area for two boxes that only
            # partially overlap.
            geometry = opt_r["geometry"] if opt_r["confidence"] >= sar_r["confidence"] else sar_r["geometry"]
            regions.append({"label": f"{phrase} (optical+SAR)", "geometry": geometry, "confidence": confidence})
        else:
            regions.append({"label": f"{phrase} (optical only)", "geometry": opt_r["geometry"], "confidence": opt_r["confidence"]})

    for i, sar_r in enumerate(sar):
        if i not in matched_sar:
            regions.append({"label": f"{phrase} (SAR only)", "geometry": sar_r["geometry"], "confidence": sar_r["confidence"]})

    total = len(regions)
    if total:
        summary = (
            f"Found {total} instance{'s' if total != 1 else ''} of '{phrase}': "
            f"{both_count} confirmed by both optical and SAR, "
            f"{len(optical) - both_count} seen only in optical, "
            f"{len(sar) - both_count} seen only in SAR."
        )
    else:
        summary = f"No instances of '{phrase}' found in either the optical or SAR image."

    return {"regions": regions, "summary": summary}


TOOLS = {
    "vqa": (vqa, ("image_id", "question")),
    "ground": (ground, ("image_id", "phrase")),
    "cross_modal": (cross_modal, ("optical_image_id", "sar_image_id", "phrase")),
}


def respond(store: ImageStore, backend: Backend, tool: str, args: Any,
            cache: ResultCache | None = None) -> tuple[int, dict]:
    """Pure function behind the HTTP handler so tests need no socket."""
    if tool not in TOOLS:
        return 404, {"status": "error", "error": f"unknown tool {tool!r}; this agent serves {sorted(TOOLS)}"}
    fn, required = TOOLS[tool]
    if not isinstance(args, dict):
        return 400, {"status": "error", "error": "body must be a JSON object"}
    missing = [k for k in required if k not in args]
    if missing:
        return 400, {"status": "error", "error": f"{tool}: missing {missing}"}
    for key in required:
        if key == "image_id" or key.endswith("_image_id"):
            if args[key] not in store:
                return 404, {"status": "error", "error": f"{tool}: image {args[key]!r} is not loaded"}
    call_args = {k: args[k] for k in required}
    if cache is not None:
        hit = cache.get(backend.name, tool, call_args)
        if hit is not None:
            return 200, {"status": "success", "result": hit, "cached": True}
    try:
        result = fn(store, backend, call_args)
    except Exception as exc:  # noqa: BLE001 - the controller wants an envelope, not a traceback
        return 500, {"status": "error", "error": f"{tool}: {type(exc).__name__}: {exc}"}
    if cache is not None:
        cache.put(backend.name, tool, call_args, result, getattr(backend, "last_model", None))
    return 200, {"status": "success", "result": result}


# --- HTTP ---------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    store: ImageStore
    backend: Backend
    cache: ResultCache | None = None

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
        status, body = respond(self.store, self.backend, self.path.strip("/"), args, self.cache)
        self._send(status, body)

    def log_message(self, fmt, *args) -> None:
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


def serve(store: ImageStore, backend: Backend, ports: list[int], cache: ResultCache | None = None) -> None:
    Handler.store, Handler.backend, Handler.cache = store, backend, cache
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
    ap.add_argument("--backend", choices=["stub", "gemini", "geochat"], default="stub")
    ap.add_argument("--models", help="gemini: comma-separated models to rotate through "
                                     f"(default {','.join(GeminiBackend.DEFAULT_MODELS)})")
    ap.add_argument("--model-path", help="GeoChat-7B directory (geochat backend)")
    ap.add_argument("--model-base", default=None)
    ap.add_argument("--load-4bit", action="store_true", help="fits one 16 GB T4")
    ap.add_argument("--load-8bit", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--images", default=str(DEFAULT_IMAGES), help="folder holding manifest.json")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE), help="answer cache for non-stub backends")
    ap.add_argument("--no-cache", action="store_true", help="always call the model")
    ap.add_argument("--port", type=int, help="serve both tools on one port instead of 8001/8002")
    ns = ap.parse_args()

    store = ImageStore(Path(ns.images))
    if ns.backend == "geochat":
        if not ns.model_path:
            ap.error("--model-path is required with --backend geochat")
        backend: Backend = GeoChatBackend(ns.model_path, ns.model_base, load_4bit=ns.load_4bit,
                                          load_8bit=ns.load_8bit, device=ns.device)
    elif ns.backend == "gemini":
        models = tuple(m.strip() for m in ns.models.split(",")) if ns.models else None
        backend = GeminiBackend(models=models)
    else:
        backend = StubBackend()
    cache = None if ns.no_cache or backend.name == "stub" else ResultCache(Path(ns.cache_dir))
    serve(store, backend, [ns.port] if ns.port else sorted(set(PORTS.values())), cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
