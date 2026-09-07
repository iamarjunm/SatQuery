"""HTTP front for the controller, for the backend and the frontend.

    GET  /health                 -> {status, images, planner}
    GET  /images                 -> every loaded scene: id, site, date, sensor, size, bounds, preview URL
    GET  /images/<image_id>.png  -> the preview chip
    POST /query  {query, image_ids?}
        -> handle_query() result, plus "display": for every geometry item in
           every step result, the same ring in pixel coordinates of the image
           that step ran on, so a viewer can draw it over the preview without
           knowing anything about map projections. The lon/lat geometry stays
           untouched; "display" is a convenience on the side.

Session: the real chips in images/manifest.json (via session.manifest_session).
`image_ids` narrows the session to one site's scenes so the planner is not
choosing among all eleven areas; omitted means every loaded image.

Run:  .venv/Scripts/python server.py [--port 8080]
Reads .env the same way e2e.py does, so the planner and agent URLs apply.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(HERE / ".env")

from controller import handle_query  # noqa: E402
from planner import providers  # noqa: E402
from session import Session, manifest_session  # noqa: E402
from tools import TOOLS  # noqa: E402

MANIFEST: dict[str, dict] = json.loads((IMAGES / "manifest.json").read_text(encoding="utf-8"))
_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def scene_list() -> list[dict]:
    out = []
    for image_id, m in MANIFEST.items():
        out.append({
            "image_id": image_id,
            "site": m.get("site", "vienna"),
            "purpose": m.get("purpose", ""),
            "acquired": m["acquired"],
            "modality": m["modality"],
            "source": m.get("source"),
            "cloudy": m.get("cloudy", m.get("scene_cloud_cover_pct", 0) > 40),
            "cloud_cover_pct": m.get("scene_cloud_cover_pct"),
            "size_px": m["size_px"],
            "bounds_epsg4326": m["bounds_epsg4326"],
            "preview": f"/images/{image_id}.png",
        })
    return out


def session_for(image_ids: list[str] | None) -> Session:
    full = manifest_session(IMAGES / "manifest.json")
    if not image_ids:
        return full
    return Session([full.get(i) for i in image_ids if i in MANIFEST])


# --- pixel projection for the viewer ---------------------------------------------

_IMAGE_ARG = {"vqa": "image_id", "ground": "image_id", "change_detect": "image_id_t2",
              "cross_modal": "optical_image_id"}


def _lonlat_to_pixels(image_id: str, rings: list[list[list[float]]]) -> list[list[list[float]]]:
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(IMAGES / MANIFEST[image_id]["file"]) as src:
        out = []
        for ring in rings:
            xs, ys = warp_transform("EPSG:4326", src.crs, [p[0] for p in ring], [p[1] for p in ring])
            out.append([[round(c, 1), round(r, 1)] for c, r in (~src.transform * (x, y) for x, y in zip(xs, ys))])
        return out


def _image_for_step(plan: dict, results: dict, step_id: str) -> str | None:
    """Which image a step's geometry belongs to. Service tools name it in
    their args; the local tools (filter_by_region, count) inherit it from the
    step their objects came from."""
    step = next((s for s in plan.get("steps", []) if s["id"] == step_id), None)
    if step is None:
        return None
    arg = _IMAGE_ARG.get(step["tool"])
    if arg:
        return step["args"].get(arg)
    for value in step["args"].values():
        if isinstance(value, str) and value.startswith("$"):
            return _image_for_step(plan, results, value[1:].split(".")[0])
    return None


def display_layer(result: dict) -> dict:
    """{step_id: {image_id, items: [{label, confidence, pixels}]}} for every
    step whose result carries geometry."""
    plan, results = result.get("plan") or {}, result.get("results") or {}
    layer: dict[str, Any] = {}
    for step_id, step_result in results.items():
        if not isinstance(step_result, dict):
            continue
        image_id = _image_for_step(plan, results, step_id)
        if image_id not in MANIFEST:
            continue
        items = []
        for field in ("objects", "regions"):
            for item in step_result.get(field, []) or []:
                geom = item.get("geometry") if isinstance(item, dict) else None
                if not geom or geom.get("type") not in ("Polygon", "MultiPolygon"):
                    continue
                polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
                rings = [poly[0] for poly in polys if poly]
                items.append({
                    "id": item.get("id"),
                    "label": item.get("label") or item.get("type") or field[:-1],
                    "confidence": item.get("confidence"),
                    "pixels": _lonlat_to_pixels(image_id, rings),
                })
        if items:
            layer[step_id] = {"image_id": image_id, "size_px": MANIFEST[image_id]["size_px"], "items": items}
    return layer


def run_query(query: str, image_ids: list[str] | None) -> dict:
    result = handle_query(query, session_for(image_ids))
    result["display"] = display_layer(result)
    result["images"] = [s for s in scene_list() if not image_ids or s["image_id"] in image_ids]
    return result


# --- HTTP ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, json.dumps(payload).encode())

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, b"")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in ("/", "/health"):
            self._json(200, {"status": "ok", "images": len(MANIFEST),
                             "planner": [p.describe() for p in providers()] or ["keyword fallback only"],
                             "tools": {name: ("mock" if _mocked(name) else "service") for name in
                                       ("vqa", "ground", "change_detect", "cross_modal")}})
        elif path == "/images":
            self._json(200, {"status": "ok", "images": scene_list()})
        elif path.startswith("/images/") and path.endswith(".png"):
            image_id = path[len("/images/"):-4]
            if not _ID.match(image_id) or image_id not in MANIFEST:
                self._json(404, {"status": "error", "error": f"no image {image_id!r}"})
                return
            self._send(200, (IMAGES / MANIFEST[image_id]["preview"]).read_bytes(), "image/png")
        else:
            self._json(404, {"status": "error", "error": "GET /health, /images, /images/<id>.png or POST /query"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?")[0] != "/query":
            self._json(404, {"status": "error", "error": "POST /query"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"status": "error", "error": "body is not JSON"})
            return
        query = body.get("query") if isinstance(body, dict) else None
        if not isinstance(query, str) or not query.strip():
            self._json(400, {"status": "error", "error": "'query' must be a non-empty string"})
            return
        image_ids = body.get("image_ids")
        if image_ids is not None and not (isinstance(image_ids, list) and all(isinstance(i, str) for i in image_ids)):
            self._json(400, {"status": "error", "error": "'image_ids' must be a list of strings"})
            return
        started = time.monotonic()
        try:
            result = run_query(query, image_ids)
        except Exception as exc:  # noqa: BLE001 - never a traceback to the frontend
            self._json(500, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            return
        result["server_elapsed_s"] = round(time.monotonic() - started, 3)
        self._json(200, result)

    def log_message(self, fmt, *args) -> None:
        print(f"{self.command} {self.path} -> {args[1] if len(args) > 1 else ''}", flush=True)


def _mocked(tool: str) -> bool:
    from tools import use_mock
    return use_mock(tool)


def main() -> int:
    ap = argparse.ArgumentParser(description="HTTP front for the SatQuery controller")
    ap.add_argument("--port", type=int, default=int(os.getenv("SATQUERY_CONTROLLER_PORT", "8080")))
    ap.add_argument("--host", default="127.0.0.1")
    ns = ap.parse_args()
    server = ThreadingHTTPServer((ns.host, ns.port), Handler)
    print(f"controller on http://{ns.host}:{ns.port}/  ({len(MANIFEST)} images; planner: "
          f"{', '.join(p.describe() for p in providers()) or 'keyword fallback only'}; Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
