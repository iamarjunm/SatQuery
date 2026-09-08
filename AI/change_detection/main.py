"""FastAPI service wrapping the change-detection pipeline.

    POST /change-detection   run the pipeline on an image pair
    GET  /health             device and model-load status
    GET  /results/...        static mask and overlay images

The pipeline runs as an explicitly traced sequence -- validate, detect,
analyse, describe, render -- and every response carries that trace, including
responses that fail partway. `analysis.workflow` names the four analytical
stages; `execution.steps` records every timed stage including image I/O, so
the two are related but not identical by design.

Results are cached on (before_path, after_path, threshold, min_region_px,
query) so a repeated request never re-enters the GPU. The model is warmed at
startup to absorb the ~2.2 s CUDA cold start.
"""

from __future__ import annotations

import os
import time
import uuid
import warnings
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from pathlib import Path
from threading import Lock

import numpy as np
import rasterio
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from rasterio.errors import NotGeoreferencedWarning

from change_model import MODEL_NAME, detect_change, get_model
from describer import answer_question, describe_change
from mask_analysis import analyse_mask

SERVICE_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = SERVICE_ROOT / "results"

# mask_analysis labels with a 3x3 structure of ones -- 8-connectivity, which
# scipy calls connectivity 2. It is not a request parameter, so it is reported
# as the constant it is rather than echoed from the request. Keep this in step
# with the `structure` argument in mask_analysis.analyse_mask.
CONNECTIVITY = 2

OVERLAY_ALPHA = 0.45
OVERLAY_COLOUR = (255, 0, 0)

WORKFLOW = ["validate_inputs", "change_detection", "mask_analysis", "change_description"]

# Error code -> HTTP status. The body carries the code either way; the status
# is there so ordinary HTTP tooling sees a failure as a failure.
ERROR_STATUS = {
    "IMAGE_NOT_FOUND": 404,
    "MISSING_SECOND_IMAGE": 400,
    "INVALID_IMAGE": 400,
    "SIZE_MISMATCH": 400,
    "MODEL_FAILED": 500,
}

_cache: dict[tuple, dict] = {}
_cache_lock = Lock()


class PipelineError(Exception):
    """A failure that maps to one of the documented error codes."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ChangeDetectionRequest(BaseModel):
    before_path: str = Field(..., description="path to the earlier image")
    # Optional so that omitting it produces MISSING_SECOND_IMAGE from the
    # validation step rather than a 422 that never enters the trace.
    after_path: str | None = Field(None, description="path to the later image")
    query: str | None = None
    before_date: str | None = None
    after_date: str | None = None
    threshold: float = 0.5
    min_region_px: int = 50


class ExecutionTrace:
    """Collects one timed entry per pipeline stage."""

    def __init__(self):
        self._started = time.perf_counter()
        self.steps: list[dict] = []

    @contextmanager
    def step(self, action: str, **extra):
        entry = {
            "step": len(self.steps) + 1,
            "action": action,
            "status": "running",
            "duration_ms": 0.0,
        }
        entry.update(extra)
        self.steps.append(entry)
        started = time.perf_counter()
        try:
            yield entry
        except Exception:
            entry["status"] = "failed"
            raise
        else:
            entry["status"] = "completed"
        finally:
            entry["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)

    def total_ms(self) -> float:
        return round((time.perf_counter() - self._started) * 1000, 1)


def _probe(path: Path) -> dict:
    """Read one image's format and georeferencing without loading pixels."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as dataset:
                return {
                    "driver": dataset.driver,
                    "width": dataset.width,
                    "height": dataset.height,
                    "bands": dataset.count,
                    "dtype": dataset.dtypes[0],
                    "crs": str(dataset.crs) if dataset.crs else None,
                    "transform": list(dataset.transform)[:6],
                }
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "INVALID_IMAGE", f"{path.name} could not be read as an image: {exc}"
        ) from exc


def _validate_inputs(before_path: str, after_path: str | None) -> dict:
    """Check count, modality, format, metadata and compatibility.

    Returns the details recorded on the validation step. Raises PipelineError
    with the appropriate code on any failure.
    """
    if not after_path:
        raise PipelineError(
            "MISSING_SECOND_IMAGE",
            "change detection needs two images; after_path was not provided",
        )

    before, after = Path(before_path), Path(after_path)

    for label, path in (("before_path", before), ("after_path", after)):
        if not path.is_file():
            raise PipelineError("IMAGE_NOT_FOUND", f"{label} does not exist: {path}")

    before_info = _probe(before)
    after_info = _probe(after)

    before_size = (before_info["width"], before_info["height"])
    after_size = (after_info["width"], after_info["height"])
    if before_size != after_size:
        raise PipelineError(
            "SIZE_MISMATCH",
            f"images must have identical dimensions: before is "
            f"{before_size[0]}x{before_size[1]}, after is {after_size[0]}x{after_size[1]}",
        )

    before_geo = before_info["crs"] is not None
    after_geo = after_info["crs"] is not None

    # Reported, not enforced: a mismatch is worth surfacing, but the pixel
    # grids already align, so the pipeline can still run.
    crs_match = before_info["crs"] == after_info["crs"]
    transform_match = before_info["transform"] == after_info["transform"]

    return {
        "count": {"required": 2, "received": 2, "ok": True},
        "modality": {
            "before_bands": before_info["bands"],
            "after_bands": after_info["bands"],
            "dtype": before_info["dtype"],
            "ok": True,
        },
        "format": {
            "before_driver": before_info["driver"],
            "after_driver": after_info["driver"],
            "before_suffix": before.suffix.lower(),
            "after_suffix": after.suffix.lower(),
            "ok": True,
        },
        "metadata": {
            "before_georeferenced": before_geo,
            "after_georeferenced": after_geo,
            "before_crs": before_info["crs"],
            "after_crs": after_info["crs"],
        },
        "compatibility": {
            "dimensions": list(before_size),
            "dimensions_match": True,
            "crs_match": crs_match,
            "transform_match": transform_match,
            "ok": True,
        },
    }


def _write_visualisation(
    query_id: str, after_path: str, probability: np.ndarray, threshold: float
) -> dict:
    """Write the binary mask and the red-tinted overlay. Returns their URLs."""
    output_dir = RESULTS_DIR / query_id
    output_dir.mkdir(parents=True, exist_ok=True)

    mask = probability > threshold

    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
        output_dir / "change_mask.png"
    )

    with Image.open(after_path) as handle:
        after_rgb = np.asarray(handle.convert("RGB"), dtype=np.float32)

    overlay = after_rgb.copy()
    tint = np.array(OVERLAY_COLOUR, dtype=np.float32)
    # Blend only the changed pixels, so unchanged ground keeps its true colour
    # and the tint stays readable when projected.
    overlay[mask] = after_rgb[mask] * (1.0 - OVERLAY_ALPHA) + tint * OVERLAY_ALPHA
    Image.fromarray(overlay.astype(np.uint8), mode="RGB").save(output_dir / "overlay.png")

    return {
        "change_mask_url": f"/results/{query_id}/change_mask.png",
        "overlay_url": f"/results/{query_id}/overlay.png",
    }


def _cache_key(request: ChangeDetectionRequest) -> tuple:
    def norm(path: str | None) -> str | None:
        return str(Path(path).resolve()).lower() if path else None

    return (
        norm(request.before_path),
        norm(request.after_path),
        request.threshold,
        request.min_region_px,
        request.query,
    )


def _error_response(query_id: str, trace: ExecutionTrace, error: PipelineError, parameters: dict):
    body = {
        "query_id": query_id,
        "status": "error",
        "error": {"code": error.code, "message": error.message},
        "execution": {
            "steps": trace.steps,
            "model": MODEL_NAME,
            "parameters": parameters,
            "total_ms": trace.total_ms(),
        },
    }
    return JSONResponse(status_code=ERROR_STATUS.get(error.code, 500), content=body)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the model so the first real request does not pay CUDA start-up."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        model, device = get_model()
        # A dummy pair triggers kernel selection as well as weight loading;
        # without it the first request still costs seconds.
        dummy = torch.zeros(1, 3, 256, 256, device=device)
        with torch.no_grad():
            model(dummy, dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        print(f"[startup] {MODEL_NAME} warmed on {device}")
    except Exception as exc:  # a cold model should not stop the service booting
        print(f"[startup] model warm-up failed: {exc}")
    yield


app = FastAPI(
    title="SatQuery change detection",
    description="BIT_CD change detection over an image pair, with region analysis.",
    lifespan=lifespan,
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


@app.get("/health")
def health() -> dict:
    """Report device and whether the model is resident, without loading it."""
    import change_model

    loaded = change_model._model is not None
    device = str(change_model._device) if loaded else None

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_loaded": loaded,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cached_responses": len(_cache),
    }


CHANGE_DETECT_IMAGE_DIR = Path(
    os.environ.get("SATQUERY_CHANGE_DETECT_IMAGE_DIR", str(SERVICE_ROOT / "images"))
)
# BIT-LEVIR-CD is trained specifically on building change (LEVIR-CD); it has
# no mechanism to distinguish construction from vegetation_loss or removal,
# so every region is reported as the one type the model actually detects.
DEFAULT_CHANGE_TYPE = "construction"


class ContractChangeDetectRequest(BaseModel):
    image_id_t1: str
    image_id_t2: str


def _resolve_contract_image_path(image_id: str) -> str | None:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        candidate = CHANGE_DETECT_IMAGE_DIR / f"{image_id}{ext}"
        if candidate.is_file():
            return str(candidate)
    return None


def _bbox_to_geojson_polygon(bbox_latlon: list[float]) -> dict:
    min_lon, min_lat, max_lon, max_lat = bbox_latlon
    return {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat],
            [max_lon, min_lat],
            [max_lon, max_lat],
            [min_lon, max_lat],
            [min_lon, min_lat],
        ]],
    }


def _contract_error(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "error", "error": {"code": code, "message": message}})


@app.post("/change_detect")
def change_detect_contract(request: ContractChangeDetectRequest):
    """Adapter for the team's binding /change_detect contract
    (docs/json-contracts-v2.md Sec 3.3): {image_id_t1, image_id_t2} ->
    {regions: [{type, geometry, confidence}], changed_area_km2, change_mask_id}.

    Reuses the same detect_change/analyse_mask pipeline as /change-detection;
    this endpoint only resolves image_ids to files (from controller/images/,
    the controller's own manifest-backed store) and reshapes the output.
    Requires georeferenced inputs, since geometry must be real EPSG:4326 --
    a plain PNG with no CRS is rejected rather than silently returning
    pixel coordinates mislabeled as lon/lat.
    """
    before_path = _resolve_contract_image_path(request.image_id_t1)
    after_path = _resolve_contract_image_path(request.image_id_t2)
    if before_path is None:
        return _contract_error("IMAGE_NOT_FOUND", f"no image for image_id_t1={request.image_id_t1!r}")
    if after_path is None:
        return _contract_error("IMAGE_NOT_FOUND", f"no image for image_id_t2={request.image_id_t2!r}")

    try:
        detection = detect_change(before_path, after_path)
        analysis = analyse_mask(detection["probability"], after_path)
    except Exception as exc:
        return _contract_error("MODEL_FAILED", str(exc))

    if not analysis["georeferenced"]:
        return _contract_error(
            "INVALID_IMAGE",
            "image pair has no CRS; /change_detect requires georeferenced inputs to produce EPSG:4326 geometry",
        )

    query_id = uuid.uuid4().hex
    _write_visualisation(query_id, after_path, detection["probability"], threshold=0.5)

    regions = [
        {
            "type": DEFAULT_CHANGE_TYPE,
            "geometry": _bbox_to_geojson_polygon(r["bbox_latlon"]),
            "confidence": r["confidence"],
        }
        for r in analysis["regions"]
    ]
    changed_area_km2 = round(sum(r["area_sq_m"] for r in analysis["regions"]) / 1_000_000, 4)

    return {
        "regions": regions,
        "changed_area_km2": changed_area_km2,
        # The rendered mask lives at /results/{change_mask_id}/change_mask.png,
        # same as the native /change-detection response's change_mask_url.
        "change_mask_id": query_id,
    }


@app.post("/change-detection")
def change_detection(request: ChangeDetectionRequest):
    parameters = {
        "threshold": request.threshold,
        "min_region_px": request.min_region_px,
        "connectivity": CONNECTIVITY,
    }

    key = _cache_key(request)
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        cached = deepcopy(hit)
        cached["execution"]["cached"] = True
        return cached

    query_id = uuid.uuid4().hex
    trace = ExecutionTrace()

    try:
        with trace.step("validate_inputs") as entry:
            entry["details"] = _validate_inputs(request.before_path, request.after_path)

        with trace.step("change_detection", model=MODEL_NAME, parameters=parameters) as entry:
            try:
                detection = detect_change(request.before_path, request.after_path)
            except Exception as exc:
                raise PipelineError("MODEL_FAILED", f"inference failed: {exc}") from exc
            entry["inference_ms"] = round(detection["inference_ms"], 1)

        probability = detection["probability"]

        with trace.step("mask_analysis") as entry:
            try:
                analysis = analyse_mask(
                    probability,
                    request.after_path,
                    threshold=request.threshold,
                    min_region_px=request.min_region_px,
                )
            except Exception as exc:
                raise PipelineError("MODEL_FAILED", f"mask analysis failed: {exc}") from exc
            entry["regions_detected"] = analysis["region_count"]

        with trace.step("change_description") as entry:
            try:
                if request.query:
                    answered = answer_question(analysis, request.query)
                    description = answered["answer"]
                    entry["mode"] = "query"
                    entry["intent"] = answered["intent"]
                    entry["supported"] = answered["supported"]
                else:
                    answered = None
                    description = describe_change(
                        analysis, request.before_date, request.after_date
                    )
                    entry["mode"] = "description"
            except Exception as exc:
                raise PipelineError("MODEL_FAILED", f"description failed: {exc}") from exc

        with trace.step("render_visualization") as entry:
            try:
                urls = _write_visualisation(
                    query_id, request.after_path, probability, request.threshold
                )
            except Exception as exc:
                raise PipelineError("MODEL_FAILED", f"could not write results: {exc}") from exc
            entry["output_dir"] = f"/results/{query_id}"

    except PipelineError as error:
        return _error_response(query_id, trace, error, parameters)

    total_ms = trace.total_ms()

    result = {
        "description": description,
        "change_percentage": analysis["change_percentage"],
        "region_count": analysis["region_count"],
        "regions": analysis["regions"],
        "georeferenced": analysis["georeferenced"],
    }
    if answered is not None:
        result["query"] = request.query
        result["intent"] = answered["intent"]
        result["supported"] = answered["supported"]

    response = {
        "query_id": query_id,
        "status": "success",
        "analysis": {"intent": "change_detection", "workflow": WORKFLOW},
        "result": result,
        "visualization": {
            "type": "temporal_comparison",
            "before_image": request.before_path,
            "after_image": request.after_path,
            **urls,
        },
        "execution": {
            "steps": trace.steps,
            "model": MODEL_NAME,
            "parameters": parameters,
            "total_ms": total_ms,
            "cached": False,
        },
        "statistics": {
            "regions_detected": analysis["region_count"],
            "processing_time_ms": total_ms,
        },
    }

    with _cache_lock:
        _cache[key] = deepcopy(response)

    return response
