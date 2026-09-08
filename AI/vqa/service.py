import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from vqa_model import VQAModel

MODEL_PATH = os.environ.get("SATQUERY_VQA_MODEL_PATH", "./checkpoints/geochat-7B")
MODEL_BASE = os.environ.get("SATQUERY_VQA_MODEL_BASE")
IMAGE_DIR = os.environ.get("SATQUERY_VQA_IMAGE_DIR", "./data_store/s2_png")

_model: VQAModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = VQAModel(model_path=MODEL_PATH, model_base=MODEL_BASE)
    yield


app = FastAPI(title="SatQuery VQA service", lifespan=lifespan)


class VQARequest(BaseModel):
    image_id: str
    question: str


def _error(code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=200, content={"status": "error", "error": {"code": code, "message": message}})


def _resolve_image_path(image_id: str) -> str | None:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = os.path.join(IMAGE_DIR, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


@app.post("/vqa")
def vqa(request: VQARequest):
    image_path = _resolve_image_path(request.image_id)
    if image_path is None:
        return _error("IMAGE_NOT_FOUND", f"no image for image_id={request.image_id!r} under {IMAGE_DIR}")
    try:
        result = _model.answer(image_path, request.question)
    except Exception as exc:
        return _error("MODEL_FAILED", str(exc))
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8011)
