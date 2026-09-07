"""HTTP service exposing the VQA model per the team's binding contract
(docs/json-contracts-v2.md §3.1, controller/tools.py TOOLS["vqa"]):

    POST /vqa
    in : {"image_id": str, "question": str}
    out: {"answer": str, "confidence": float}

The controller never imports a model -- it only ever speaks HTTP to this
service. Default port 8001 matches SATQUERY_VQA_URL's default
(http://127.0.0.1:8001/vqa) in the controller's tool registry.

See README.md in this directory for setup (GeoChat clone + patch, checkpoint,
image store).
"""
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from vqa_model import VQAModel

MODEL_PATH = os.environ.get("SATQUERY_VQA_MODEL_PATH", "./checkpoints/geochat-7B")
MODEL_BASE = os.environ.get("SATQUERY_VQA_MODEL_BASE")  # set when MODEL_PATH is a LoRA adapter dir
IMAGE_DIR = os.environ.get("SATQUERY_VQA_IMAGE_DIR", "./data_store/s2_png")

app = FastAPI(title="SatQuery VQA service")
_model: VQAModel | None = None


class VQARequest(BaseModel):
    image_id: str
    question: str


class VQAResponse(BaseModel):
    answer: str
    confidence: float


def _get_model() -> VQAModel:
    global _model
    if _model is None:
        _model = VQAModel(model_path=MODEL_PATH, model_base=MODEL_BASE)
    return _model


def _resolve_image_path(image_id: str) -> str:
    for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = os.path.join(IMAGE_DIR, f"{image_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    raise HTTPException(status_code=404, detail={"error": {"code": "IMAGE_NOT_FOUND", "message": f"no image for image_id={image_id!r} under {IMAGE_DIR}"}})


@app.post("/vqa", response_model=VQAResponse)
def vqa(request: VQARequest) -> VQAResponse:
    image_path = _resolve_image_path(request.image_id)
    try:
        result = _get_model().answer(image_path, request.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"error": {"code": "MODEL_FAILED", "message": str(exc)}}) from exc
    return VQAResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
