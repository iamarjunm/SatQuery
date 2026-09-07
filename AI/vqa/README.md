# VQA service (P2)

Implements the binding `/vqa` contract from `docs/json-contracts-v2.md` §3.1 /
`controller/tools.py`'s `TOOLS["vqa"]`:

```
POST /vqa
in : {"image_id": str, "question": str}
out: {"answer": str, "confidence": float}
```

Base model: [GeoChat-7B](https://github.com/mbzuai-oryx/GeoChat) (LLaVA-1.5
architecture, remote-sensing adapted), loaded 4-bit (QLoRA-compatible) via
bitsandbytes. A QLoRA fine-tune on BigEarthNet.txt is in progress; until that
lands, this serves the base pretrained checkpoint, which already answers real
questions about remote-sensing imagery (verified working end-to-end locally).

## Setup

1. **Python deps** (a CUDA-enabled PyTorch build matching your GPU/driver is
   important — the generic `pip install torch` may give you a CPU-only build):

   ```bash
   pip install -r requirements.txt
   ```

2. **Clone GeoChat and apply the compatibility patch.** GeoChat's code predates
   transformers 5.x by about two years; `geochat_transformers5_compat.patch`
   fixes four real incompatibilities found by actually running it (an unused
   MPT backend import that no longer builds, a deprecated quantization kwarg,
   a vision-tower loader that broke under meta-device model construction, and
   — the one that actually mattered for output quality — missing
   `cache_position`/`position_ids` resync during generation, which silently
   produced coherent-looking but *content-free* garbage without ever raising
   an error):

   ```bash
   git clone https://github.com/mbzuai-oryx/GeoChat.git
   cd GeoChat
   git apply ../AI/vqa/geochat_transformers5_compat.patch
   pip install -e .
   cd ..
   ```

3. **Download the checkpoint** (~14GB):

   ```bash
   python -c "from huggingface_hub import snapshot_download; snapshot_download('MBZUAI/geochat-7B', local_dir='./checkpoints/geochat-7B')"
   ```

4. **Set up an image store.** `image_id` resolves to
   `<SATQUERY_VQA_IMAGE_DIR>/<image_id>.<ext>` (png/jpg/jpeg/tif/tiff tried in
   that order). Point it at wherever your session's images actually live —
   for a quick smoke test, GeoChat's own `demo_images/` folder works.

5. **Run it:**

   ```bash
   SATQUERY_VQA_MODEL_PATH=./checkpoints/geochat-7B \
   SATQUERY_VQA_IMAGE_DIR=./GeoChat/demo_images \
   python AI/vqa/service.py
   ```

   Then from the controller side, point `SATQUERY_VQA_URL` at
   `http://127.0.0.1:8001/vqa` (its default already assumes this) and set
   `SATQUERY_MOCK_VQA=0` to stop using the mock adapter.

## Smoke test

```bash
curl -s -X POST http://127.0.0.1:8001/vqa -H "Content-Type: application/json" \
  -d '{"image_id":"04133","question":"What type of land cover dominates this image?"}'
# {"answer":"The type of land cover that dominates this image is baseball fields.","confidence":0.897...}
```

## Environment variables

| variable | default | effect |
|---|---|---|
| `SATQUERY_VQA_MODEL_PATH` | `./checkpoints/geochat-7B` | base checkpoint dir, or a LoRA adapter dir once fine-tuning lands |
| `SATQUERY_VQA_MODEL_BASE` | unset | set this to the base checkpoint dir *only* when `MODEL_PATH` is a LoRA adapter |
| `SATQUERY_VQA_IMAGE_DIR` | `./data_store/s2_png` | where `image_id` files live |

## Files

| file | what |
|---|---|
| `vqa_model.py` | `VQAModel` class: loads GeoChat, `answer(image_path, question) -> {"answer", "confidence"}` |
| `service.py` | FastAPI wrapper exposing `POST /vqa` on port 8001 per the contract |
| `geochat_transformers5_compat.patch` | apply to a fresh GeoChat clone before `pip install -e .` |
