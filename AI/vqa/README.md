# VQA service (P2)

Implements the binding `/vqa` contract from `docs/json-contracts-v2.md` §3.1 /
`controller/tools.py`'s `TOOLS["vqa"]`:

```
POST /vqa
in : {"image_id": str, "question": str}
out: {"answer": str, "confidence": float}
```

Base model: [GeoChat-7B](https://github.com/mbzuai-oryx/GeoChat) (LLaVA-1.5
architecture, remote-sensing adapted), loaded 4-bit via bitsandbytes and
QLoRA-fine-tuned locally on a BigEarthNet.txt (arXiv:2603.29630) subset —
this is what satisfies the hackathon's mandatory remote-sensing-adaptation
requirement; `AI/geochat`'s `geochat` backend loads the same base checkpoint
but performs no fine-tuning of its own.

**Port note:** this service listens on **8011**, not 8001, because
`AI/geochat/service.py` already claims 8001 for its own `/vqa` (three
backends: stub, Gemini, pretrained-only GeoChat). Both can run side by side
until the team decides which is canonical — set `SATQUERY_VQA_URL`
accordingly for whichever one you're pointing the controller at.

## Setup

1. **Python deps** (a CUDA-enabled PyTorch build matching your GPU/driver is
   important — the generic `pip install torch` may give you a CPU-only build):

   ```bash
   pip install -r requirements.txt
   ```

2. **Clone GeoChat and apply the compatibility patch.** GeoChat's code predates
   transformers 5.x by about two years; `geochat_transformers5_compat.patch`
   fixes real incompatibilities found by actually running it end to end
   (inference *and* fine-tuning) — an unused MPT backend import that no
   longer builds, a deprecated quantization kwarg, a vision-tower loader
   that broke under meta-device model construction, missing
   `cache_position`/`position_ids` resync during generation (silently
   produced coherent-looking but *content-free* garbage without ever
   raising), stale `Trainer`/sampler/checkpoint APIs, and three more hit
   specifically when loading a LoRA checkpoint (device_map re-dispatch,
   a stale `quantization_config` carried over from the training run, and
   dtype drift after `merge_and_unload()`):

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

   Then from the controller side, set
   `SATQUERY_VQA_URL=http://127.0.0.1:8011/vqa` (the contract's default of
   8001 points at `AI/geochat` instead) and `SATQUERY_MOCK_VQA=0` to stop
   using the mock adapter.

## Smoke test

```bash
curl -s -X POST http://127.0.0.1:8011/vqa -H "Content-Type: application/json" \
  -d '{"image_id":"04133","question":"What type of land cover dominates this image?"}'
# {"answer":"The type of land cover that dominates this image is baseball fields.","confidence":0.897...}
```

## Loading a fine-tuned checkpoint

`geochat/mm_utils.py`'s `get_model_name_from_path()` decides which loading
branch runs based on whether the checkpoint directory's *name* contains the
substring `"geochat"` — a directory named e.g. `vqa-lora` silently takes the
wrong (generic, non-quantized, non-multimodal) code path and fails with
confusing offload errors that have nothing to do with the real problem. Name
LoRA output/checkpoint directories so they contain `geochat`, e.g.
`geochat-vqa-lora`.

## Fine-tuning results (QLoRA on BigEarthNet.txt, capped to fit ~1hr/epoch)

389 steps (6,200 examples, effective batch 16, r=64/alpha=16) on the local
12GB GPU. Verified the LoRA weights are real (non-zero, ~91% of the target
layer's dequantized values changed from base) and do shift generation on
longer, open-ended answers. On this eval set's mostly short yes/no/mcq
questions, though, greedy decoding picked the same argmax token as the base
model for most examples — measurable weight change, but not yet a measurable
accuracy change at this scale/schedule. Contains-accuracy (reference answer
found anywhere in the free-text response) on 500 held-out examples: base
31.8%, tuned 31.8%. Longer training / higher LoRA alpha / more data would be
the next lever, not a fix to a code issue — the loading and eval pipeline
itself is verified working.

## Environment variables

| variable | default | effect |
|---|---|---|
| `SATQUERY_VQA_MODEL_PATH` | `./checkpoints/geochat-7B` | base checkpoint dir, or a LoRA adapter dir (see note below) |
| `SATQUERY_VQA_MODEL_BASE` | unset | set this to the base checkpoint dir *only* when `MODEL_PATH` is a LoRA adapter |
| `SATQUERY_VQA_IMAGE_DIR` | `./data_store/s2_png` | where `image_id` files live |

## Files

| file | what |
|---|---|
| `vqa_model.py` | `VQAModel` class: loads GeoChat, `answer(image_path, question) -> {"answer", "confidence"}` |
| `service.py` | FastAPI wrapper exposing `POST /vqa` on port 8011 (see port note above) |
| `geochat_transformers5_compat.patch` | apply to a fresh GeoChat clone before `pip install -e .` |
