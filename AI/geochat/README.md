# GeoChat agent

The VQA and grounding services, backed by [GeoChat-7B](https://github.com/mbzuai-oryx/GeoChat):
a LLaVA-1.5 vision-language model fine-tuned on remote-sensing imagery, Apache 2.0,
pre-trained, nothing to train. It answers questions about a chip and returns rotated
boxes for a phrase. It does not do change detection or SAR; those stay on the
controller's mock service until their own agents exist.

`service.py` wraps it in the team contract (`POST /vqa`, `POST /ground`, envelope
`{"status": "success", "result": ...}`, GeoJSON in EPSG:4326). The controller never
imports it; it only sets two URLs.

## On a laptop, no GPU

```
cd AI/geochat
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python tests.py          # parsing, geometry, contract, controller round-trip
.venv/Scripts/python service.py        # stub backend on ports 8001 (vqa) and 8002 (ground)
```

Then in `controller/.env`:

```
SATQUERY_MOCK_VQA=0
SATQUERY_VQA_URL=http://127.0.0.1:8001/vqa
SATQUERY_MOCK_GROUND=0
SATQUERY_GROUND_URL=http://127.0.0.1:8002/ground
```

and `python e2e.py` in the controller runs real plans against the stub agent. The stub
answers with canned text and two boxes, one rotated, so the whole path except the model
is exercised: the contract, the pixel-to-lon/lat conversion, the footprint check, and
the executor.

## On Kaggle, real model

One notebook, GPU on (one T4 is enough in 4-bit), internet on. Cells:

```
!git clone https://github.com/mbzuai-oryx/GeoChat && pip install -q -e GeoChat
!pip install -q rasterio pyngrok
!huggingface-cli download MBZUAI/geochat-7B --local-dir /kaggle/working/geochat-7B
```

Upload `controller/images` (the chips plus `manifest.json`) as a Kaggle dataset and
`AI/geochat/service.py` alongside it, then:

```
!nohup python service.py --backend geochat --model-path /kaggle/working/geochat-7B \
    --load-4bit --images /kaggle/input/satquery-images --port 8000 > agent.log 2>&1 &
```

Check `!curl localhost:8000/health`, then open the tunnel (free ngrok account, one-time
token):

```
from pyngrok import ngrok
ngrok.set_auth_token("...")
print(ngrok.connect(8000).public_url)
```

Put that URL in `controller/.env`:

```
SATQUERY_MOCK_VQA=0
SATQUERY_VQA_URL=https://<tunnel>/vqa
SATQUERY_MOCK_GROUND=0
SATQUERY_GROUND_URL=https://<tunnel>/ground
```

The URL changes every Kaggle session. Sessions cap at 12 hours.

## What to expect on the first GPU run

`GeoChatBackend` mirrors the repo's `geochat/eval/batch_geochat_grounding.py`
(llava_v1 template, 504 px preprocessing, greedy decoding) but was written against the
code, not run on a GPU here. The first run is where any drift shows up; the two places
to look are the decode slice in `infer` (older transformers return the prompt tokens
too) and the image-token constant names. Test by hand before wiring the controller:

```
curl -X POST localhost:8000/vqa    -d '{"image_id":"udaipur_2026_opt","question":"What is in this image?"}'
curl -X POST localhost:8000/ground -d '{"image_id":"jnpt_port_2026_opt","phrase":"ships"}'
```

The `ground` result carries the model's raw text in `raw`, so a parsing miss is visible.

## Decisions written down

- **Confidence.** GeoChat produces none. The service attaches a fixed value per tool,
  0.70 for VQA and 0.60 for grounding, overridable with `GEOCHAT_VQA_CONFIDENCE` and
  `GEOCHAT_GROUND_CONFIDENCE`. The controller takes the minimum across a chain and
  abstains below 0.5, so these are a policy about how much to trust the model, not a
  measurement. Say so if a judge asks.
- **Rotation.** GeoChat's angle is applied about the box centre in image axes, then
  corners are clamped to the image. If boxes look mirrored on the map, flip the sign
  of `theta` in `box_to_pixel_corners`; the footprint check passes either way.
- **Chip size.** The chips are 500 to 1100 px; GeoChat sees them resized to 504 px.
  Ships in the Singapore chip are a few pixels wide at that scale, so counting there
  is the stress test. Tiling is the fix if it fails, and belongs in this file.
- **One request at a time.** Generation holds a lock; concurrent controller calls
  queue on the GPU rather than fight for it.
