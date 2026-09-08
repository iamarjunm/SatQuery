# change_detection — setup

Change-detection service. Wraps [BIT_CD](https://github.com/justchenhao/BIT_CD)
(Chen et al., *Remote Sensing Image Change Detection with Transformers*) as a
vendored dependency and exposes it over FastAPI. This document covers getting
it running from a clean checkout; see **Running the service** at the end.

## Prerequisites

- **Python 3.11.** Not 3.12+, and specifically not 3.14: torch, rasterio and
  opencv-python have no wheels for it, so `pip install` either fails or falls
  back to a source build. Check with `py --list`.
- **Git.**
- **NVIDIA driver** supporting CUDA 12.1 or newer, if you want GPU inference.
  Verify with `nvidia-smi`. CPU-only works but is roughly 20× slower.

## Setup

### 1. Create the virtualenv

```bash
cd services/change_detection && py -3.11 -m venv .venv
```

All commands below assume the venv's interpreter. On Windows that is
`.venv/Scripts/python.exe`; on Linux/macOS, `.venv/bin/python`. Activate it
instead if you prefer:

```bash
source .venv/Scripts/activate
```

### 2. Install torch with CUDA support

Do this **before** `requirements.txt`. Plain `pip install torch` pulls the
CPU-only build from PyPI, and pip will then consider the requirement satisfied
and never replace it.

```bash
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Pick the index to match your driver — `cu121`, `cu124`, `cu126` — from
[pytorch.org/get-started](https://pytorch.org/get-started/locally/). This
project is verified on `cu121`. For CPU-only, skip this step entirely and let
step 3 install the PyPI build.

Confirm the GPU is visible:

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.5.1+cu121 True NVIDIA GeForce GTX 1650`

### 3. Install the remaining dependencies

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Note that `requirements.txt` carries five packages upstream never declared —
`matplotlib`, `imageio`, `scikit-image`, `scipy`, `tifffile`. BIT_CD imports
them from `misc/imutils.py`, `misc/torchutils.py` and the dataset code, so
without them `demo.py` dies at import time.

### 4. Clone the vendor model

```bash
git clone https://github.com/justchenhao/BIT_CD.git vendor/BIT_CD
```

`vendor/` is gitignored — it stays an independent clone rather than getting
embedded in this repo's history.

### 5. Patch the vendor tree

Upstream was written against ~2021 dependencies and will not import as-is.
`fix_vendor.py` applies the two required source fixes:

```bash
.venv/Scripts/python.exe fix_vendor.py
```

```
applied 3 fix(es):
  models\resnet.py: torchvision.models.utils -> torch.hub
  datasets\CD_dataset.py: np.str -> str
  misc\torchutils.py: np.float -> float
  checkpoint present: checkpoints\BIT_LEVIR\best_ckpt.pt (57 MB)
```

What it changes and why:

| Fix | Reason |
|---|---|
| `from torchvision.models.utils import load_state_dict_from_url` → `from torch.hub import ...` | `torchvision.models.utils` was a private module, removed in torchvision 0.13. |
| `np.str` → `str`, `np.float` → `float` | Aliases for builtins, deprecated in NumPy 1.20 and removed in 1.24. They raise `AttributeError` on any current NumPy. |

The script also handles `np.int`, `np.bool`, `np.object`, `np.complex`,
`np.long` and `np.unicode` if a future upstream revision introduces them. Sized
dtypes (`np.float32`, `np.uint8`, `np.bool_`) are deliberately left alone —
those are still valid.

It is idempotent, so re-run it freely, including after a `git pull` in the
vendor tree. To see what it *would* do without writing anything:

```bash
.venv/Scripts/python.exe fix_vendor.py --check
```

That exits 1 when fixes are outstanding, which makes it usable as a CI or
pre-flight gate. To revert all patches, run `git checkout .` inside
`vendor/BIT_CD`.

### 6. The checkpoint

**No download needed.** The pretrained LEVIR-CD checkpoint is committed to the
upstream repo, so a fresh clone already has it at:

```
vendor/BIT_CD/checkpoints/BIT_LEVIR/best_ckpt.pt   (57 MB)
```

The upstream README tells you to fetch it from Baidu or Google Drive; that file
is byte-identical to the committed one (verified by md5), so the step is
redundant. `fix_vendor.py` reports the checkpoint's presence on every run.

If it *is* missing — a partial clone, or a git-lfs configuration that skipped
it — download from the [Google Drive link](https://drive.google.com/file/d/1IVdF5a3e1_7DiSndtMkhpZuCSgDLLFcg/view?usp=sharing)
in the vendor README and place it at exactly that path. The directory name
`BIT_LEVIR` matters: it is `demo.py`'s default `--project_name`, and the
checkpoint path is built as `checkpoint_root/project_name/checkpoint_name`.

For your own trained checkpoints, add a sibling directory under `checkpoints/`
and pass `--project_name <dirname>`.

### 7. Verify

Run the upstream demo against the 7 bundled sample image pairs:

```bash
cd vendor/BIT_CD && ../../.venv/Scripts/python.exe demo.py --gpu_ids 0
```

Use `--gpu_ids -1` for CPU. Expected output:

```
initialize network with normal
cuda:0
process: [np.str_('test_77_0512_0256.png')]
...
```

Seven binary change masks land in `vendor/BIT_CD/samples/predict/`. Add
`--output_folder <path>` to write them elsewhere and keep the vendor tree
clean.

## Known warnings

Both are harmless and expected — no action needed:

- `FutureWarning: You are using torch.load with weights_only=False` — upstream
  loads the checkpoint with the legacy pickle path. Fine for a checkpoint you
  trust; it will need addressing when torch flips the default.
- `SyntaxWarning: "is" with a literal` in `models/networks.py:297,299` —
  upstream compares strings with `is`. It works today only because CPython
  interns these particular literals. Worth fixing if we ever pass `pool_mode`
  from a config file or CLI argument, where the string would not be interned
  and the comparison would silently fail.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: torchvision.models.utils` | Step 5 not run. |
| `AttributeError: module 'numpy' has no attribute 'str'` | Step 5 not run, or the vendor tree was re-cloned since. |
| `torch.cuda.is_available()` is `False` | CPU wheel installed. Uninstall torch/torchvision, redo step 2 before step 3. |
| `ERROR: Could not find a version that satisfies torch` | Running on Python 3.14. Rebuild the venv with `py -3.11`. |
| `FileNotFoundError: ...best_ckpt.pt` | See step 6. |
| CUDA out of memory | The GTX 1650 has 4 GB. Keep `--batch_size 1` for inference at 256×256. |

## Running the service

```bash
.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
```

The model is warmed during startup, so the first request does not pay the
~2.2 s CUDA cold start. `GET /health` reports the device and whether the model
is resident. Mask and overlay PNGs are written to `results/{query_id}/` and
served at `/results/...`; that directory is generated output and is gitignored.

Responses are cached on `(before_path, after_path, threshold, min_region_px,
query)`, so a repeated request is replayed from memory and never re-enters the
GPU. The cache is in-process: it does not survive a restart.

Tests:

```bash
.venv/Scripts/python.exe test_change_model.py
```

```bash
.venv/Scripts/python.exe test_api.py
```

## `/change_detect`: the team's binding contract

`main.py` also exposes `docs/json-contracts-v2.md` Sec 3.3's actual contract
alongside the native `/change-detection` endpoint above (kept for its own
tests): `POST /change_detect` with `{"image_id_t1", "image_id_t2"}`, returning
`{"regions": [{"type", "geometry", "confidence"}], "changed_area_km2",
"change_mask_id"}`. `image_id` resolves to
`<SATQUERY_CHANGE_DETECT_IMAGE_DIR>/<image_id>.<ext>` (tif/tiff/png/jpg/jpeg
tried in that order) — point it at the controller's own `controller/images/`
to use its real (georeferenced) chips directly:

```bash
SATQUERY_CHANGE_DETECT_IMAGE_DIR=../../controller/images \
.venv/Scripts/python.exe -m uvicorn main:app --port 8000
```

This endpoint requires **georeferenced inputs** — a plain PNG with no CRS
(like the LEVIR-CD/BIT_CD sample imagery used above) cannot produce real
EPSG:4326 geometry and gets `INVALID_IMAGE` rather than a silently-wrong
pixel-as-lon/lat answer. Every region is reported as `type: "construction"`:
BIT-LEVIR-CD is trained specifically on building change (LEVIR-CD) and has no
mechanism to distinguish that from vegetation loss or removal.

Verified against real controller imagery (`controller/images/`, Sentinel-2,
10 m/px): correctly found ~26,000 sqm of new construction at the Jewar
airport site (2023 to 2026 pair) with 0.816 confidence. Found nothing at a
port site over the same window — plausibly a real result (LEVIR-CD trains on
~0.5 m/px imagery, so building-scale change can be below what 10 m/px
resolves; large clear changes like a new airport still register).

## Fixed: full-resolution images silently produced near-zero change

`change_model.py`'s `detect_change` used to resize *any* input straight to
256x256 (`IMG_SIZE`, what BIT_CD was trained on) before inference. That is
correct for already-256x256 imagery (the bundled BIT_CD samples above), but a
real 1024x1024 LEVIR-CD test scene downsampled 4x this way loses most of the
detail the model needs — verified against `LEVIR-CD/test/`: a scene with
17.24% ground-truth change scored 0.0% detected before this fix. Large images
are now tiled into native-resolution 256x256 crops, run individually, and
stitched back with overlap-averaging (`_tile_starts`) — the same scene now
scores 17.03%. Images at or under 256x256 still take the original single-shot
path unchanged.

## Verified configuration

| | |
|---|---|
| Python | 3.11.9 |
| torch / torchvision | 2.14.0+cu130 / matching |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB) |
| Date | 2026-09-08 |

Originally verified on Python 3.11.0 / torch 2.5.1+cu121 / GTX 1650 (4GB) --
both configurations work; match the CUDA index to your own driver (step 2).
