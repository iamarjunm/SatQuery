"""Direct inference wrapper around the vendored BIT_CD change-detection model.

Bypasses BIT_CD's dataset/dataloader machinery: takes two image paths, returns
a per-pixel change probability map at the input images' original resolution.
Nothing in ``vendor/`` is modified.

The model is loaded once into a module-level singleton and reused across calls.

Preprocessing mirrors ``datasets/data_utils.py::CDDataAugmentation.transform``
as instantiated for evaluation (``ImageDataset`` with ``is_train=False``, i.e.
``CDDataAugmentation(img_size=256)`` with every augmentation flag off):

    RGB -> bicubic resize to 256x256 (skipped if already 256x256)
        -> to_tensor (scales to [0, 1])
        -> normalize(mean=0.5, std=0.5) (shifts to [-1, 1])

Using different normalization would silently degrade output rather than error,
so keep this in step with the vendor code if it is ever updated.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "BIT_CD"
CHECKPOINT_PATH = VENDOR_ROOT / "checkpoints" / "BIT_LEVIR" / "best_ckpt.pt"

MODEL_NAME = "BIT-LEVIR-CD"

# Must match the training configuration. IMG_SIZE is demo.py's --img_size
# default; the normalization constants are hardcoded in data_utils.py.
IMG_SIZE = 256
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)

# Kwargs for the 'base_transformer_pos_s4_dd8_dedim8' branch of
# models/networks.py::define_G -- demo.py's default --net_G, and the
# architecture the BIT_LEVIR checkpoint was trained with.
NET_G_KWARGS = dict(
    input_nc=3,
    output_nc=2,
    token_len=4,
    resnet_stages_num=4,
    with_pos="learned",
    enc_depth=1,
    dec_depth=8,
    decoder_dim_head=8,
)

_model: torch.nn.Module | None = None
_device: torch.device | None = None
_load_lock = threading.Lock()


def _import_vendor():
    """Import BIT_CD's network module by putting its root on sys.path.

    BIT_CD uses absolute imports rooted at its own directory (``import
    models``, ``from misc.imutils import ...``), so the path entry is required
    and cannot be scoped to this module. Note the side effect: it makes the
    vendor's generic top-level names -- ``models``, ``utils``, ``datasets``,
    ``misc`` -- importable process-wide. Avoid those names for our own modules
    in this service, or the vendor tree will shadow them.
    """
    if not VENDOR_ROOT.is_dir():
        raise FileNotFoundError(
            f"vendor tree not found at {VENDOR_ROOT}. "
            "Clone it and run fix_vendor.py -- see setup.md."
        )

    vendor_path = str(VENDOR_ROOT)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)

    try:
        from models.networks import BASE_Transformer
    except ImportError as exc:
        raise ImportError(
            f"could not import BIT_CD from {VENDOR_ROOT} ({exc}). "
            "The vendor tree likely needs patching: run fix_vendor.py."
        ) from exc

    return BASE_Transformer


def get_device() -> torch.device:
    """CUDA if available, else CPU."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_model() -> tuple[torch.nn.Module, torch.device]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"checkpoint not found at {CHECKPOINT_PATH}. See setup.md step 6."
        )

    base_transformer = _import_vendor()
    device = get_device()

    # Constructing this pulls ImageNet resnet18 weights via torch.hub
    # (models/networks.py hardcodes pretrained=True). They are immediately
    # overwritten by the checkpoint below, but the first run still needs
    # network access unless the torch hub cache is already populated.
    net = base_transformer(**NET_G_KWARGS)

    # weights_only=False is required: the checkpoint is a dict carrying
    # optimizer/epoch state alongside the weights. Passing it explicitly
    # silences the FutureWarning and keeps behavior stable when torch flips
    # the default.
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint["model_G_state_dict"])

    net.to(device)
    net.eval()
    return net, device


def get_model() -> tuple[torch.nn.Module, torch.device]:
    """Return the singleton (model, device), loading it on first call.

    Double-checked locking keeps a cold start from loading the model several
    times if concurrent requests arrive before the first load finishes.
    """
    global _model, _device
    if _model is None:
        with _load_lock:
            if _model is None:
                _model, _device = _build_model()
    return _model, _device


def _load_rgb(path: str | Path) -> Image.Image:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as handle:
        return handle.convert("RGB")


def _to_model_tensor(img: Image.Image) -> torch.Tensor:
    """Apply the vendor's evaluation transform to one IMG_SIZE-or-smaller crop.

    Mirrors ``datasets/data_utils.py::CDDataAugmentation.transform`` as
    instantiated for evaluation: bicubic resize to 256x256 (skipped if
    already that size) -> to_tensor -> normalize(mean=0.5, std=0.5).
    """
    if img.size != (IMG_SIZE, IMG_SIZE):
        # interpolation=3 in the vendor source is PIL's BICUBIC enum value.
        img = TF.resize(img, [IMG_SIZE, IMG_SIZE], interpolation=InterpolationMode.BICUBIC)
    tensor = TF.to_tensor(img)
    return TF.normalize(tensor, NORM_MEAN, NORM_STD)


def _run_tile(model: torch.nn.Module, device: torch.device, before: torch.Tensor, after: torch.Tensor) -> np.ndarray:
    batch_a = before.unsqueeze(0).to(device)
    batch_b = after.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(batch_a, batch_b)  # [1, 2, 256, 256], raw scores
        # Class 1 is "changed"; the net has output_sigmoid=False, so these are
        # unnormalized logits and softmax is the correct conversion.
        probability = torch.softmax(logits, dim=1)[:, 1]
    return probability[0].cpu().numpy().astype(np.float32)


def _tile_starts(length: int, tile: int) -> list[int]:
    """Starting offsets covering [0, length) with tiles of size `tile`.

    Evenly spaced when `length` is a multiple of `tile`. Otherwise the final
    tile is pulled back flush with the far edge instead of padded, so every
    tile stays full native resolution and no synthetic border is introduced
    -- the overlap this creates with the second-to-last tile is resolved by
    averaging in detect_change.
    """
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def detect_change(before_path: str | Path, after_path: str | Path) -> dict:
    """Run change detection on a co-registered image pair.

    BIT_CD was trained on 256x256 crops (IMG_SIZE); resizing a larger scene
    straight down to that size before inference -- rather than tiling it --
    discards most of the spatial detail the model needs and silently produces
    near-zero change on real (e.g. 1024x1024 LEVIR-CD test) imagery. Images at
    or under IMG_SIZE in both dimensions run in a single shot, matching the
    original single-tile behaviour exactly; larger images are tiled at native
    IMG_SIZE resolution and the (possibly overlapping) tile predictions are
    averaged back into one full-resolution probability map.

    Args:
        before_path: earlier image.
        after_path: later image. Must have the same dimensions as `before_path`.

    Returns a dict with:
        probability:   float32 HxW array in [0, 1] -- P(pixel changed).
        original_size: (width, height) of the input images.
        model_name:    "BIT-LEVIR-CD".
        inference_ms:  wall-clock milliseconds for the forward passes only,
                       excluding image loading and stitching.
    """
    model, device = get_model()

    before_img = _load_rgb(before_path)
    after_img = _load_rgb(after_path)

    before_size = before_img.size
    after_size = after_img.size
    if before_size != after_size:
        raise ValueError(
            f"image pair must be the same size, got {before_size} "
            f"(before) and {after_size} (after)"
        )

    width, height = before_size

    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    if width <= IMG_SIZE and height <= IMG_SIZE:
        before_t = _to_model_tensor(before_img)
        after_t = _to_model_tensor(after_img)
        tile_prob = _run_tile(model, device, before_t, after_t)
        if tile_prob.shape != (height, width):
            resized = F.interpolate(
                torch.from_numpy(tile_prob).unsqueeze(0).unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            probability = resized[0, 0].numpy()
        else:
            probability = tile_prob
    else:
        prob_sum = np.zeros((height, width), dtype=np.float32)
        prob_count = np.zeros((height, width), dtype=np.float32)
        for y0 in _tile_starts(height, IMG_SIZE):
            for x0 in _tile_starts(width, IMG_SIZE):
                before_crop = before_img.crop((x0, y0, x0 + IMG_SIZE, y0 + IMG_SIZE))
                after_crop = after_img.crop((x0, y0, x0 + IMG_SIZE, y0 + IMG_SIZE))
                tile_prob = _run_tile(
                    model, device, _to_model_tensor(before_crop), _to_model_tensor(after_crop)
                )
                prob_sum[y0 : y0 + IMG_SIZE, x0 : x0 + IMG_SIZE] += tile_prob
                prob_count[y0 : y0 + IMG_SIZE, x0 : x0 + IMG_SIZE] += 1.0
        probability = prob_sum / prob_count

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - started) * 1000.0

    return {
        "probability": probability,
        "original_size": before_size,
        "model_name": MODEL_NAME,
        "inference_ms": inference_ms,
    }
