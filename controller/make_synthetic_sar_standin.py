"""Generate one synthetic SAR-shaped image, paired to Vienna's cloudy optical
scene, so the cross_modal demo path can actually run end to end.

This is NOT real SAR data. Real Sentinel-1 GRD scenes exist for this AOI and
date window (confirmed against both AWS Open Data and Microsoft Planetary
Computer's STAC catalogs), but the product as published is not georeferenced
-- it needs GCP/DEM-based terrain correction before it can be placed in
EPSG:4326, which is a different and much heavier pipeline than the simple
"crop a window from a georeferenced COG" fetch_chips.py does for Sentinel-2.
That is future work, not this script's job.

What this script does instead: reads img_cloud_opt's own real, already
georeferenced GeoTIFF (so the stand-in is genuinely co-registered with it --
same CRS, transform, size, footprint, no faking there), converts it to
grayscale, and applies multiplicative speckle noise (the actual noise model
real SAR imagery has, Goodman 1976) so it is visually distinguishable from
the optical scene it's derived from, then writes it as a new image_id with
modality "sar" and source "synthetic-sar-standin" -- a label the frontend
and manifest both surface honestly, never as "sentinel-1".

Run: python make_synthetic_sar_standin.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
SOURCE_ID = "img_cloud_opt"
STANDIN_ID = "img_cloud_sar"
RNG_SEED = 20260908  # reproducible speckle pattern run to run


def make_speckled_grayscale(rgb: np.ndarray) -> np.ndarray:
    """rgb: (bands, rows, cols) uint8 -> (rows, cols) uint16 SAR-shaped array.

    Real SAR speckle is multiplicative: observed = true_backscatter * noise,
    noise ~ Gamma(shape=L, scale=1/L) for an L-look image. Applying that to a
    luminance image gives the grainy look real SAR has, instead of a merely
    desaturated copy of the optical scene."""
    luminance = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).astype(np.float64)
    # Invert, with a floor: SAR backscatter from a cloud-covered scene is
    # dominated by the ground it can see through the cloud, not the cloud
    # itself, so a bright (cloudy) optical pixel should not simply map to a
    # bright SAR pixel. The floor keeps a visible (if dim) speckle texture
    # everywhere instead of a near-black frame under the near-white cloud
    # that covers most of this scene -- multiplying speckle by ~0 would
    # otherwise erase it.
    backscatter = 30.0 + (255.0 - luminance) * 0.85
    rng = np.random.default_rng(RNG_SEED)
    looks = 4.0
    speckle = rng.gamma(shape=looks, scale=1.0 / looks, size=backscatter.shape)
    speckled = backscatter * speckle
    stretched = np.clip(speckled / max(speckled.max(), 1e-6) * 65535, 0, 65535)
    return stretched.astype(np.uint16)


def main() -> int:
    manifest_path = IMAGES / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_entry = manifest[SOURCE_ID]

    src_tif = IMAGES / source_entry["file"]
    with rasterio.open(src_tif) as src:
        rgb = src.read([1, 2, 3])
        profile = src.profile.copy()
        crs, transform = src.crs, src.transform

    sar_band = make_speckled_grayscale(rgb)

    profile.update(count=1, dtype="uint16", driver="GTiff", compress="deflate")
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)

    out_tif = IMAGES / f"{STANDIN_ID}.tif"
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(sar_band, 1)

    out_png = IMAGES / f"{STANDIN_ID}.png"
    preview_8bit = (sar_band.astype(np.float64) / 65535 * 255).astype(np.uint8)
    Image.fromarray(preview_8bit, mode="L").convert("RGB").save(out_png)

    manifest[STANDIN_ID] = {
        "image_id": STANDIN_ID,
        "file": out_tif.name,
        "preview": out_png.name,
        "modality": "sar",
        "source": "synthetic-sar-standin",
        "acquired": source_entry["acquired"],
        "resolution_m": source_entry.get("resolution_m"),
        "crs": str(crs),
        "size_px": source_entry["size_px"],
        "bounds_epsg4326": source_entry["bounds_epsg4326"],
        "site": source_entry.get("site", "vienna"),
        "purpose": (
            "demo stand-in for cross_modal routing -- NOT real SAR; see "
            "make_synthetic_sar_standin.py"
        ),
        "pair_id": SOURCE_ID,
    }
    manifest[SOURCE_ID]["pair_id"] = STANDIN_ID
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_tif}, {out_png}, and updated {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
