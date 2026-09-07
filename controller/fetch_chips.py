"""Pull small Sentinel-2 RGB chips from the public AWS Open Data archive.

Sentinel-2 L2A scenes are stored there as cloud-optimised GeoTIFFs, so a
window can be read over HTTP without downloading the whole 100 MB scene. No
account is needed. Output per chip: a georeferenced GeoTIFF (what the model
services read), a PNG preview (what a human looks at), and one entry in
images/manifest.json with the EPSG:4326 bounds the controller needs.

Run: .venv/Scripts/python fetch_chips.py [--sites vienna khavda ...] [--out images]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import requests

STAC = "https://earth-search.aws.element84.com/v1/search"
# Each site is a ~5 km box (500 px at 10 m, the size GeoChat likes) chosen for
# one of the problem statement's query types. Date windows are the site's dry
# season so the clearest scene in the window is genuinely clear.
#   name, (W, S, E, N), what it is for
SITES = {
    # Central Vienna: the original demo AOI. Dense city, river, parks: VQA and grounding.
    "vienna":       ((16.35, 48.19, 16.42, 48.24), "city centre, Danube; VQA + grounding + cloudy-scene routing"),
    # Navi Mumbai International Airport (18.99 N, 73.07 E): earthworks in 2023, runways and terminal by 2026.
    "navi_mumbai":  ((73.02, 18.96, 73.10, 19.02), "airport under construction; 'built after 2023' chain"),
    # Noida International Airport, Jewar (28.17 N, 77.61 E): fields in 2023, a finished airport by 2026.
    "jewar":        ((77.57, 28.14, 77.65, 28.20), "greenfield airport build-out; change detection, construction"),
    # JNPT / Nhava Sheva: container port with ships at anchor; count queries.
    "jnpt_port":    ((72.92, 18.92, 72.99, 18.97), "container port; 'how many ships' ground + count"),
    # Rondonia, Brazil: active deforestation frontier; vegetation loss.
    "rondonia":     ((-63.10, -9.95, -63.03, -9.90), "deforestation frontier; vegetation_loss change"),
    # Udaipur (24.58 N, 73.68 E): Pichola and Fateh Sagar lakes inside the city; water bodies,
    # and their level differs between the pre-monsoon and post-monsoon dates.
    "udaipur":      ((73.64, 24.55, 73.72, 24.62), "city lakes; 'find the water bodies', water-level change"),
    # Bhadla Solar Park (27.54 N, 71.92 E): 56 km2 of panels in the desert; 'find solar panels'.
    "bhadla":       ((71.87, 27.50, 71.95, 27.56), "solar park; grounding + land-cover VQA"),
    # Jayant opencast mine, Singrauli (24.11-24.19 N, 82.61-82.69 E): pit expansion; mining land cover.
    "singrauli":    ((82.61, 24.11, 82.69, 24.17), "opencast coal mine; expansion change, land-cover VQA"),
    # Singapore eastern anchorages (1.28-1.33 N, 103.96-104.06 E): dozens of ships at anchor; counting.
    "singapore":    ((103.96, 1.27, 104.06, 1.33), "ship anchorage; 'how many ships' ground + count"),
    # Punjab cropland near Ludhiana (30.88 N, 75.74 E): field mosaic; agriculture VQA.
    "punjab":       ((75.70, 30.85, 75.78, 30.91), "irrigated cropland; agriculture land-cover VQA"),
    # Sundarbans (21.88 N, 88.79 E): mangrove and tidal channels; wetland VQA, water/land boundary.
    "sundarbans":   ((88.75, 21.85, 88.83, 21.91), "mangrove wetland; land-cover VQA, water grounding"),
}

# Per-site date windows: (suffix, window, pick). India and Brazil windows are
# dry season. Vienna keeps its original ids so nothing downstream renames.
_INDIA = [("2023_opt", "2023-01-15T00:00:00Z/2023-03-31T23:59:59Z", "clearest"),
          ("2026_opt", "2026-01-15T00:00:00Z/2026-03-31T23:59:59Z", "clearest")]
_DRY_S = [("2023_opt", "2023-06-01T00:00:00Z/2023-08-31T23:59:59Z", "clearest"),
          ("2026_opt", "2026-06-01T00:00:00Z/2026-08-31T23:59:59Z", "clearest")]
WINDOWS = {
    "vienna": [("2023_opt", "2023-05-01T00:00:00Z/2023-09-30T23:59:59Z", "clearest"),
               ("2026_opt", "2026-05-01T00:00:00Z/2026-09-07T23:59:59Z", "clearest"),
               ("cloud_opt", "2026-05-01T00:00:00Z/2026-09-07T23:59:59Z", "cloudiest")],
    "navi_mumbai": _INDIA, "jewar": _INDIA, "jnpt_port": _INDIA, "rondonia": _DRY_S,
    # Pre-monsoon low water vs post-monsoon high water, so the lakes visibly change.
    "udaipur": [("2023_opt", "2023-04-01T00:00:00Z/2023-05-31T23:59:59Z", "clearest"),
                ("2026_opt", "2026-01-15T00:00:00Z/2026-03-15T23:59:59Z", "clearest")],
    "bhadla": _INDIA, "singrauli": _INDIA, "punjab": _INDIA,
    # Tropics: take the clearest scene of each whole year.
    "singapore": [("2023_opt", "2023-01-01T00:00:00Z/2023-12-31T23:59:59Z", "clearest"),
                  ("2026_opt", "2026-01-01T00:00:00Z/2026-09-07T23:59:59Z", "clearest")],
    "sundarbans": [("2023_opt", "2023-11-01T00:00:00Z/2024-02-28T23:59:59Z", "clearest"),
                   ("2026_opt", "2026-01-01T00:00:00Z/2026-03-31T23:59:59Z", "clearest")],
}


def image_id_for(site: str, suffix: str) -> str:
    return f"img_{suffix}" if site == "vienna" else f"{site}_{suffix}"


def search(aoi, window: str, limit: int = 40) -> list[dict]:
    body = {"collections": ["sentinel-2-l2a"], "bbox": list(aoi), "datetime": window, "limit": limit}
    r = requests.post(STAC, json=body, timeout=60)
    r.raise_for_status()
    # Sentinel-2 scenes are 100 km tiles; a box near a tile edge is only
    # partly inside some of them and would come back as a sliver. Keep the
    # scenes whose footprint contains the whole box.
    w, s, e, n = aoi
    return [f for f in r.json()["features"]
            if f["bbox"][0] <= w and f["bbox"][1] <= s and f["bbox"][2] >= e and f["bbox"][3] >= n]


def fetch_chip(item: dict, aoi, out_dir: Path, image_id: str) -> dict:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    from PIL import Image

    href = item["assets"]["visual"]["href"]  # true-colour 8-bit RGB, 10 m
    with rasterio.Env(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            # AOI is lon/lat; the scene is UTM. Convert, then read just that window.
            w, s, e, n = transform_bounds("EPSG:4326", src.crs, *aoi)
            win = from_bounds(w, s, e, n, src.transform).round_offsets().round_lengths()
            data = src.read(window=win)
            transform = src.window_transform(win)
            profile = src.profile.copy()
            crs = src.crs
    profile.update(driver="GTiff", height=data.shape[1], width=data.shape[2],
                   transform=transform, compress="deflate", tiled=False)
    profile.pop("blockxsize", None); profile.pop("blockysize", None)

    tif = out_dir / f"{image_id}.tif"
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(data)
    png = out_dir / f"{image_id}.png"
    Image.fromarray(data.transpose(1, 2, 0)).save(png)

    # Bounds of what was actually read, back in lon/lat, which is what the
    # controller's footprint check uses. Pixel bounds never leave this file.
    with rasterio.open(tif) as chk:
        lonlat = transform_bounds(chk.crs, "EPSG:4326", *chk.bounds)
    p = item["properties"]
    return {
        "image_id": image_id,
        "file": tif.name,
        "preview": png.name,
        "modality": "optical",
        "source": "sentinel-2",
        "acquired": p["datetime"][:10],
        "scene_cloud_cover_pct": round(p["eo:cloud_cover"], 1),
        "resolution_m": 10,
        "crs": str(crs),
        "size_px": [data.shape[2], data.shape[1]],
        "bounds_epsg4326": [round(v, 6) for v in lonlat],
        "stac_id": item["id"],
        "href": href,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sites", nargs="*", default=list(SITES), help="subset of sites to fetch")
    ap.add_argument("--out", default="images")
    ns = ap.parse_args()
    out_dir = Path(__file__).with_name(ns.out)
    out_dir.mkdir(exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    for site in ns.sites:
        aoi, purpose = SITES[site]
        for suffix, window, pick in WINDOWS[site]:
            image_id = image_id_for(site, suffix)
            items = search(aoi, window)
            if not items:
                print(f"{image_id}: no scenes in {window}"); continue
            items.sort(key=lambda f: f["properties"]["eo:cloud_cover"], reverse=(pick == "cloudiest"))
            entry = fetch_chip(items[0], aoi, out_dir, image_id)
            entry["site"] = site
            entry["purpose"] = purpose
            manifest[image_id] = entry
            print(f"{image_id:26s} {entry['acquired']}  scene cloud {entry['scene_cloud_cover_pct']:5.1f}%  "
                  f"{entry['size_px'][0]}x{entry['size_px'][1]} px  {os.path.getsize(out_dir / entry['file']) // 1024} KB")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} ({len(manifest)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
