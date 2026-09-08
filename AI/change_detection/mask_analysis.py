"""Turn a change-probability map into discrete, described regions.

Thresholds the probability map from ``change_model.detect_change``, groups the
result into 8-connected components, drops specks, and describes what survives
-- in pixel coordinates always, and in geographic coordinates when the source
raster carries a CRS.

Georeferencing is optional by design: the BIT_CD sample imagery is plain PNG
with no CRS, so the ungeoreferenced path is the one exercised by the tests, not
a fallback.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform as warp_transform
from scipy import ndimage

WGS84 = "EPSG:4326"

# Mean metres per degree of latitude on the WGS84 ellipsoid. Only used to
# approximate area when the source CRS is geographic -- see _pixel_area_sq_m.
METRES_PER_DEGREE = 111_320.0

# Row-major 3x3 grid, indexed [row][col] from the image's top-left.
COMPASS_GRID = (
    ("north-west", "north", "north-east"),
    ("west", "centre", "east"),
    ("south-west", "south", "south-east"),
)


def _read_georeferencing(image_path: str | Path):
    """Return (crs, affine transform, (width, height)) for the raster.

    ``crs`` is None for an image with no spatial reference, such as a PNG.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")

    with warnings.catch_warnings():
        # A plain PNG is an expected input here, so this warning is noise.
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as dataset:
            return dataset.crs, dataset.transform, (dataset.width, dataset.height)


def _position(x: float, y: float, width: int, height: int) -> str:
    """Compass label for a point, from its cell in a 3x3 grid over the image."""
    col = min(2, int(x / width * 3))
    row = min(2, int(y / height * 3))
    return COMPASS_GRID[row][col]


def _pixel_area_sq_m(transform, crs, latitude: float) -> float:
    """Ground area of one pixel in square metres.

    The determinant of the affine transform gives pixel area in the CRS's own
    units, which handles rotated and non-square pixels.
    """
    area_in_crs_units = abs(transform.a * transform.e - transform.b * transform.d)

    if crs.is_projected:
        # linear_units_factor is (unit_name, metres_per_unit); squared because
        # this is an area. Handles feet-based CRSs as well as metre-based ones.
        _, metres_per_unit = crs.linear_units_factor
        return area_in_crs_units * metres_per_unit**2

    # Geographic CRS: units are degrees, and a degree of longitude shrinks
    # towards the poles. This is a local approximation evaluated at the
    # region's own latitude -- good to well under a percent for a single
    # scene, but not a substitute for an equal-area projection over a
    # continent-scale extent.
    metres_per_degree_lon = METRES_PER_DEGREE * math.cos(math.radians(latitude))
    return area_in_crs_units * METRES_PER_DEGREE * metres_per_degree_lon


def _to_wgs84(xs, ys, crs) -> tuple[list[float], list[float]]:
    """Reproject CRS coordinates to WGS84, returning (lons, lats)."""
    lons, lats = warp_transform(crs, WGS84, list(xs), list(ys))
    return lons, lats


def analyse_mask(
    probability: np.ndarray,
    after_image_path: str | Path,
    threshold: float = 0.5,
    min_region_px: int = 50,
) -> dict:
    """Describe the changed regions in a probability map.

    Args:
        probability: HxW float array in [0, 1], as returned by
            ``change_model.detect_change``.
        after_image_path: the later image of the pair. Read only for its CRS
            and affine transform; pixel values are ignored.
        threshold: probability above which a pixel counts as changed.
        min_region_px: regions smaller than this are discarded as noise.

    Regions are 8-connected: pixels touching diagonally belong to the same
    region. A cluster linked only by corners therefore counts once, and can
    clear `min_region_px` where its separate parts would not have.

    Returns a dict with:
        change_percentage: percent of pixels above `threshold`, to 2dp. Counts
            every such pixel, including those in regions later dropped by the
            size filter, so it will not always agree with the summed region
            areas.
        region_count: number of regions surviving the size filter.
        georeferenced: whether the source raster carried a CRS.
        regions: list of region dicts, largest first.

    Each region carries `id`, `bbox` ([x_min, y_min, x_max, y_max], inclusive
    pixel indices), `centroid` ([x, y] sub-pixel), `area_px`, `confidence`
    (mean probability inside the region, 3dp), and `position`. When
    georeferenced it also carries `bbox_latlon`, `centroid_latlon` (both
    [lon, lat] ordered) and `area_sq_m`.
    """
    probability = np.asarray(probability)
    if probability.ndim != 2:
        raise ValueError(f"probability must be a 2-D HxW array, got shape {probability.shape}")

    crs, transform, (raster_width, raster_height) = _read_georeferencing(after_image_path)

    height, width = probability.shape
    if (width, height) != (raster_width, raster_height):
        raise ValueError(
            f"probability map is {width}x{height} but {Path(after_image_path).name} "
            f"is {raster_width}x{raster_height}; they must describe the same grid"
        )

    changed = probability > threshold
    change_percentage = round(float(changed.mean()) * 100.0, 2)

    # 8-connectivity (scipy's connectivity 2): a 3x3 structure of ones makes
    # diagonal neighbours count, so regions touching only at a corner merge
    # into one. Omitting `structure` would give scipy's 4-connectivity
    # default, which splits them.
    labels, label_count = ndimage.label(changed, structure=np.ones((3, 3)))

    georeferenced = crs is not None
    result = {
        "change_percentage": change_percentage,
        "region_count": 0,
        "georeferenced": georeferenced,
        "regions": [],
    }

    if label_count == 0:
        return result

    # Index 0 is background; areas[i] is the pixel count of label i.
    areas = np.bincount(labels.ravel(), minlength=label_count + 1)
    kept_labels = [i for i in range(1, label_count + 1) if areas[i] >= min_region_px]
    if not kept_labels:
        return result

    # Largest first; the id suffix follows this order.
    kept_labels.sort(key=lambda i: int(areas[i]), reverse=True)

    # Both are vectorized over the label list, which beats masking per region.
    centroids = ndimage.center_of_mass(changed, labels, kept_labels)
    confidences = ndimage.mean(probability, labels, kept_labels)
    boxes = ndimage.find_objects(labels)

    regions = []
    for rank, label_id in enumerate(kept_labels, start=1):
        y_slice, x_slice = boxes[label_id - 1]
        # find_objects returns half-open slices; -1 makes the bbox inclusive.
        x_min, y_min = int(x_slice.start), int(y_slice.start)
        x_max, y_max = int(x_slice.stop) - 1, int(y_slice.stop) - 1

        centroid_y, centroid_x = centroids[rank - 1]
        centroid_x, centroid_y = float(centroid_x), float(centroid_y)

        region = {
            "id": f"change_{rank:03d}",
            "bbox": [x_min, y_min, x_max, y_max],
            "centroid": [round(centroid_x, 2), round(centroid_y, 2)],
            "area_px": int(areas[label_id]),
            "confidence": round(float(confidences[rank - 1]), 3),
            "position": _position(centroid_x, centroid_y, width, height),
        }

        if georeferenced:
            # Offset by one pixel on the far edge so the box spans the full
            # footprint of the corner pixels rather than their origins.
            corner_cols = [x_min, x_max + 1, x_min, x_max + 1]
            corner_rows = [y_min, y_min, y_max + 1, y_max + 1]
            corner_xs, corner_ys = zip(
                *(transform * (col, row) for col, row in zip(corner_cols, corner_rows))
            )
            corner_lons, corner_lats = _to_wgs84(corner_xs, corner_ys, crs)

            centre_x, centre_y = transform * (centroid_x, centroid_y)
            centre_lon, centre_lat = _to_wgs84([centre_x], [centre_y], crs)

            # Take min/max rather than assuming corner order: north-up rasters
            # have a negative y scale, so pixel row and latitude run opposite
            # ways, and a rotated transform can reorder them further.
            region["bbox_latlon"] = [
                round(min(corner_lons), 6),
                round(min(corner_lats), 6),
                round(max(corner_lons), 6),
                round(max(corner_lats), 6),
            ]
            region["centroid_latlon"] = [round(centre_lon[0], 6), round(centre_lat[0], 6)]
            region["area_sq_m"] = round(
                int(areas[label_id]) * _pixel_area_sq_m(transform, crs, centre_lat[0]), 2
            )

        regions.append(region)

    result["region_count"] = len(regions)
    result["regions"] = regions
    return result
