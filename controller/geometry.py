"""Geometry helpers.

Everything that crosses a service boundary is a GeoJSON polygon in EPSG:4326
(lon/lat). Adapters are responsible for reprojecting model output before it
gets here; nothing downstream knows about pixels or UTM.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator

from pyproj import Geod
from shapely import make_valid
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

GeoJSON = dict[str, Any]

_GEOD = Geod(ellps="WGS84")
_POLYGONAL = {"Polygon", "MultiPolygon"}


class GeometryError(ValueError):
    """Raised when geometry is malformed or is not plausibly lon/lat."""


def _positions(coords: Any) -> Iterator[tuple[float, float]]:
    if not isinstance(coords, (list, tuple)) or not coords:
        raise GeometryError(f"expected a non-empty coordinate list, got {coords!r}")

    if isinstance(coords[0], (int, float)):
        if len(coords) < 2:
            raise GeometryError(f"position {coords!r} needs at least two values")
        try:
            yield float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            raise GeometryError(f"position {coords!r} is not numeric") from None
        return

    for part in coords:
        yield from _positions(part)


def validate(geom: Any, *, where: str = "geometry") -> None:
    """Reject anything that is not a polygon in a plausible lon/lat range.

    The range check is the guard against projected coordinates leaking out of
    an adapter. BigEarthNet patches are stored in UTM, so a missed reprojection
    produces eastings like 4534120 that would otherwise flow through the whole
    pipeline looking like valid data.
    """
    if not isinstance(geom, dict):
        raise GeometryError(f"{where}: expected a GeoJSON object, got {type(geom).__name__}")

    gtype = geom.get("type")
    if gtype not in _POLYGONAL:
        raise GeometryError(f"{where}: type must be Polygon or MultiPolygon, got {gtype!r}")

    coords = geom.get("coordinates")
    if not coords:
        raise GeometryError(f"{where}: missing or empty 'coordinates'")

    for lon, lat in _positions(coords):
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise GeometryError(
                f"{where}: ({lon}, {lat}) is outside the EPSG:4326 range. "
                "Projected coordinates must be reprojected inside the adapter "
                "(rasterio.warp.transform_geom)."
            )


def _polygonal(g: BaseGeometry) -> BaseGeometry | None:
    """The polygon parts of a repaired geometry, or None if there are none."""
    if g.geom_type in _POLYGONAL:
        return g
    parts = getattr(g, "geoms", None)
    if parts is None:
        return None
    polygons = [p for p in parts if p.geom_type in _POLYGONAL]
    return unary_union(polygons) if polygons else None


def to_shape(geom: GeoJSON, *, where: str = "geometry") -> BaseGeometry:
    validate(geom, where=where)
    try:
        g = shape(geom)
    except Exception as exc:
        raise GeometryError(f"{where}: not a usable geometry ({exc})") from exc

    if not g.is_valid:
        # Self-intersecting rings are common in polygonised model output.
        # make_valid keeps every lobe of a bowtie where buffer(0) drops one.
        # The "structure" method is the one that treats a hole poking past its
        # shell as a hole; the default "linework" turns the overhang into a
        # phantom positive-area lobe.
        g = _polygonal(make_valid(g, method="structure"))
        if g is None:
            raise GeometryError(f"{where}: could not be repaired into a polygon")
    if g.is_empty:
        raise GeometryError(f"{where}: geometry is empty")
    return g


def _ring_area_km2(ring) -> float:
    lons, lats = ring.coords.xy
    square_metres, _ = _GEOD.polygon_area_perimeter(lons, lats)
    return abs(square_metres) / 1_000_000.0


def area_km2(geom: GeoJSON | BaseGeometry) -> float:
    """Geodesic area. Degrees are not equal-area, so this cannot be done with
    plain shapely .area.

    Rings are measured one at a time. The geodesic area of a ring is signed by
    its winding, and GeoJSON producers do not wind consistently, so measuring a
    whole polygon in one call lets a same-wound hole add to the total instead
    of subtracting from it, and lets the parts of a multi-polygon cancel.
    """
    g = geom if isinstance(geom, BaseGeometry) else to_shape(geom)

    parts = getattr(g, "geoms", None)
    if parts is not None:
        return sum(area_km2(part) for part in parts)

    exterior = getattr(g, "exterior", None)
    if exterior is None:
        return 0.0
    return _ring_area_km2(exterior) - sum(_ring_area_km2(hole) for hole in g.interiors)


def overlap_fraction(inner: BaseGeometry, outer: BaseGeometry) -> float:
    """Share of `inner` that falls within `outer`, 0.0 to 1.0."""
    if inner.is_empty or outer.is_empty:
        return 0.0
    if not inner.intersects(outer):
        return 0.0
    denominator = inner.area
    if denominator <= 0:
        return 1.0 if outer.contains(inner.centroid) else 0.0
    return inner.intersection(outer).area / denominator


def union(shapes: Iterable[BaseGeometry]) -> BaseGeometry:
    """Merge geometries into one. Change masks arrive polygonised into many
    adjacent fragments, and anything measuring coverage against them has to
    treat them as a single area or it under-counts objects that straddle a seam.
    """
    return unary_union(list(shapes))


def footprint(bounds: tuple[float, float, float, float]) -> BaseGeometry:
    """Scene bounds (west, south, east, north) as a polygon."""
    return box(*bounds)
