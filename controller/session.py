"""Image registry for a single user session.

The planner is shown this registry so it can only ever reference images that
actually exist. The validator uses it to reject plans that invent an image_id.
"""

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Iterator, Literal

Modality = Literal["optical", "sar"]
Bounds = tuple[float, float, float, float]  # west, south, east, north (EPSG:4326)

MODALITIES = ("optical", "sar")

# Footprint IoU below this means two scenes are not the same area, and any
# comparison between them is meaningless however plausible the output looks.
MIN_CO_REGISTRATION = 0.9


class UnknownImage(KeyError):
    pass


def _checked_bounds(bounds, image_id: str) -> Bounds:
    try:
        w, s, e, n = (float(v) for v in bounds)
    except (TypeError, ValueError):
        raise ValueError(f"{image_id}: bounds must be four numbers, got {bounds!r}") from None

    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
        raise ValueError(f"{image_id}: bounds are outside EPSG:4326: {bounds}")
    if w > 0 > e:
        # transform_bounds returns west > east for a scene straddling the
        # antimeridian. Handling that means splitting the footprint, which
        # nothing here does, so refuse it rather than let a 358-degree-wide
        # box through.
        raise ValueError(
            f"{image_id}: scene straddles the antimeridian (west={w}, east={e}); not supported"
        )
    if not (w < e and s < n):
        raise ValueError(f"{image_id}: bounds are not west<east, south<north: {bounds}")
    return (w, s, e, n)


@dataclass(frozen=True)
class Scene:
    image_id: str
    modality: Modality
    source: str
    acquired: date
    bounds: Bounds
    resolution_m: float | None = None
    cloudy: bool = False
    snow: bool = False
    pair_id: str | None = None
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise ValueError(
                f"{self.image_id}: modality must be one of {MODALITIES}, got {self.modality!r}"
            )
        if not isinstance(self.acquired, date):
            # The date decides which scene is "earlier" in a change comparison,
            # so a string here would not fail — it would compare wrongly.
            raise TypeError(
                f"{self.image_id}: acquired must be a datetime.date, got {type(self.acquired).__name__}"
            )
        # A datetime is a date subclass but does not compare with one, so a
        # loader passing timestamps would make the validator raise mid-check.
        if isinstance(self.acquired, datetime):
            object.__setattr__(self, "acquired", self.acquired.date())
        object.__setattr__(self, "bounds", _checked_bounds(self.bounds, self.image_id))
        object.__setattr__(self, "labels", tuple(self.labels))

    def overlap(self, other: "Scene") -> float:
        """Intersection over union of the two footprints, 0.0 to 1.0.

        Co-registered scenes score close to 1. IoU rather than intersection
        over the smaller footprint, so a small patch nested inside a much
        larger tile scores near 0 instead of a perfect 1.0.
        """
        aw, as_, ae, an = self.bounds
        bw, bs, be, bn = other.bounds
        dx = min(ae, be) - max(aw, bw)
        dy = min(an, bn) - max(as_, bs)
        if dx <= 0 or dy <= 0:
            return 0.0
        intersection = dx * dy
        union = (ae - aw) * (an - as_) + (be - bw) * (bn - bs) - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def from_geotiff(
        cls, path: str, image_id: str, *, acquired: date, modality: Modality, **meta
    ) -> "Scene":
        """Read bounds off a real GeoTIFF, reprojecting to lon/lat.

        BigEarthNet patches are stored in UTM, so the transform_bounds call is
        not optional.

        `acquired` and `modality` are required rather than defaulted: the date
        decides which scene is "earlier" in a change comparison, and guessing it
        would invert the answer while looking entirely reasonable.
        """
        if "bounds" in meta:
            raise TypeError("from_geotiff reads bounds from the file; do not pass them")

        import rasterio
        from rasterio.warp import transform_bounds

        with rasterio.open(path) as src:
            bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
        meta.setdefault("source", "unknown")
        return cls(
            image_id=image_id, bounds=tuple(bounds), acquired=acquired, modality=modality, **meta
        )


class Session:
    """The images one query can see.

    Build it completely, then use it. Area labels are assigned over the whole
    set of scenes, so adding a scene after describe() has been rendered could
    change what "area A" refers to under a plan already drafted against it;
    add() refuses once the labels have been handed out.
    """

    def __init__(self, scenes: Iterable[Scene] = ()) -> None:
        self._scenes: dict[str, Scene] = {}
        self._areas: dict[str, str] | None = None
        for scene in scenes:
            self.add(scene)

    def add(self, scene: Scene) -> None:
        if self._areas is not None:
            raise RuntimeError(
                "session is already in use; add every scene before calling areas() or describe()"
            )
        if scene.image_id in self._scenes:
            raise ValueError(f"duplicate image_id: {scene.image_id}")
        self._scenes[scene.image_id] = scene

    def get(self, image_id: str) -> Scene:
        try:
            return self._scenes[image_id]
        except KeyError:
            raise UnknownImage(image_id) from None

    def has(self, image_id: str) -> bool:
        return image_id in self._scenes

    def ids(self) -> list[str]:
        return list(self._scenes)

    def counterpart(self, image_id: str) -> Scene | None:
        """The co-registered scene from the other sensor, if one is loaded."""
        pair = self.get(image_id).pair_id
        return self._scenes.get(pair) if pair else None

    def __len__(self) -> int:
        return len(self._scenes)

    def __iter__(self) -> Iterator[Scene]:
        return iter(self._scenes.values())

    def areas(self) -> dict[str, str]:
        """Label each scene with the AOI it covers, grouping co-registered scenes.

        This is the one definition of "same area". The planner sees these labels
        in describe(), and the validator answers same_area() from the same
        partition, so the two can never disagree about which scenes may be
        compared.

        A scene joins a group only if it is co-registered with every member,
        and scenes are visited in a fixed order, so two scenes given the same
        label always clear the threshold with each other and the grouping does
        not depend on load order. Overlap is not transitive, so a borderline
        pair can still land in different groups; that errs towards refusing a
        comparison, never towards allowing a bad one.
        """
        if self._areas is None:
            labels: dict[str, str] = {}
            groups: list[list[Scene]] = []
            for scene in sorted(self._scenes.values(), key=lambda s: s.image_id):
                for group in groups:
                    if all(scene.overlap(member) >= MIN_CO_REGISTRATION for member in group):
                        group.append(scene)
                        labels[scene.image_id] = labels[group[0].image_id]
                        break
                else:
                    groups.append([scene])
                    labels[scene.image_id] = _area_label(len(groups) - 1)
            self._areas = labels
        return self._areas

    def same_area(self, a: str, b: str) -> bool:
        areas = self.areas()
        return areas[a] == areas[b]

    def describe(self) -> str:
        """Rendered into the planner prompt. Keep it compact and factual."""
        if not self._scenes:
            return "(no images loaded)"
        areas = self.areas()
        lines = []
        for s in self._scenes.values():
            flags = []
            if s.cloudy:
                flags.append("cloudy")
            if s.snow:
                flags.append("snow")
            if s.pair_id and self.has(s.pair_id):
                flags.append(f"paired with {s.pair_id}")
            suffix = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"- {s.image_id}: {s.modality}, {s.source}, {s.acquired.isoformat()}, "
                f"{areas[s.image_id]}{suffix}"
            )
        return "\n".join(lines)


def _area_label(index: int) -> str:
    """A, B, ... Z, AA, AB, ... so the label stays a clean identifier however
    many areas are loaded. These are echoed back by the planner."""
    letters = ""
    while True:
        letters = chr(ord("A") + index % 26) + letters
        index = index // 26 - 1
        if index < 0:
            return f"area {letters}"


# 1.2 km square, matching a BigEarthNet patch footprint.
_AOI_CLEAR: Bounds = (16.3700, 48.2100, 16.3862, 48.2208)
_AOI_CLOUDY: Bounds = (16.4200, 48.2500, 16.4362, 48.2608)


def demo_session() -> Session:
    """Fixture covering every routing case: a bi-temporal pair, an optical/SAR
    pair, and a clouded scene that should push the planner onto SAR."""
    return Session([
        Scene("img_2023_opt", "optical", "sentinel-2", date(2023, 6, 15), _AOI_CLEAR,
              resolution_m=10, labels=("Arable land", "Mixed forest")),
        Scene("img_2026_opt", "optical", "sentinel-2", date(2026, 6, 15), _AOI_CLEAR,
              resolution_m=10, pair_id="img_2026_sar", labels=("Arable land", "Urban fabric")),
        Scene("img_2026_sar", "sar", "sentinel-1", date(2026, 6, 15), _AOI_CLEAR,
              resolution_m=10, pair_id="img_2026_opt"),
        Scene("img_cloud_opt", "optical", "sentinel-2", date(2026, 7, 2), _AOI_CLOUDY,
              resolution_m=10, cloudy=True, pair_id="img_cloud_sar"),
        Scene("img_cloud_sar", "sar", "sentinel-1", date(2026, 7, 2), _AOI_CLOUDY,
              resolution_m=10, pair_id="img_cloud_opt"),
    ])


# Scene-level cloud cover above this is treated as "cloudy" for routing. It is
# the same 0.4 threshold the implementation doc gives the planner's sensor rule.
CLOUDY_ABOVE_PCT = 40.0


def manifest_session(path: str | Path = Path(__file__).with_name("images") / "manifest.json") -> Session:
    """Real imagery: one Scene per entry in images/manifest.json, as written by
    fetch_chips.py. Bounds are read off each GeoTIFF, not trusted from the
    manifest, so the footprint the planner sees is the footprint the file has.
    A manifest entry may carry `cloudy`, `pair_id` and `labels` explicitly;
    otherwise cloudiness comes from the scene cloud cover."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    scenes = []
    for image_id, m in manifest.items():
        cloudy = bool(m.get("cloudy", m.get("scene_cloud_cover_pct", 0) > CLOUDY_ABOVE_PCT))
        scenes.append(Scene.from_geotiff(
            str(path.with_name(m["file"])), image_id,
            acquired=date.fromisoformat(m["acquired"]), modality=m["modality"],
            source=m.get("source", "unknown"), resolution_m=m.get("resolution_m"),
            cloudy=cloudy, pair_id=m.get("pair_id"), labels=tuple(m.get("labels", ())),
        ))
    return Session(scenes)
