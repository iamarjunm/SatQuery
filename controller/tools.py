"""Tool registry and adapters.

The registry is the single source of truth. The planner renders its menu from
here, the validator checks tool names and arguments against it, and the
executor dispatches through it. Adding a capability means editing this file
and nothing else.

Remote tools are reached over HTTP only. The controller never imports a model,
so a service can live on a Kaggle notebook behind a tunnel and be swapped,
mocked or killed without touching this code.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable

from shapely import prepare

import geometry
from session import Session

Result = dict[str, Any]
Adapter = Callable[[dict[str, Any], Session], Result]

HTTP_TIMEOUT_S = 60

# A building that merely grazes the corner of a changed region is not a new
# building. Requiring half its footprint inside keeps the chained answer honest.
MIN_REGION_OVERLAP = 0.5

# Model output may spill slightly past the scene edge after resampling, but a
# result mostly outside the source image means the adapter got the CRS wrong.
MIN_FOOTPRINT_OVERLAP = 0.5

# Argument and return kinds that carry a list of geometry items. A reference
# feeding one of these must come from another.
LIST_KINDS = frozenset({"objects", "regions"})


class ToolError(RuntimeError):
    """A tool failed in a way the executor should report, not crash on."""


@dataclass(frozen=True)
class ArgSpec:
    kind: str  # image_id | text | objects | regions
    description: str
    required: bool = True


@dataclass(frozen=True)
class ReturnSpec:
    kind: str  # objects | regions | text | number
    description: str
    nullable: bool = False


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    args: dict[str, ArgSpec]
    returns: dict[str, ReturnSpec]
    local: bool = False
    port: int | None = None

    @property
    def required_args(self) -> set[str]:
        return {n for n, a in self.args.items() if a.required}

    @property
    def list_fields(self) -> list[str]:
        """Return fields that carry geometry items, by declared kind rather
        than by name, so a tool may call its list whatever it likes."""
        return [f for f, r in self.returns.items() if r.kind in LIST_KINDS]


TOOLS: dict[str, ToolSpec] = {
    "vqa": ToolSpec(
        name="vqa",
        summary="Answer a free-text question about one image. Use for description, "
                "land cover and image-quality questions that need no localisation.",
        args={
            "image_id": ArgSpec("image_id", "image to look at"),
            "question": ArgSpec("text", "question phrased for the model"),
        },
        returns={
            "answer": ReturnSpec("text", "natural-language answer"),
            "confidence": ReturnSpec("number", "0-1"),
        },
        port=8001,
    ),
    "ground": ToolSpec(
        name="ground",
        summary="Locate every instance of a phrase in one image and return its "
                "outline. Use for find / show / locate / highlight questions.",
        args={
            "image_id": ArgSpec("image_id", "image to search"),
            "phrase": ArgSpec("text", "what to look for, e.g. 'buildings'"),
        },
        returns={"objects": ReturnSpec("objects", "list of {label, geometry, confidence}")},
        port=8002,
    ),
    "change_detect": ToolSpec(
        name="change_detect",
        summary="Compare two images of the same area taken at different times and "
                "return the regions that differ.",
        args={
            "image_id_t1": ArgSpec("image_id", "earlier image"),
            "image_id_t2": ArgSpec("image_id", "later image"),
        },
        returns={
            "regions": ReturnSpec("regions", "list of {type, geometry, confidence}"),
            "changed_area_km2": ReturnSpec("number", "total changed area"),
            "change_mask_id": ReturnSpec("text", "id of the rendered mask"),
        },
        port=8003,
    ),
    "cross_modal": ToolSpec(
        name="cross_modal",
        summary="Analyse a co-registered optical and SAR pair together. Use when the "
                "question needs both sensors, or when the optical scene is clouded.",
        args={
            "optical_image_id": ArgSpec("image_id", "Sentinel-2 image"),
            "sar_image_id": ArgSpec("image_id", "co-registered Sentinel-1 image"),
            "phrase": ArgSpec("text", "what to identify, e.g. 'built-up and water'"),
        },
        returns={
            "regions": ReturnSpec("regions", "list of {label, geometry, confidence}"),
            "summary": ReturnSpec("text", "text"),
        },
        port=8004,
    ),
    "filter_by_region": ToolSpec(
        name="filter_by_region",
        summary="Keep only the objects that lie inside the given regions. This is how "
                "'new buildings' is answered: ground the later image, then keep the "
                "buildings that fall inside the changed regions.",
        args={
            "objects": ArgSpec("objects", "objects to filter, as $step.objects"),
            "regions": ArgSpec("regions", "regions to keep within, as $step.regions"),
        },
        returns={
            "objects": ReturnSpec("objects", "surviving objects"),
            "kept": ReturnSpec("number", "count kept"),
            "dropped": ReturnSpec("number", "count dropped"),
        },
        local=True,
    ),
    "count": ToolSpec(
        name="count",
        summary="Count objects and report their mean confidence.",
        args={"objects": ArgSpec("objects", "objects to count, as $step.objects")},
        returns={
            "n": ReturnSpec("number", "number of objects"),
            "mean_confidence": ReturnSpec("number", "mean item confidence, null when n is 0",
                                          nullable=True),
        },
        local=True,
    ),
}


def render_menu() -> str:
    """Tool menu for the planner prompt, generated so it cannot drift from the code."""
    blocks = []
    for spec in TOOLS.values():
        args = "\n".join(
            f"    {name} ({a.kind}){'' if a.required else ', optional'}: {a.description}"
            for name, a in spec.args.items()
        )
        returns = ", ".join(f"{name} ({r.kind})" for name, r in spec.returns.items())
        blocks.append(f"{spec.name}\n  {spec.summary}\n  args:\n{args}\n  returns: {returns}")
    return "\n\n".join(blocks)


# --- item checks -------------------------------------------------------------

def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise ToolError(f"{where} must be a list, got {type(value).__name__}")
    return value


def _as_dict(item: Any, where: str) -> dict:
    if not isinstance(item, dict):
        raise ToolError(f"{where} must be an object, got {type(item).__name__}")
    return item


def _confidence_of(item: Any, where: str) -> float:
    confidence = _as_dict(item, where).get("confidence")
    if not _is_number(confidence) or not 0.0 <= confidence <= 1.0:
        raise ToolError(f"{where}: 'confidence' must be a number from 0 to 1, got {confidence!r}")
    return confidence


def _check_item(item: Any, where: str, footprints: list = ()):
    """One geometry item: well-formed lon/lat, located where its source image
    is, and carrying a usable confidence. Returns the parsed shape.

    Applied to everything a tool returns and to everything a local tool is
    given, so an item is judged the same way whichever path it arrives by.
    Failures are ToolErrors rather than KeyErrors so the executor can report
    them instead of dying.
    """
    geom = _as_dict(item, where).get("geometry")
    if geom is None:
        raise ToolError(f"{where} has no 'geometry' field")
    shape = geometry.to_shape(geom, where=where)

    if footprints:
        # The range check inside to_shape catches projected coordinates. It
        # cannot catch a lat/lon swap, because both numbers stay in range;
        # landing outside every source image catches that and every other
        # wrong-CRS case. Inside any one input is enough: the inputs are
        # co-registered (the validator refuses the call otherwise), and an
        # honest result near the edge may sit in the sliver only one covers.
        inside = max(geometry.overlap_fraction(shape, f) for f in footprints)
        if inside < MIN_FOOTPRINT_OVERLAP:
            extents = "; ".join(str(tuple(round(v, 5) for v in f.bounds)) for f in footprints)
            raise ToolError(
                f"{where}: only {inside:.0%} of this geometry falls inside the source "
                f"image footprint(s) {extents}. Check the adapter for swapped lon/lat "
                f"ordering or a missing reprojection to EPSG:4326."
            )

    _confidence_of(item, where)
    return shape


# --- local tools -------------------------------------------------------------

def _filter_by_region(args: dict[str, Any], session: Session) -> Result:
    objects = _as_list(args.get("objects"), "filter_by_region.objects")
    regions = _as_list(args.get("regions"), "filter_by_region.regions")

    # Every input is checked before the empty-input shortcut, so a malformed
    # object is reported whether or not there happened to be any regions, and
    # whether or not it would have survived the filter.
    shapes = [_check_item(o, f"filter_by_region.objects[{i}]") for i, o in enumerate(objects)]
    fragments = [_check_item(r, f"filter_by_region.regions[{i}]") for i, r in enumerate(regions)]
    if not shapes or not fragments:
        return {"objects": [], "kept": 0, "dropped": len(objects)}

    # Merged, not compared one at a time: a change mask is polygonised into many
    # adjacent fragments, and an object lying across a seam between two of them
    # is still inside the changed area.
    changed = geometry.union(fragments)
    prepare(changed)
    kept = [
        obj for obj, shape in zip(objects, shapes)
        if geometry.overlap_fraction(shape, changed) >= MIN_REGION_OVERLAP
    ]
    return {"objects": kept, "kept": len(kept), "dropped": len(objects) - len(kept)}


def _count(args: dict[str, Any], session: Session) -> Result:
    objects = _as_list(args.get("objects"), "count.objects")
    scores = [_confidence_of(o, f"count.objects[{i}]") for i, o in enumerate(objects)]
    return {
        "n": len(objects),
        # None rather than 0.0: an empty set has no mean, and 0.0 would read as
        # "detections with no confidence".
        "mean_confidence": round(sum(scores) / len(scores), 3) if scores else None,
    }


LOCAL: dict[str, Adapter] = {
    "filter_by_region": _filter_by_region,
    "count": _count,
}


# --- mocks -------------------------------------------------------------------

def _rect(bounds, x0: float, y0: float, x1: float, y1: float) -> dict:
    """Polygon from fractional offsets inside `bounds`. y runs from the top so
    the literals below read like image coordinates."""
    w, s, e, n = bounds
    lon0, lon1 = w + (e - w) * x0, w + (e - w) * x1
    lat0, lat1 = n - (n - s) * y1, n - (n - s) * y0
    return {
        "type": "Polygon",
        "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]],
    }


# Two of these five sit inside the mock changed regions below, so a
# ground -> filter_by_region chain returns a real subset rather than all or nothing.
_MOCK_OBJECTS = [
    (0.10, 0.10, 0.18, 0.18, 0.94),
    (0.30, 0.22, 0.38, 0.30, 0.91),
    (0.55, 0.45, 0.63, 0.53, 0.88),
    (0.60, 0.60, 0.68, 0.68, 0.86),
    (0.80, 0.20, 0.86, 0.26, 0.79),
]

_MOCK_CHANGES = [
    (0.25, 0.18, 0.45, 0.35, 0.92, "construction"),
    (0.55, 0.55, 0.75, 0.75, 0.87, "vegetation_loss"),
]


def _mock_vqa(args, session):
    scene = session.get(args["image_id"])
    labels = ", ".join(scene.labels) if scene.labels else "built-up areas and vegetation"
    return {"answer": f"The image shows {labels}.", "confidence": 0.89}


def _mock_ground(args, session):
    bounds = session.get(args["image_id"]).bounds
    label = str(args["phrase"])
    return {
        "objects": [
            {
                "id": f"obj_{i:03d}",
                "label": label,
                "geometry": _rect(bounds, x0, y0, x1, y1),
                "confidence": conf,
            }
            for i, (x0, y0, x1, y1, conf) in enumerate(_MOCK_OBJECTS, start=1)
        ]
    }


def _mock_change_detect(args, session):
    bounds = session.get(args["image_id_t2"]).bounds
    regions = [
        {
            "id": f"change_{i:03d}",
            "type": kind,
            "geometry": _rect(bounds, x0, y0, x1, y1),
            "confidence": conf,
        }
        for i, (x0, y0, x1, y1, conf, kind) in enumerate(_MOCK_CHANGES, start=1)
    ]
    return {
        "regions": regions,
        "changed_area_km2": round(sum(geometry.area_km2(r["geometry"]) for r in regions), 4),
        "change_mask_id": f"mask_{args['image_id_t1']}_{args['image_id_t2']}",
    }


def _mock_cross_modal(args, session):
    bounds = session.get(args["optical_image_id"]).bounds
    return {
        "regions": [
            {"id": "region_001", "label": "built_up",
             "geometry": _rect(bounds, 0.05, 0.05, 0.45, 0.40), "confidence": 0.91},
            {"id": "region_002", "label": "water",
             "geometry": _rect(bounds, 0.55, 0.50, 0.95, 0.90), "confidence": 0.94},
        ],
        "summary": "The scene contains built-up regions and water-covered regions.",
    }


MOCKS: dict[str, Adapter] = {
    "vqa": _mock_vqa,
    "ground": _mock_ground,
    "change_detect": _mock_change_detect,
    "cross_modal": _mock_cross_modal,
}


def _check_registry() -> None:
    """Fail at import if a tool is declared without an implementation, rather
    than on first call with a message blaming the adapter."""
    for name, spec in TOOLS.items():
        table, what = (LOCAL, "local implementation") if spec.local else (MOCKS, "mock adapter")
        if name not in table:
            raise RuntimeError(f"tool '{name}' is registered without a {what}")
    stray = (set(LOCAL) | set(MOCKS)) - set(TOOLS)
    if stray:
        raise RuntimeError(f"implementations without a ToolSpec: {', '.join(sorted(stray))}")


_check_registry()


# --- dispatch ----------------------------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}
_mock_warned: set[str] = set()


def _flag(value: str | None, default: bool) -> bool:
    """Unset or blank means the default. Anything else not clearly true is
    false, so 'off', '0 ' and a stray typo all count as off — opting out of
    the mocks is the case that has to work."""
    if value is None or not value.strip():
        return default
    return value.strip().lower() in _TRUTHY


def use_mock(name: str) -> bool:
    """Per-tool override wins, so real services can be swapped in one at a time."""
    specific = os.getenv(f"SATQUERY_MOCK_{name.upper()}")
    if specific is not None and specific.strip():
        return _flag(specific, True)
    return _flag(os.getenv("SATQUERY_MOCK"), True)


def endpoint(spec: ToolSpec) -> str:
    return os.getenv(
        f"SATQUERY_{spec.name.upper()}_URL",
        f"http://127.0.0.1:{spec.port}/{spec.name}",
    )


def _unwrap(spec: ToolSpec, payload: Any) -> Result:
    """Services following the team contract wrap results in an envelope
    {status, result}; a bare result object is accepted too. The presence of
    'result' is what decides which one this is, so a bare result may carry a
    'status' field of its own without being mistaken for a failed envelope."""
    if not isinstance(payload, dict):
        raise ToolError(f"{spec.name}: expected a JSON object, got {type(payload).__name__}")

    if "result" in payload:
        status = payload.get("status")
        if status not in (None, "success", "ok"):
            detail = payload.get("error") or payload
            raise ToolError(f"{spec.name}: service reported status={status!r} ({detail})")
        inner = payload["result"]
        if isinstance(inner, dict):
            return inner
        raise ToolError(
            f"{spec.name}: 'result' must be an object with fields {', '.join(spec.returns)}, "
            f"got {type(inner).__name__}. A bare list needs wrapping, e.g. "
            f'{{"result": {{"{next(iter(spec.returns))}": [...]}}}}'
        )

    if "error" in payload and any(f not in payload for f in spec.returns):
        # A failure without an envelope. Surface the service's own message
        # rather than reporting the fields it did not send.
        raise ToolError(f"{spec.name}: service reported an error: {payload['error']}")
    return payload


def _check_shape(spec: ToolSpec, result: Any) -> None:
    """Every declared field must be present and of its declared kind. A service
    answering with the wrong key names would otherwise read as a clean, empty
    result, and a 'n/a' in a number field would be handed on as a measurement."""
    if not isinstance(result, dict):
        raise ToolError(f"{spec.name}: result must be an object, got {type(result).__name__}")
    missing = [f for f in spec.returns if f not in result]
    if missing:
        raise ToolError(
            f"{spec.name}: response is missing {', '.join(missing)}; "
            f"expected fields: {', '.join(spec.returns)}"
        )
    for field, ret in spec.returns.items():
        value = result[field]
        if value is None and ret.nullable:
            continue
        ok = (
            isinstance(value, list) if ret.kind in LIST_KINDS
            else _is_number(value) if ret.kind == "number"
            else isinstance(value, str)
        )
        if not ok:
            expected = "a list" if ret.kind in LIST_KINDS else f"a {ret.kind}"
            raise ToolError(
                f"{spec.name}: '{field}' must be {expected}, got {type(value).__name__}"
            )


def _source_footprints(spec: ToolSpec, args: dict[str, Any], session: Session) -> list:
    """Footprint of each image this call was given.

    An unresolvable image_id raises instead of quietly disabling the check —
    that is exactly when a wrong CRS is most likely to slip through.
    """
    footprints = []
    for name, arg in spec.args.items():
        if arg.kind != "image_id":
            continue
        value = args.get(name)
        if value is None and not arg.required:
            continue
        if not isinstance(value, str) or not session.has(value):
            raise ToolError(
                f"{spec.name}: '{name}'={value!r} is not a loaded image, so its result "
                f"cannot be checked against a footprint"
            )
        footprints.append(geometry.footprint(session.get(value).bounds))
    return footprints


def _check_items(spec: ToolSpec, result: Result, footprints: list) -> None:
    for field in spec.list_fields:
        for i, item in enumerate(result[field]):
            _check_item(item, f"{spec.name}.{field}[{i}]", footprints)


def _http(spec: ToolSpec, args: dict[str, Any]) -> Result:
    import requests

    url = endpoint(spec)
    try:
        response = requests.post(url, json=args, timeout=HTTP_TIMEOUT_S)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise ToolError(f"{spec.name}: call to {url} failed ({exc})") from exc
    return _unwrap(spec, payload)


def call(
    name: str, args: dict[str, Any], session: Session, *, mocked_inputs: bool = False
) -> Result:
    """Run one tool and return its checked result.

    Mock results carry `mock: True`. A local tool's inputs may have come from a
    mock several steps upstream, and once a list has been filtered to nothing
    there is no way to tell from the data, so the executor passes what it knows
    through `mocked_inputs`. Provenance for a chain is the executor's to track.
    """
    spec = TOOLS.get(name)
    if spec is None:
        raise ToolError(f"unknown tool: {name}")
    if not isinstance(args, dict):
        raise ToolError(f"{name}: args must be an object, got {type(args).__name__}")

    missing = sorted(a for a in spec.required_args if args.get(a) is None)
    if missing:
        raise ToolError(f"{name}: missing required args: {', '.join(missing)}")
    for arg_name, arg in spec.args.items():
        value = args.get(arg_name)
        if arg.kind == "text" and value is not None and (not isinstance(value, str) or not value.strip()):
            raise ToolError(f"{name}: '{arg_name}' must be a non-empty string, got {value!r}")
    # Resolves every image_id before the adapter runs, so an unknown one fails
    # here rather than as a KeyError inside it, and never costs an HTTP round trip.
    footprints = _source_footprints(spec, args, session)

    try:
        if spec.local:
            result = LOCAL[name](args, session)
            if mocked_inputs:
                result["mock"] = True
        elif use_mock(name):
            result = MOCKS[name](args, session)
            result["mock"] = True
            if name not in _mock_warned:
                _mock_warned.add(name)
                warnings.warn(f"{name} is answering from a mock adapter", stacklevel=2)
        else:
            result = _http(spec, args)

        # Applied to mock and local output too, so a bad mock fails like a bad
        # service would.
        _check_shape(spec, result)
        _check_items(spec, result, footprints)
    except ToolError:
        raise
    except geometry.GeometryError as exc:
        raise ToolError(f"{name}: {exc}") from exc
    except Exception as exc:
        # An adapter bug has to surface as a failed step, not a crashed controller.
        raise ToolError(f"{name}: adapter failed ({type(exc).__name__}: {exc})") from exc

    return result
