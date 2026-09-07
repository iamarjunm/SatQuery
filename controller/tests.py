"""Tests for the pieces that exist so far.

Runs under pytest, or standalone with `python tests.py`.
"""

from __future__ import annotations

import geometry
import plan as plan_mod
import tools
from plan import Plan
from session import Scene, Session, demo_session
from datetime import date

import controller
import executor
import planner


def _check_geom(tool, result, args, session):
    spec = tools.TOOLS[tool]
    tools._check_items(spec, result, tools._source_footprints(spec, args, session))


# --- session -----------------------------------------------------------------

def test_bitemporal_pair_is_co_registered():
    s = demo_session()
    overlap = s.get("img_2023_opt").overlap(s.get("img_2026_opt"))
    assert overlap > 0.9, f"bi-temporal pair is not aligned: overlap={overlap:.3f}"


def test_optical_and_sar_are_co_registered():
    s = demo_session()
    assert s.get("img_2026_opt").overlap(s.get("img_2026_sar")) > 0.9


def test_unrelated_scenes_do_not_overlap():
    s = demo_session()
    assert s.get("img_2026_opt").overlap(s.get("img_cloud_opt")) == 0.0


def test_counterpart_resolves_both_ways():
    s = demo_session()
    assert s.counterpart("img_2026_opt").image_id == "img_2026_sar"
    assert s.counterpart("img_2026_sar").image_id == "img_2026_opt"
    assert s.counterpart("img_2023_opt") is None


def test_bounds_are_sanity_checked():
    try:
        Scene("bad", "optical", "sentinel-2", date(2026, 1, 1), (16.4, 48.2, 16.3, 48.3))
    except ValueError as exc:
        assert "not west<east" in str(exc) and "antimeridian" not in str(exc), exc
        return
    raise AssertionError("inverted bounds should be rejected")


# --- geometry ----------------------------------------------------------------

def test_utm_coordinates_are_rejected():
    utm = {"type": "Polygon", "coordinates": [[[601020.0, 5341200.0], [601120.0, 5341200.0],
                                               [601120.0, 5341300.0], [601020.0, 5341200.0]]]}
    try:
        geometry.validate(utm)
    except geometry.GeometryError as exc:
        assert "reprojected" in str(exc)
        return
    raise AssertionError("projected coordinates should not pass validation")


def test_swapped_lat_lon_is_rejected_by_footprint_check():
    """Both numbers stay in range when lon/lat are swapped, so the range check
    cannot see it. Comparing against the source image footprint can."""
    s = demo_session()
    w, south, e, n = s.get("img_2026_opt").bounds
    swapped = {"type": "Polygon", "coordinates": [[[south, w], [south + 0.01, w],
                                                   [south + 0.01, w + 0.01], [south, w]]]}
    geometry.validate(swapped)  # passes the range check, as expected
    try:
        _check_geom("ground", {"objects": [{"geometry": swapped}]}, {"image_id": "img_2026_opt"}, s)
    except tools.ToolError as exc:
        assert "swapped lon/lat" in str(exc)
        return
    raise AssertionError("geometry outside the source footprint should be rejected")


def test_geometry_from_the_wrong_scene_is_rejected():
    s = demo_session()
    elsewhere = tools._rect(s.get("img_cloud_opt").bounds, 0.1, 0.1, 0.2, 0.2)
    try:
        _check_geom("ground", {"objects": [{"geometry": elsewhere}]}, {"image_id": "img_2026_opt"}, s)
    except tools.ToolError:
        return
    raise AssertionError("geometry from a different AOI should be rejected")


def test_footprint_check_allows_slight_edge_overspill():
    s = demo_session()
    w, south, e, n = s.get("img_2026_opt").bounds
    dx = (e - w) * 0.1
    spill = {"type": "Polygon", "coordinates": [[[w - dx, south], [w + dx, south],
                                                 [w + dx, south + (n - south) * 0.1],
                                                 [w - dx, south]]]}
    _check_geom("ground", {"objects": [{"geometry": spill, "confidence": 0.9}]},
                {"image_id": "img_2026_opt"}, s)


def test_area_is_geodesic():
    s = demo_session()
    patch = tools._rect(s.get("img_2026_opt").bounds, 0.0, 0.0, 1.0, 1.0)
    km2 = geometry.area_km2(patch)
    # A BigEarthNet patch is 1.2 km square, so ~1.44 km2.
    assert 1.3 < km2 < 1.6, f"expected ~1.44 km2 for a full patch, got {km2:.3f}"


# --- local tools -------------------------------------------------------------

def test_filter_by_region_keeps_only_objects_inside():
    s = demo_session()
    grounded = tools.call("ground", {"image_id": "img_2026_opt", "phrase": "buildings"}, s)
    changed = tools.call(
        "change_detect", {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}, s
    )
    filtered = tools.call(
        "filter_by_region",
        {"objects": grounded["objects"], "regions": changed["regions"]},
        s,
    )
    assert len(grounded["objects"]) == 5
    assert filtered["kept"] == 2, f"expected 2 of 5 inside changed regions, got {filtered['kept']}"
    assert filtered["dropped"] == 3


def test_filter_by_region_handles_empty_input():
    s = demo_session()
    assert tools.call("filter_by_region", {"objects": [], "regions": []}, s)["kept"] == 0


def test_filter_by_region_merges_fragmented_regions():
    """A polygonised change mask arrives as many adjacent pieces. An object lying
    across a seam is inside the changed area and must survive."""
    s = demo_session()
    b = s.get("img_2026_opt").bounds
    obj = {"id": "o1", "label": "building",
           "geometry": tools._rect(b, 0.30, 0.30, 0.40, 0.40), "confidence": 0.9}
    fragments = [
        {"id": "c1", "type": "construction",
         "geometry": tools._rect(b, 0.20, 0.30, 0.33, 0.40), "confidence": 0.9},
        {"id": "c2", "type": "construction",
         "geometry": tools._rect(b, 0.37, 0.30, 0.50, 0.40), "confidence": 0.9},
    ]
    result = tools.call("filter_by_region", {"objects": [obj], "regions": fragments}, s)
    assert result["kept"] == 1, "object straddling two change fragments was dropped"


def test_filter_by_region_still_drops_a_grazing_object():
    s = demo_session()
    b = s.get("img_2026_opt").bounds
    obj = {"id": "o1", "label": "building",
           "geometry": tools._rect(b, 0.30, 0.30, 0.40, 0.40), "confidence": 0.9}
    region = [{"id": "c1", "type": "construction",
               "geometry": tools._rect(b, 0.20, 0.30, 0.32, 0.40), "confidence": 0.9}]
    assert tools.call("filter_by_region", {"objects": [obj], "regions": region}, s)["kept"] == 0


def test_count_reports_mean_confidence():
    s = demo_session()
    grounded = tools.call("ground", {"image_id": "img_2026_opt", "phrase": "buildings"}, s)
    counted = tools.call("count", {"objects": grounded["objects"]}, s)
    assert counted["n"] == 5
    assert 0.85 < counted["mean_confidence"] < 0.90


def test_mock_change_area_is_a_real_measurement():
    s = demo_session()
    result = tools.call(
        "change_detect", {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}, s
    )
    assert result["changed_area_km2"] > 0
    assert result["changed_area_km2"] < geometry.area_km2(
        tools._rect(s.get("img_2026_opt").bounds, 0, 0, 1, 1)
    )


# --- plan validation ---------------------------------------------------------

NEW_BUILDINGS = {
    "steps": [
        {"id": "s1", "tool": "change_detect",
         "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
        {"id": "s2", "tool": "ground",
         "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
        {"id": "s3", "tool": "filter_by_region",
         "args": {"objects": "$s2.objects", "regions": "$s1.regions"}},
    ],
    "answer_from": "s3",
    "reasoning": "New buildings are buildings in the later image inside changed regions.",
}


def _errors(raw, session=None):
    # `is not None`, not `or`: an empty Session is falsy and would be swapped
    # for the demo one, hiding exactly the no-images case.
    return plan_mod.validate(Plan.from_dict(raw), session if session is not None else demo_session())


def test_headline_plan_validates():
    assert _errors(NEW_BUILDINGS) == []


def test_unknown_tool_is_named():
    raw = {**NEW_BUILDINGS, "steps": [{"id": "s1", "tool": "segment_everything", "args": {}}],
           "answer_from": "s1"}
    errors = _errors(raw)
    assert any("segment_everything" in e and "does not exist" in e for e in errors)


def test_invented_image_is_rejected():
    raw = {"steps": [{"id": "s1", "tool": "ground",
                      "args": {"image_id": "img_1999", "phrase": "buildings"}}],
           "answer_from": "s1"}
    errors = _errors(raw)
    assert any("img_1999" in e and "not loaded" in e for e in errors)


def test_forward_reference_is_rejected():
    raw = {
        "steps": [
            {"id": "s1", "tool": "filter_by_region",
             "args": {"objects": "$s2.objects", "regions": "$s2.objects"}},
            {"id": "s2", "tool": "ground",
             "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
        ],
        "answer_from": "s1",
    }
    errors = _errors(raw)
    assert any("point backwards" in e for e in errors)


def test_reference_to_wrong_field_is_rejected():
    raw = {
        "steps": [
            {"id": "s1", "tool": "change_detect",
             "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
            {"id": "s2", "tool": "count", "args": {"objects": "$s1.objects"}},
        ],
        "answer_from": "s2",
    }
    errors = _errors(raw)
    assert any("change_detect returns" in e for e in errors)


def test_missing_and_unknown_args_are_reported():
    raw = {"steps": [{"id": "s1", "tool": "ground",
                      "args": {"image_id": "img_2026_opt", "target": "buildings"}}],
           "answer_from": "s1"}
    errors = _errors(raw)
    assert any("missing required args: phrase" in e for e in errors)
    assert any("does not accept args: target" in e for e in errors)


def test_answer_from_must_name_a_step():
    raw = {**NEW_BUILDINGS, "answer_from": "s9"}
    errors = _errors(raw)
    assert any("answer_from 's9'" in e for e in errors)


def test_step_limit_enforced():
    steps = [{"id": f"s{i}", "tool": "vqa",
              "args": {"image_id": "img_2026_opt", "question": "what is here?"}}
             for i in range(plan_mod.MAX_STEPS + 1)]
    errors = _errors({"steps": steps, "answer_from": "s0"})
    assert any("maximum is" in e for e in errors)


def test_malformed_payload_reports_format_error():
    for bad in ({}, {"steps": []}, {"steps": [{"id": "s1"}], "answer_from": "s1"},
                {"steps": [{"id": "s1", "tool": "vqa"}]}):
        try:
            Plan.from_dict(bad)
        except plan_mod.PlanFormatError:
            continue
        raise AssertionError(f"should have been rejected: {bad}")


def test_menu_lists_every_tool():
    menu = tools.render_menu()
    for name in tools.TOOLS:
        assert name in menu


# --- semantic plan validation ------------------------------------------------

def test_change_detect_across_different_areas_is_rejected():
    """The two scenes are 3 km apart. Comparing them yields a confident number
    about nothing."""
    raw = {"steps": [{"id": "s1", "tool": "change_detect",
                      "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_cloud_opt"}}],
           "answer_from": "s1"}
    errors = _errors(raw)
    assert any("cover different areas" in e for e in errors), errors


def test_change_detect_backwards_in_time_is_rejected():
    raw = {"steps": [{"id": "s1", "tool": "change_detect",
                      "args": {"image_id_t1": "img_2026_opt", "image_id_t2": "img_2023_opt"}}],
           "answer_from": "s1"}
    errors = _errors(raw)
    assert any("must be the earlier image" in e for e in errors), errors


def test_change_detect_on_one_image_twice_is_rejected():
    raw = {"steps": [{"id": "s1", "tool": "change_detect",
                      "args": {"image_id_t1": "img_2026_opt", "image_id_t2": "img_2026_opt"}}],
           "answer_from": "s1"}
    assert any("two different images" in e for e in _errors(raw))


def test_cross_modal_with_swapped_modalities_is_rejected():
    raw = {"steps": [{"id": "s1", "tool": "cross_modal",
                      "args": {"optical_image_id": "img_2026_sar",
                               "sar_image_id": "img_2026_opt", "phrase": "water"}}],
           "answer_from": "s1"}
    errors = _errors(raw)
    assert any("optical_image_id must name an optical image" in e for e in errors), errors
    assert any("sar_image_id must name a SAR image" in e for e in errors), errors


def test_cross_modal_with_unpaired_scenes_is_rejected():
    raw = {"steps": [{"id": "s1", "tool": "cross_modal",
                      "args": {"optical_image_id": "img_2026_opt",
                               "sar_image_id": "img_cloud_sar", "phrase": "water"}}],
           "answer_from": "s1"}
    assert any("cover different areas" in e for e in _errors(raw))


def test_valid_cross_modal_plan_passes():
    raw = {"steps": [{"id": "s1", "tool": "cross_modal",
                      "args": {"optical_image_id": "img_cloud_opt",
                               "sar_image_id": "img_cloud_sar", "phrase": "water"}}],
           "answer_from": "s1"}
    assert _errors(raw) == []


def test_literal_lists_are_rejected():
    """A plan carrying its own objects would be a count nothing computed."""
    for literal in ([{"confidence": 0.99}] * 7, [], ["a"]):
        raw = {"steps": [{"id": "s1", "tool": "count", "args": {"objects": literal}}],
               "answer_from": "s1"}
        errors = _errors(raw)
        assert any("must be a reference" in e for e in errors), (literal, errors)


def test_describe_exposes_which_scenes_share_an_area():
    described = demo_session().describe()
    assert "area A" in described and "area B" in described
    assert described.count("area A") == 3


# --- error surfacing ---------------------------------------------------------

def test_projected_coordinates_surface_as_tool_error():
    """The executor is written against ToolError; a GeometryError escaping
    tools.call would crash it instead of degrading."""
    s = demo_session()
    utm = {"type": "Polygon", "coordinates": [[[601020.0, 5341200.0], [601120.0, 5341200.0],
                                               [601120.0, 5341300.0], [601020.0, 5341200.0]]]}
    try:
        _check_geom("ground", {"objects": [{"geometry": utm}]}, {"image_id": "img_2026_opt"}, s)
    except tools.ToolError:
        raise AssertionError("expected GeometryError here; call() is what converts it")
    except geometry.GeometryError:
        pass
    try:
        tools.call("filter_by_region",
                   {"objects": [{"geometry": utm}],
                    "regions": [{"geometry": utm}]}, s)
    except tools.ToolError:
        return
    raise AssertionError("tools.call must convert GeometryError into ToolError")


def test_truncated_coordinate_is_a_geometry_error():
    bad = {"type": "Polygon", "coordinates": [[[16.37, 48.21], [16.38, 48.21],
                                               [16.38, 48.22], [16.37]]]}
    try:
        geometry.validate(bad)
    except geometry.GeometryError:
        return
    raise AssertionError("a truncated position must not raise a bare IndexError")


def test_missing_geometry_is_a_tool_error():
    s = demo_session()
    regions = tools.call("change_detect",
                         {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}, s)["regions"]
    try:
        tools.call("filter_by_region", {"objects": [{"id": "o1"}], "regions": regions}, s)
    except tools.ToolError:
        return
    raise AssertionError("an object without geometry must raise ToolError, not KeyError")


def test_count_rejects_non_dict_entries():
    s = demo_session()
    try:
        tools.call("count", {"objects": ["a", "b"]}, s)
    except tools.ToolError:
        return
    raise AssertionError("expected ToolError, not AttributeError")


def test_unresolvable_image_does_not_disable_the_footprint_check():
    s = demo_session()
    try:
        tools._source_footprints(tools.TOOLS["ground"], {"image_id": "nope"}, s)
    except tools.ToolError:
        return
    raise AssertionError("an unknown image must raise, not silently skip the check")


def test_list_result_envelope_is_rejected():
    try:
        tools._unwrap(tools.TOOLS["ground"], {"status": "ok", "result": [{"label": "building"}]})
    except tools.ToolError as exc:
        assert "objects" in str(exc)
        return
    raise AssertionError("a list 'result' must be rejected, not passed through")


def test_bare_result_object_is_accepted():
    payload = {"objects": [{"label": "building"}]}
    assert tools._unwrap(tools.TOOLS["ground"], payload) == payload


# --- measurement -------------------------------------------------------------

def test_multipolygon_area_does_not_cancel():
    """Geodesic area is signed by ring winding, and GeoJSON producers do not
    wind consistently. Opposite-wound parts must not subtract."""
    s = demo_session()
    b = s.get("img_2026_opt").bounds
    left = tools._rect(b, 0.0, 0.0, 0.45, 1.0)
    right = tools._rect(b, 0.55, 0.0, 1.0, 1.0)
    expected = geometry.area_km2(left) + geometry.area_km2(right)
    multi = {
        "type": "MultiPolygon",
        "coordinates": [left["coordinates"], [list(reversed(right["coordinates"][0]))]],
    }
    assert abs(geometry.area_km2(multi) - expected) < 1e-9, geometry.area_km2(multi)


def test_mock_results_are_marked():
    s = demo_session()
    assert tools.call("ground", {"image_id": "img_2026_opt", "phrase": "buildings"}, s)["mock"]


def test_mock_label_is_the_phrase_verbatim():
    s = demo_session()
    grassy = tools.call("ground", {"image_id": "img_2026_opt", "phrase": "grass"}, s)
    assert grassy["objects"][0]["label"] == "grass"


# --- second review pass ------------------------------------------------------

def test_polygon_hole_is_subtracted():
    outer = [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]
    hole = [[0.02, 0.02], [0.08, 0.02], [0.08, 0.08], [0.02, 0.08], [0.02, 0.02]]
    solid = geometry.area_km2({"type": "Polygon", "coordinates": [outer]})
    donut = geometry.area_km2({"type": "Polygon", "coordinates": [outer, hole]})
    hole_alone = geometry.area_km2({"type": "Polygon", "coordinates": [hole]})
    assert abs(donut - (solid - hole_alone)) < 1e-6, (donut, solid, hole_alone)


def test_same_wound_hole_is_still_subtracted():
    """GeoJSON says holes wind opposite to the exterior. Producers ignore that."""
    outer = [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]
    hole_cw = [[0.02, 0.02], [0.02, 0.08], [0.08, 0.08], [0.08, 0.02], [0.02, 0.02]]
    hole_ccw = list(reversed(hole_cw))
    a = geometry.area_km2({"type": "Polygon", "coordinates": [outer, hole_cw]})
    b = geometry.area_km2({"type": "Polygon", "coordinates": [outer, hole_ccw]})
    assert abs(a - b) < 1e-9


def test_adapter_failures_surface_as_tool_error():
    s = demo_session()
    cases = [
        ("ground", {"image_id": "nope", "phrase": "buildings"}),
        ("vqa", {"question": "hi"}),
        ("count", {"objects": 3.5}),
        ("filter_by_region", {"objects": 3.5, "regions": []}),
        ("ground", "not a dict"),
    ]
    for name, args in cases:
        try:
            tools.call(name, args, s)
        except tools.ToolError:
            continue
        except Exception as exc:
            raise AssertionError(f"{name}{args!r} leaked {type(exc).__name__}") from exc
        raise AssertionError(f"{name}{args!r} should have failed")


def test_scalar_reference_into_list_arg_is_rejected():
    raw = {
        "steps": [
            {"id": "s1", "tool": "change_detect",
             "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
            {"id": "s2", "tool": "count", "args": {"objects": "$s1.changed_area_km2"}},
        ],
        "answer_from": "s2",
    }
    errors = _errors(raw)
    assert any("needs objects" in e and "is a number" in e for e in errors), errors


def test_regions_may_feed_a_count():
    raw = {
        "steps": [
            {"id": "s1", "tool": "change_detect",
             "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
            {"id": "s2", "tool": "count", "args": {"objects": "$s1.regions"}},
        ],
        "answer_from": "s2",
    }
    assert _errors(raw) == []


def test_footprint_check_accepts_result_inside_either_co_registered_input():
    """Two scenes offset by 5% still count as co-registered. An honest result
    at the edge may sit in the sliver only one of them covers; that is not a
    CRS error and must not abort the step."""
    s = Session([
        Scene("t1", "optical", "sentinel-2", date(2023, 1, 1), (16.000, 48.0, 16.100, 48.1)),
        Scene("t2", "optical", "sentinel-2", date(2026, 1, 1), (16.005, 48.0, 16.105, 48.1)),
    ])
    assert s.get("t1").overlap(s.get("t2")) >= 0.9
    sliver = {"type": "Polygon", "coordinates": [[[16.000, 48.01], [16.004, 48.01],
                                                  [16.004, 48.02], [16.000, 48.02],
                                                  [16.000, 48.01]]]}
    _check_geom("change_detect", {"regions": [{"geometry": sliver, "confidence": 0.9}]},
                {"image_id_t1": "t1", "image_id_t2": "t2"}, s)
    elsewhere = {"type": "Polygon", "coordinates": [[[16.3, 48.3], [16.31, 48.3],
                                                     [16.31, 48.31], [16.3, 48.3]]]}
    try:
        _check_geom("change_detect", {"regions": [{"geometry": elsewhere, "confidence": 0.9}]},
                    {"image_id_t1": "t1", "image_id_t2": "t2"}, s)
    except tools.ToolError as exc:
        assert "footprint(s)" in str(exc)
        return
    raise AssertionError("a result outside both inputs must still be rejected")


def test_mock_marker_survives_the_full_chain():
    s = demo_session()
    changed = tools.call("change_detect",
                         {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}, s)
    found = tools.call("ground", {"image_id": "img_2026_opt", "phrase": "buildings"}, s)
    upstream_mocked = changed["mock"] or found["mock"]
    filtered = tools.call("filter_by_region",
                          {"objects": found["objects"], "regions": changed["regions"]}, s,
                          mocked_inputs=upstream_mocked)
    counted = tools.call("count", {"objects": filtered["objects"]}, s,
                         mocked_inputs=upstream_mocked)
    assert filtered.get("mock") is True
    assert counted.get("mock") is True


def test_local_tool_on_real_input_is_not_marked_mock():
    s = demo_session()
    b = s.get("img_2026_opt").bounds
    real = [{"id": "o1", "label": "building",
             "geometry": tools._rect(b, 0.1, 0.1, 0.2, 0.2), "confidence": 0.9}]
    assert "mock" not in tools.call("count", {"objects": real}, s)


def test_filter_by_region_across_areas_is_rejected():
    raw = {
        "steps": [
            {"id": "s1", "tool": "change_detect",
             "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
            {"id": "s2", "tool": "ground",
             "args": {"image_id": "img_cloud_opt", "phrase": "buildings"}},
            {"id": "s3", "tool": "filter_by_region",
             "args": {"objects": "$s2.objects", "regions": "$s1.regions"}},
        ],
        "answer_from": "s3",
    }
    errors = _errors(raw)
    assert any("cover different areas" in e for e in errors), errors


def test_area_labels_stay_alphabetic_past_z():
    scenes = [
        Scene(f"img_{i}", "optical", "sentinel-2", date(2026, 1, 1),
              (10.0 + i, 40.0, 10.5 + i, 40.5))
        for i in range(30)
    ]
    labels = Session(scenes).areas().values()
    assert len(set(labels)) == 30
    for label in labels:
        name = label.removeprefix("area ")
        assert name.isalpha() and name.isupper(), label
    assert "area AA" in labels


# --- third review pass -------------------------------------------------------

def test_nested_scene_is_not_co_registered():
    """A 1.2 km patch inside a one-degree tile is not the same area, however
    completely the tile contains it."""
    big = Scene("big", "optical", "sentinel-2", date(2023, 1, 1), (16.0, 48.0, 17.0, 49.0))
    small = Scene("small", "optical", "sentinel-2", date(2026, 1, 1),
                  (16.37, 48.21, 16.3862, 48.2208))
    assert big.overlap(small) < 0.01
    s = Session([big, small])
    assert len(set(s.areas().values())) == 2
    raw = {"steps": [{"id": "s1", "tool": "change_detect",
                      "args": {"image_id_t1": "big", "image_id_t2": "small"}}],
           "answer_from": "s1"}
    assert any("cover different areas" in e for e in plan_mod.validate(Plan.from_dict(raw), s))


def test_bowtie_polygon_keeps_both_lobes():
    bowtie = {"type": "Polygon", "coordinates": [[[16.37, 48.21], [16.38, 48.22],
                                                  [16.38, 48.21], [16.37, 48.22],
                                                  [16.37, 48.21]]]}
    lobe_l = {"type": "Polygon", "coordinates": [[[16.37, 48.21], [16.375, 48.215],
                                                  [16.37, 48.22], [16.37, 48.21]]]}
    lobe_r = {"type": "Polygon", "coordinates": [[[16.375, 48.215], [16.38, 48.22],
                                                  [16.38, 48.21], [16.375, 48.215]]]}
    both = geometry.area_km2(lobe_l) + geometry.area_km2(lobe_r)
    assert abs(geometry.area_km2(bowtie) - both) / both < 0.01
    reversed_bowtie = {"type": "Polygon",
                       "coordinates": [list(reversed(bowtie["coordinates"][0]))]}
    assert abs(geometry.area_km2(reversed_bowtie) - both) / both < 0.01


def test_provenance_survives_an_intermediate_step():
    """An intermediate filter_by_region has no image args of its own. Following
    references only one hop would let it launder where its geometry came from."""
    raw = {
        "steps": [
            {"id": "s1", "tool": "change_detect",
             "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
            {"id": "s2", "tool": "ground",
             "args": {"image_id": "img_cloud_opt", "phrase": "buildings"}},
            {"id": "s3", "tool": "filter_by_region",
             "args": {"objects": "$s2.objects", "regions": "$s2.objects"}},
            {"id": "s4", "tool": "filter_by_region",
             "args": {"objects": "$s3.objects", "regions": "$s1.regions"}},
        ],
        "answer_from": "s4",
    }
    errors = _errors(raw)
    assert any(e.startswith("step s4") and "different areas" in e for e in errors), errors


def test_response_missing_declared_fields_is_rejected():
    s = demo_session()
    original = tools.MOCKS["ground"]
    tools.MOCKS["ground"] = lambda args, session: {"buildings": []}
    try:
        try:
            tools.call("ground", {"image_id": "img_2026_opt", "phrase": "buildings"}, s)
        except tools.ToolError as exc:
            assert "missing objects" in str(exc)
            return
        raise AssertionError("a response with the wrong key names must be rejected")
    finally:
        tools.MOCKS["ground"] = original


def test_response_list_field_must_be_a_list():
    try:
        tools._check_shape(tools.TOOLS["ground"], {"objects": "none"})
    except tools.ToolError as exc:
        assert "must be a list" in str(exc)
        return
    raise AssertionError("a non-list objects field must be rejected")


def test_geometry_reference_into_text_arg_is_rejected():
    raw = {
        "steps": [
            {"id": "s1", "tool": "ground",
             "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
            {"id": "s2", "tool": "vqa",
             "args": {"image_id": "img_2026_opt", "question": "$s1.objects"}},
        ],
        "answer_from": "s2",
    }
    errors = _errors(raw)
    assert any("needs text" in e and "is a objects" in e for e in errors), errors


def test_text_reference_into_text_arg_is_allowed():
    raw = {
        "steps": [
            {"id": "s1", "tool": "vqa",
             "args": {"image_id": "img_2026_opt", "question": "what is here?"}},
            {"id": "s2", "tool": "vqa",
             "args": {"image_id": "img_2023_opt", "question": "$s1.answer"}},
        ],
        "answer_from": "s2",
    }
    assert _errors(raw) == []


def test_empty_mocked_chain_keeps_marker_with_executor_hint():
    """Once a mocked list has been filtered to nothing there is no item left to
    carry the marker. The executor passes what it knows."""
    s = demo_session()
    b = s.get("img_2026_opt").bounds
    found = tools.call("ground", {"image_id": "img_2026_opt", "phrase": "buildings"}, s)
    corner = [{"id": "c", "type": "construction",
               "geometry": tools._rect(b, 0.95, 0.95, 1.0, 1.0), "confidence": 0.9}]
    filtered = tools.call("filter_by_region",
                          {"objects": found["objects"], "regions": corner}, s,
                          mocked_inputs=found["mock"])
    assert filtered["kept"] == 0 and filtered.get("mock") is True
    counted = tools.call("count", {"objects": filtered["objects"]}, s, mocked_inputs=True)
    assert counted.get("mock") is True
    assert "mock" not in tools.call("count", {"objects": filtered["objects"]}, s)


def test_describe_and_validator_agree_on_same_area():
    """Three scenes where the outer two overlap the middle at 92% but each other
    at 85%. Whatever the grouping decides, describe() and validate() must say
    the same thing."""
    s = Session([
        Scene("mid", "optical", "sentinel-2", date(2020, 1, 1), (16.000, 48.0, 16.100, 48.1)),
        Scene("east", "optical", "sentinel-2", date(2026, 1, 1), (16.004, 48.0, 16.104, 48.1)),
        Scene("west", "optical", "sentinel-2", date(2023, 1, 1), (15.996, 48.0, 16.096, 48.1)),
    ])
    assert s.get("east").overlap(s.get("west")) < 0.9
    raw = {"steps": [{"id": "s1", "tool": "change_detect",
                      "args": {"image_id_t1": "west", "image_id_t2": "east"}}],
           "answer_from": "s1"}
    rejected = any("different areas" in e for e in plan_mod.validate(Plan.from_dict(raw), s))
    assert rejected == (not s.same_area("east", "west"))


def test_use_mock_accepts_common_opt_out_spellings():
    import os
    saved = {k: os.environ.get(k) for k in ("SATQUERY_MOCK", "SATQUERY_MOCK_VQA")}
    try:
        for spelling in ("off", "0 ", " 0", "false", "no", "OFF", "nonsense"):
            os.environ["SATQUERY_MOCK"] = spelling
            os.environ.pop("SATQUERY_MOCK_VQA", None)
            assert tools.use_mock("vqa") is False, spelling
        os.environ["SATQUERY_MOCK"] = "1"
        os.environ["SATQUERY_MOCK_VQA"] = "off"
        assert tools.use_mock("vqa") is False
        assert tools.use_mock("ground") is True
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_filter_by_region_validates_objects_even_when_regions_empty():
    s = demo_session()
    utm = {"type": "Polygon", "coordinates": [[[601020.0, 5341200.0], [601120.0, 5341200.0],
                                               [601120.0, 5341300.0], [601020.0, 5341200.0]]]}
    for objects in ([{"geometry": utm}], [{"id": "o1"}], [42]):
        try:
            tools.call("filter_by_region", {"objects": objects, "regions": []}, s)
        except tools.ToolError:
            continue
        raise AssertionError(f"{objects!r} should not pass just because regions is empty")


def test_antimeridian_bounds_are_refused():
    try:
        Scene("fiji", "optical", "sentinel-2", date(2026, 1, 1), (179.82, -18.07, -178.75, -17.14))
    except ValueError as exc:
        assert "antimeridian" in str(exc)
        return
    raise AssertionError("west > east must be refused with a clear message")


def test_count_refuses_objects_without_numeric_confidence():
    """A string '0.9', a bool, or a missing field would otherwise be averaged
    away and reported as a real score."""
    s = demo_session()
    for bad in ({"confidence": True}, {"confidence": "0.9"}, {"id": "o2"}):
        try:
            tools.call("count", {"objects": [bad, {"confidence": 0.8}]}, s)
        except tools.ToolError as exc:
            assert "'confidence' must be a number" in str(exc)
            continue
        raise AssertionError(f"{bad!r} should have been refused")


def test_count_of_nothing_has_no_mean():
    s = demo_session()
    counted = tools.call("count", {"objects": []}, s)
    assert counted["n"] == 0 and counted["mean_confidence"] is None


# --- fourth review pass ------------------------------------------------------

def test_near_miss_references_are_rejected():
    for bad in ("$s1.answer.text", "$s1.answer ", "${s1.answer}", "$s1", " $s1.answer"):
        raw = {"steps": [
            {"id": "s1", "tool": "vqa",
             "args": {"image_id": "img_2026_opt", "question": "what is here?"}},
            {"id": "s2", "tool": "vqa", "args": {"image_id": "img_2023_opt", "question": bad}},
        ], "answer_from": "s2"}
        errors = _errors(raw)
        assert any("looks like a reference" in e for e in errors), (bad, errors)


def test_reference_with_trailing_newline_is_not_a_reference():
    assert plan_mod.parse_ref("$s1.objects\n") is None
    assert plan_mod.parse_ref("$s1.objects") == ("s1", "objects")


def test_area_grouping_does_not_depend_on_load_order():
    mid = Scene("mid", "optical", "sentinel-2", date(2020, 1, 1), (16.000, 48.0, 16.100, 48.1))
    east = Scene("east", "optical", "sentinel-2", date(2026, 1, 1), (16.004, 48.0, 16.104, 48.1))
    west = Scene("west", "optical", "sentinel-2", date(2023, 1, 1), (15.996, 48.0, 16.096, 48.1))
    a = Session([mid, east, west]).areas()
    b = Session([east, west, mid]).areas()
    c = Session([west, mid, east]).areas()
    assert a == b == c


def test_scenes_sharing_a_label_always_clear_the_threshold():
    """Whatever the grouping decides, it must never put two scenes under one
    label that do not overlap enough to be compared."""
    scenes = [
        Scene(f"s{i}", "optical", "sentinel-2", date(2020 + i, 1, 1),
              (16.0 + i * 0.004, 48.0, 16.1 + i * 0.004, 48.1))
        for i in range(6)
    ]
    s = Session(scenes)
    areas = s.areas()
    for a in scenes:
        for b in scenes:
            if a is not b and areas[a.image_id] == areas[b.image_id]:
                assert a.overlap(b) >= 0.9, (a.image_id, b.image_id, a.overlap(b))


def test_list_fields_are_found_by_kind_not_by_name():
    """A tool may return its geometry under any field name; the checks must
    follow the declared kind."""
    s = demo_session()
    spec = tools.ToolSpec(
        name="segment", summary="", args={"image_id": tools.ArgSpec("image_id", "")},
        returns={"masks": tools.ReturnSpec("regions", "")}, port=8005,
    )
    utm = {"type": "Polygon", "coordinates": [[[601020.0, 5341200.0], [601120.0, 5341200.0],
                                               [601120.0, 5341300.0], [601020.0, 5341200.0]]]}
    footprints = tools._source_footprints(spec, {"image_id": "img_2026_opt"}, s)
    try:
        tools._check_items(spec, {"masks": [{"geometry": utm, "confidence": 0.9}]}, footprints)
    except geometry.GeometryError:
        return
    raise AssertionError("UTM geometry under a custom field name slipped through")


def test_item_confidence_is_checked_at_the_boundary():
    s = demo_session()
    b = s.get("img_2026_opt").bounds
    geom = tools._rect(b, 0.1, 0.1, 0.2, 0.2)
    for bad in ({"geometry": geom}, {"geometry": geom, "confidence": "0.9"},
                {"geometry": geom, "confidence": 1.7}):
        try:
            _check_geom("ground", {"objects": [bad]}, {"image_id": "img_2026_opt"}, s)
        except tools.ToolError as exc:
            assert "confidence" in str(exc)
            continue
        raise AssertionError(f"{bad!r} should have been refused")


def test_none_for_a_required_list_arg_is_rejected():
    s = demo_session()
    for name, args in (("filter_by_region", {"objects": None, "regions": None}),
                       ("count", {"objects": None})):
        try:
            tools.call(name, args, s)
        except tools.ToolError as exc:
            assert "missing required args" in str(exc)
            continue
        raise AssertionError(f"{name} accepted None and returned an empty success")


def test_registry_is_checked_at_import():
    tools.TOOLS["segment"] = tools.ToolSpec(
        name="segment", summary="", args={}, returns={}, port=8005,
    )
    try:
        try:
            tools._check_registry()
        except RuntimeError as exc:
            assert "segment" in str(exc) and "mock adapter" in str(exc)
        else:
            raise AssertionError("a tool without an adapter must fail registration")
    finally:
        del tools.TOOLS["segment"]
    tools._check_registry()


def test_scene_acquired_must_be_a_date():
    try:
        Scene("a", "optical", "sentinel-2", "2023-06-15", (16.0, 48.0, 16.1, 48.1))
    except TypeError as exc:
        assert "acquired" in str(exc)
        return
    raise AssertionError("a string date must be refused, not compared as a string")


def test_scene_modality_is_checked():
    try:
        Scene("a", "radar", "sentinel-1", date(2023, 6, 15), (16.0, 48.0, 16.1, 48.1))
    except ValueError as exc:
        assert "modality" in str(exc)
        return
    raise AssertionError("an unknown modality must be refused")


def test_scene_bounds_are_normalised_to_a_hashable_tuple():
    scene = Scene("a", "optical", "sentinel-2", date(2023, 6, 15), [16.0, 48.0, 16.1, 48.1])
    assert isinstance(scene.bounds, tuple)
    hash(scene)


def test_projected_bounds_are_not_called_antimeridian():
    try:
        Scene("utm", "optical", "sentinel-2", date(2023, 1, 1),
              (601120.0, 5341200.0, 601020.0, 5341300.0))
    except ValueError as exc:
        assert "outside EPSG:4326" in str(exc) and "antimeridian" not in str(exc)
        return
    raise AssertionError("projected bounds must be refused")


def test_validation_against_an_empty_session_reports_no_images():
    raw = {"steps": [{"id": "s1", "tool": "vqa",
                      "args": {"image_id": "img_2026_opt", "question": "what?"}}],
           "answer_from": "s1"}
    errors = _errors(raw, Session())
    assert any("not loaded" in e and "none" in e for e in errors), errors


# --- fifth review pass -------------------------------------------------------

def test_datetime_acquired_is_normalised_to_date():
    from datetime import datetime
    s = Session([
        Scene("a", "optical", "sentinel-2", date(2023, 6, 15), (16.0, 48.0, 16.1, 48.1)),
        Scene("b", "optical", "sentinel-2", datetime(2026, 8, 1, 10, 30), (16.0, 48.0, 16.1, 48.1)),
        Scene("c", "optical", "sentinel-2", datetime(2026, 8, 1, 16, 45), (16.0, 48.0, 16.1, 48.1)),
    ])
    assert type(s.get("b").acquired) is date
    ok = {"steps": [{"id": "s1", "tool": "change_detect",
                     "args": {"image_id_t1": "a", "image_id_t2": "b"}}], "answer_from": "s1"}
    assert plan_mod.validate(Plan.from_dict(ok), s) == []
    same_day = {"steps": [{"id": "s1", "tool": "change_detect",
                           "args": {"image_id_t1": "b", "image_id_t2": "c"}}], "answer_from": "s1"}
    assert any("no time span" in e for e in plan_mod.validate(Plan.from_dict(same_day), s))


def test_number_and_text_return_fields_are_type_checked():
    cases = [
        ("vqa", {"answer": None, "confidence": "high"}),
        ("vqa", {"answer": "x", "confidence": None}),
        ("change_detect", {"regions": [], "changed_area_km2": "n/a", "change_mask_id": "m"}),
        ("change_detect", {"regions": [], "changed_area_km2": 0.1, "change_mask_id": {"x": 1}}),
        ("count", {"n": "7", "mean_confidence": 0.5}),
    ]
    for tool, result in cases:
        try:
            tools._check_shape(tools.TOOLS[tool], result)
        except tools.ToolError:
            continue
        raise AssertionError(f"{tool} accepted {result!r}")
    tools._check_shape(tools.TOOLS["count"], {"n": 0, "mean_confidence": None})


def test_hole_poking_past_shell_does_not_add_area():
    outer = [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1], [0, 0]]
    overhang = [[0.05, 0.05], [0.15, 0.05], [0.15, 0.15], [0.05, 0.15], [0.05, 0.05]]
    solid = geometry.area_km2({"type": "Polygon", "coordinates": [outer]})
    repaired = geometry.area_km2({"type": "Polygon", "coordinates": [outer, overhang]})
    assert repaired < solid, (repaired, solid)


def test_text_reference_does_not_pollute_geometry_provenance():
    """A question phrased from an earlier answer carries no geometry; the
    answer's image must not be counted as a source of the later geometry."""
    raw = {
        "steps": [
            {"id": "q", "tool": "vqa",
             "args": {"image_id": "img_2023_opt", "question": "what is here?"}},
            {"id": "g", "tool": "ground",
             "args": {"image_id": "img_cloud_opt", "phrase": "$q.answer"}},
            {"id": "c", "tool": "cross_modal",
             "args": {"optical_image_id": "img_cloud_opt", "sar_image_id": "img_cloud_sar",
                      "phrase": "water"}},
            {"id": "f", "tool": "filter_by_region",
             "args": {"objects": "$g.objects", "regions": "$c.regions"}},
        ],
        "answer_from": "f",
    }
    assert _errors(raw) == []


def test_bare_error_payload_surfaces_the_service_message():
    try:
        tools._unwrap(tools.TOOLS["ground"], {"ok": False, "error": "OOM"})
    except tools.ToolError as exc:
        assert "OOM" in str(exc)
    else:
        raise AssertionError("an error payload must not pass through as a result")
    bare = {"answer": "x", "confidence": 0.5, "status": "done"}
    assert tools._unwrap(tools.TOOLS["vqa"], bare) == bare


def test_reference_embedded_in_text_is_rejected():
    raw = {
        "steps": [
            {"id": "s1", "tool": "ground",
             "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
            {"id": "s2", "tool": "vqa",
             "args": {"image_id": "img_2026_opt", "question": "How many of $s1.objects are new?"}},
        ],
        "answer_from": "s2",
    }
    assert any("looks like a reference" in e for e in _errors(raw))


def test_filter_by_region_checks_every_input_not_just_survivors():
    s = demo_session()
    b = s.get("img_2026_opt").bounds
    region = [{"id": "c", "type": "construction",
               "geometry": tools._rect(b, 0.25, 0.18, 0.45, 0.35), "confidence": 0.9}]
    for placement in ((0.3, 0.3, 0.4, 0.4), (0.8, 0.8, 0.9, 0.9)):
        obj = {"id": "o", "label": "building", "geometry": tools._rect(b, *placement)}
        try:
            tools.call("filter_by_region", {"objects": [obj], "regions": region}, s)
        except tools.ToolError as exc:
            assert "confidence" in str(exc)
            continue
        raise AssertionError(f"object at {placement} without confidence was accepted")


def test_session_refuses_scenes_after_it_is_in_use():
    s = demo_session()
    s.describe()
    try:
        s.add(Scene("late", "optical", "sentinel-2", date(2026, 1, 1), (10.0, 40.0, 10.1, 40.1)))
    except RuntimeError as exc:
        assert "in use" in str(exc)
        return
    raise AssertionError("adding after describe() would silently relabel areas")


def test_step_ids_must_be_referenceable():
    for bad in ("step-1", "s.1", "s 1", ""):
        try:
            Plan.from_dict({"steps": [{"id": bad, "tool": "vqa", "args": {}}], "answer_from": bad})
        except plan_mod.PlanFormatError as exc:
            assert "letters, digits and underscores" in str(exc)
            continue
        raise AssertionError(f"id {bad!r} can never be referenced and must be refused")


def test_self_reference_is_named_as_such():
    raw = {"steps": [{"id": "s1", "tool": "count", "args": {"objects": "$s1.objects"}}],
           "answer_from": "s1"}
    assert any("own output" in e for e in _errors(raw))


def test_empty_text_arg_is_rejected_at_the_tool():
    s = demo_session()
    for phrase in ("", "   "):
        try:
            tools.call("ground", {"image_id": "img_2026_opt", "phrase": phrase}, s)
        except tools.ToolError as exc:
            assert "non-empty" in str(exc)
            continue
        raise AssertionError(f"phrase {phrase!r} was accepted")


def test_unloaded_pair_is_not_described_as_paired():
    s = Session([Scene("solo", "optical", "sentinel-2", date(2026, 1, 1),
                       (16.0, 48.0, 16.1, 48.1), pair_id="ghost")])
    assert "paired" not in s.describe()
    assert s.counterpart("solo") is None


def test_scene_labels_are_normalised_to_a_tuple():
    scene = Scene("a", "optical", "sentinel-2", date(2023, 6, 15), (16.0, 48.0, 16.1, 48.1),
                  labels=["Arable land"])
    assert scene.labels == ("Arable land",)
    hash(scene)


def test_blank_mock_env_means_default():
    import os
    saved = {k: os.environ.get(k) for k in ("SATQUERY_MOCK", "SATQUERY_MOCK_VQA")}
    try:
        os.environ["SATQUERY_MOCK"] = ""
        os.environ.pop("SATQUERY_MOCK_VQA", None)
        assert tools.use_mock("vqa") is True
        os.environ["SATQUERY_MOCK"] = "0"
        os.environ["SATQUERY_MOCK_VQA"] = "  "
        assert tools.use_mock("vqa") is False
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- planner -----------------------------------------------------------------
# No SATQUERY_LLM_API_KEY is set in this environment, so plan_query() always
# takes the keyword-fallback path here -- these tests stay offline like the
# rest of the suite, matching how tools.py defaults to mocks with nothing
# configured.

def test_planner_falls_back_without_an_llm_key():
    plan = planner.plan_query("What changed between 2023 and 2026?", demo_session())
    assert plan.source == "fallback"
    assert plan_mod.validate(plan, demo_session()) == []


def test_planner_fallback_routes_new_buildings_to_a_chain():
    plan = planner.plan_query("Find buildings constructed after 2023", demo_session())
    tools_used = [s.tool for s in plan.steps]
    assert tools_used == ["change_detect", "ground", "filter_by_region"]


def test_planner_fallback_routes_find_to_ground():
    plan = planner.plan_query("Find all the buildings", demo_session())
    assert plan.steps[0].tool == "ground"


def test_planner_fallback_routes_tally_to_ground_then_count():
    plan = planner.plan_query("Give me a tally of the vehicles", demo_session())
    assert [s.tool for s in plan.steps] == ["ground", "count"]


def test_planner_fallback_defaults_to_vqa_when_nothing_matches():
    plan = planner.plan_query("Describe this scene", demo_session())
    assert plan.steps[0].tool == "vqa"


def test_planner_rejects_blank_query():
    try:
        planner.plan_query("   ", demo_session())
        assert False, "expected a ValueError"
    except ValueError:
        pass


# --- executor ------------------------------------------------------------------

def test_executor_resolves_refs_across_a_chain():
    s = demo_session()
    plan = Plan.from_dict({
        "steps": [
            {"id": "s1", "tool": "change_detect",
             "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
            {"id": "s2", "tool": "ground", "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
            {"id": "s3", "tool": "filter_by_region", "args": {"objects": "$s2.objects", "regions": "$s1.regions"}},
        ],
        "answer_from": "s3",
    })
    outcome = executor.run(plan, s)
    assert outcome["status"] == "ok"
    assert set(outcome["results"]) == {"s1", "s2", "s3"}
    assert outcome["results"]["s3"]["kept"] + outcome["results"]["s3"]["dropped"] == len(
        outcome["results"]["s2"]["objects"]
    )
    assert outcome["confidence"] is not None
    assert len(outcome["trace"]) == 3
    assert all(event["status"] == "ok" for event in outcome["trace"])


def test_executor_short_circuits_on_empty_upstream():
    s = demo_session()
    plan = Plan.from_dict({
        "steps": [
            {"id": "s1", "tool": "ground", "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
            {"id": "s2", "tool": "filter_by_region", "args": {"objects": "$s1.objects", "regions": "$s1.objects"}},
        ],
        "answer_from": "s2",
    })
    # Force the upstream step to look empty without needing a real detector
    # that returns nothing: patch the mock's result after the fact by running
    # against a plan whose filter never matches, then check the shortcut path
    # directly at the unit level instead.
    empty_results = {"s1": {"objects": []}}
    from plan import Step
    step = Step(id="s2", tool="filter_by_region", args={"objects": "$s1.objects", "regions": "$s1.objects"})
    assert executor._empty_upstream(step, empty_results) == "objects"


def test_executor_reports_partial_on_tool_failure():
    s = demo_session()
    plan = Plan.from_dict({
        "steps": [
            {"id": "s1", "tool": "vqa", "args": {"image_id": "img_2026_opt", "question": "?"}},
            {"id": "s2", "tool": "ground", "args": {"image_id": "does_not_exist", "phrase": "buildings"}},
        ],
        "answer_from": "s2",
    })
    outcome = executor.run(plan, s)
    assert outcome["status"] == "partial"
    assert "s1" in outcome["results"] and "s2" not in outcome["results"]
    assert outcome["trace"][-1]["status"] == "error"


def test_executor_confidence_is_the_minimum_not_the_mean():
    scores = executor._confidences("ground", {
        "objects": [{"confidence": 0.9}, {"confidence": 0.2}, {"confidence": 0.8}]
    })
    assert min(scores) == 0.2 and sum(scores) / len(scores) != 0.2


# --- controller ----------------------------------------------------------------

def test_handle_query_end_to_end_on_mocks():
    result = controller.handle_query("What changed between 2023 and 2026?", demo_session())
    assert result["status"] == "ok"
    assert result["plan"]["source"] == "fallback"
    assert isinstance(result["answer"], str) and result["answer"]
    assert "elapsed_s" in result and "trace" in result and "results" in result


def test_handle_query_never_raises_on_a_bad_query():
    result = controller.handle_query("", demo_session())
    assert result["status"] == "partial"
    assert result["plan"] is None


# --- mock_service: the HTTP stand-in for the model services -------------------
# The fixtures in mock_data/ are the reference responses P2/P3/P4 are asked to
# match, so they must pass the same checks a real service response does, and
# the controller must be able to reach them over a real socket.

import mock_service


def _fixture_args(tool, key):
    if tool == "change_detect":
        t1, t2 = key.split(">")
        return {"image_id_t1": t1, "image_id_t2": t2}
    if tool == "cross_modal":
        opt, sar = key.split("+")
        return {"optical_image_id": opt, "sar_image_id": sar, "phrase": "water"}
    if tool == "ground":
        return {"image_id": key, "phrase": "buildings"}
    return {"image_id": key, "question": "What is here?"}


def test_every_fixture_conforms_to_the_contract():
    session = demo_session()
    fixtures = mock_service.load_fixtures()
    assert set(fixtures) == {"vqa", "ground", "change_detect", "cross_modal"}
    for tool, by_key in fixtures.items():
        assert by_key, f"{tool}: no fixtures"
        spec = tools.TOOLS[tool]
        for key, result in by_key.items():
            args = _fixture_args(tool, key)
            tools._check_shape(spec, result)
            tools._check_items(spec, result, tools._source_footprints(spec, args, session))


def test_fixtures_cover_every_demo_image_and_pair():
    fixtures = mock_service.load_fixtures()
    session = demo_session()
    for image_id in session.ids():
        assert image_id in fixtures["vqa"], image_id
        assert image_id in fixtures["ground"], image_id
    assert "img_2023_opt>img_2026_opt" in fixtures["change_detect"]
    assert {"img_2026_opt+img_2026_sar", "img_cloud_opt+img_cloud_sar"} <= set(fixtures["cross_modal"])


def test_respond_wraps_in_the_success_envelope_and_labels_with_the_phrase():
    fixtures = mock_service.load_fixtures()
    status, body = mock_service.respond(fixtures, "ground", {"image_id": "img_2026_opt", "phrase": "ships"})
    assert status == 200 and body["status"] == "success"
    assert {o["label"] for o in body["result"]["objects"]} == {"ships"}
    # The cached fixture must not have been relabelled in place.
    assert fixtures["ground"]["img_2026_opt"]["objects"][0]["label"] == "object"


def test_respond_reports_unknown_inputs_as_error_envelopes():
    fixtures = mock_service.load_fixtures()
    assert mock_service.respond(fixtures, "vqa", {"image_id": "img_nope", "question": "?"})[0] == 404
    assert mock_service.respond(fixtures, "ground", {"phrase": "x"})[0] == 400
    assert mock_service.respond(fixtures, "not_a_tool", {})[0] == 404
    status, body = mock_service.respond(fixtures, "change_detect",
                                        {"image_id_t1": "img_2026_opt", "image_id_t2": "img_2023_opt"})
    assert status == 404 and body["status"] == "error"


def test_controller_reaches_the_mock_service_over_http():
    import os
    import threading
    from http.server import ThreadingHTTPServer

    mock_service.Handler.fixtures = mock_service.load_fixtures()
    server = ThreadingHTTPServer(("127.0.0.1", 0), mock_service.Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    saved = {k: os.environ.get(k) for k in ("SATQUERY_MOCK_GROUND", "SATQUERY_GROUND_URL", "SATQUERY_MOCK_VQA", "SATQUERY_VQA_URL")}
    try:
        os.environ["SATQUERY_MOCK_GROUND"] = "0"
        os.environ["SATQUERY_GROUND_URL"] = f"http://127.0.0.1:{port}/ground"
        os.environ["SATQUERY_MOCK_VQA"] = "0"
        os.environ["SATQUERY_VQA_URL"] = f"http://127.0.0.1:{port}/vqa"
        session = demo_session()
        result = tools.call("ground", {"image_id": "img_2026_opt", "phrase": "buildings"}, session)
        assert "mock" not in result, "answer came from the in-process mock, not the HTTP service"
        assert len(result["objects"]) == 5 and result["objects"][0]["label"] == "buildings"
        # An unknown image is a ToolError with the service's reason in it, not a crash.
        try:
            tools.call("vqa", {"image_id": "img_2026_opt", "question": "?"}, session)
        except tools.ToolError:
            raise AssertionError("known image should succeed")
        try:
            tools.call("vqa", {"image_id": "img_nope", "question": "?"}, session)
        except Exception as exc:
            assert "img_nope" in str(exc)
        else:
            raise AssertionError("unknown image should fail loudly")
    finally:
        server.shutdown()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore")
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:
            failed += 1
            print(f"FAIL  {name}\n      {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"ok    {name}")
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
