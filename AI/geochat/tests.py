"""Offline tests for the GeoChat agent. Run: python tests.py (from AI/geochat).

The model is never loaded here. What is under test is everything around it:
parsing GeoChat's answer format, turning grid boxes into lon/lat polygons
that land inside the image footprint, the contract envelope, and the
controller reaching the service over a real socket. The controller's own
validators are imported so "conforms" means the same thing on both sides.
"""

from __future__ import annotations

import os
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONTROLLER = HERE.parents[1] / "controller"
sys.path.insert(0, str(CONTROLLER))

import service  # noqa: E402
import tools  # noqa: E402  (controller)
from session import manifest_session  # noqa: E402  (controller)

STORE = service.ImageStore(CONTROLLER / "images")
STUB = service.StubBackend()
SESSION = manifest_session(CONTROLLER / "images" / "manifest.json")


# --- parsing -------------------------------------------------------------------

def test_parse_labelled_boxes_with_and_without_angle():
    text = "<p>buildings</p> {<10><12><30><28>|<0>} and <p>ships</p> {<55><50><72><66>}"
    boxes = service.parse_boxes(text)
    assert [b["label"] for b in boxes] == ["buildings", "ships"]
    assert boxes[0]["angle"] == 0 and boxes[1]["angle"] == 0
    assert (boxes[1]["x0"], boxes[1]["y0"], boxes[1]["x1"], boxes[1]["y1"]) == (55, 50, 72, 66)


def test_parse_normalises_reversed_and_out_of_range_corners():
    boxes = service.parse_boxes("{<90><80><10><20>|<-30>} {<-5><0><120><40>}")
    assert (boxes[0]["x0"], boxes[0]["x1"], boxes[0]["y0"], boxes[0]["y1"]) == (10, 90, 20, 80)
    assert boxes[0]["angle"] == -30
    assert (boxes[1]["x0"], boxes[1]["x1"]) == (0, 100)


def test_parse_drops_degenerate_boxes_and_ignores_prose():
    assert service.parse_boxes("There are no ships here.") == []
    assert service.parse_boxes("{<10><10><10><40>}") == []  # zero width
    assert service.parse_boxes("{<1><2>}") == []  # too few numbers


# --- geometry ------------------------------------------------------------------

def test_rotated_corners_stay_inside_the_image():
    box = {"x0": 80, "y0": 80, "x1": 100, "y1": 100, "angle": 45}
    for x, y in service.box_to_pixel_corners(box, 530, 565):
        assert 0 <= x <= 530 and 0 <= y <= 565


def test_unrotated_box_maps_to_the_expected_pixel_corners():
    corners = service.box_to_pixel_corners({"x0": 0, "y0": 0, "x1": 50, "y1": 100, "angle": 0}, 200, 400)
    assert corners == [(0.0, 0.0), (100.0, 0.0), (100.0, 400.0), (0.0, 400.0)]


def test_polygons_are_epsg4326_and_inside_the_footprint_for_every_image():
    """The controller's own footprint check, run on every chip in the
    manifest, with a rotated box included: a wrong CRS or a swapped axis in
    pixels_to_lonlat fails here, not in a demo."""
    spec = tools.TOOLS["ground"]
    for image_id in STORE.manifest:
        _, body = service.respond(STORE, STUB, "ground", {"image_id": image_id, "phrase": "buildings"})
        result = body["result"]
        tools._check_shape(spec, result)
        footprints = tools._source_footprints(spec, {"image_id": image_id, "phrase": "buildings"}, SESSION)
        tools._check_items(spec, result, footprints)
        assert len(result["objects"]) == 2
        ring = result["objects"][1]["geometry"]["coordinates"][0]
        assert ring[0] == ring[-1] and len(ring) == 5
        assert all(-180 <= lon <= 180 and -90 <= lat <= 90 for lon, lat in ring)


# --- contract ------------------------------------------------------------------

def test_vqa_envelope():
    status, body = service.respond(STORE, STUB, "vqa", {"image_id": "img_2026_opt", "question": "What is here?"})
    assert status == 200 and body["status"] == "success"
    tools._check_shape(tools.TOOLS["vqa"], body["result"])
    assert 0 <= body["result"]["confidence"] <= 1


def test_ground_labels_come_from_the_answer_or_the_phrase():
    _, body = service.respond(STORE, STUB, "ground", {"image_id": "img_2026_opt", "phrase": "ships"})
    assert {o["label"] for o in body["result"]["objects"]} == {"ships"}
    assert "raw" in body["result"]  # the model's text, kept for debugging; extra fields are allowed


def test_errors_are_envelopes_not_tracebacks():
    assert service.respond(STORE, STUB, "vqa", {"image_id": "img_nope", "question": "?"})[0] == 404
    assert service.respond(STORE, STUB, "ground", {"phrase": "x"})[0] == 400
    assert service.respond(STORE, STUB, "change_detect", {})[0] == 404
    assert service.respond(STORE, STUB, "vqa", "not an object")[0] == 400

    class Boom:
        name = "boom"

        def infer(self, image, prompt):
            raise RuntimeError("cuda out of memory")

    status, body = service.respond(STORE, Boom(), "vqa", {"image_id": "img_2026_opt", "question": "?"})
    assert status == 500 and "cuda out of memory" in body["error"]


# --- the controller talking to it -------------------------------------------------

def test_controller_calls_the_agent_over_http():
    service.Handler.store, service.Handler.backend = STORE, STUB
    server = ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    keys = ("SATQUERY_MOCK_VQA", "SATQUERY_VQA_URL", "SATQUERY_MOCK_GROUND", "SATQUERY_GROUND_URL")
    saved = {k: os.environ.get(k) for k in keys}
    try:
        os.environ.update({"SATQUERY_MOCK_VQA": "0", "SATQUERY_VQA_URL": f"http://127.0.0.1:{port}/vqa",
                           "SATQUERY_MOCK_GROUND": "0", "SATQUERY_GROUND_URL": f"http://127.0.0.1:{port}/ground"})
        found = tools.call("ground", {"image_id": "jnpt_port_2026_opt", "phrase": "ships"}, SESSION)
        assert "mock" not in found and len(found["objects"]) == 2
        answer = tools.call("vqa", {"image_id": "udaipur_2026_opt", "question": "Describe the scene."}, SESSION)
        assert "mock" not in answer and answer["answer"].startswith("The image shows")
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
