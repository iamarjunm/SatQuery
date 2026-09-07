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

from PIL import Image  # noqa: E402

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


# --- gemini backend, offline ---------------------------------------------------------
# The network call is replaced by a fake client. Under test: Gemini's JSON boxes
# becoming GeoChat-format text, rotation to the next model on a quota error,
# and the answer cache.

class _FakeCompletions:
    """Scripted chat.completions.create: per model, a string to return or an
    exception to raise. Records which models were called."""

    def __init__(self, script: dict):
        self.script, self.calls = script, []

    def create(self, *, model, messages, **kwargs):
        self.calls.append(model)
        outcome = self.script[model]
        if isinstance(outcome, Exception):
            raise outcome

        class _Msg:
            content = outcome

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        return _Resp()


def _gemini(script: dict, models: tuple) -> service.GeminiBackend:
    backend = service.GeminiBackend(api_key="test-key", models=models)
    backend.client = type("C", (), {})()
    backend.client.chat = type("Chat", (), {})()
    backend.client.chat.completions = _FakeCompletions(script)
    return backend


def test_gemini_boxes_become_geochat_text_on_the_100_grid():
    raw = '{"objects": [{"label": "ship", "box_2d": [100, 200, 150, 260]}, {"label": "ship", "box_2d": [500, 500, 600, 700]}]}'
    backend = _gemini({"m1": raw}, ("m1",))
    text = backend.infer(Image.new("RGB", (100, 100)), service.REFER_PROMPT.format(phrase="ships"))
    boxes = service.parse_boxes(text)
    assert [b["label"] for b in boxes] == ["ship", "ship"]
    # box_2d is [ymin, xmin, ymax, xmax] on 0-1000 -> x0,y0,x1,y1 on 0-100
    assert (boxes[0]["x0"], boxes[0]["y0"], boxes[0]["x1"], boxes[0]["y1"]) == (20.0, 10.0, 26.0, 15.0)
    assert backend.last_model == "m1"


def test_gemini_empty_or_malformed_json_means_no_objects():
    assert service.GeminiBackend._boxes_from_json('{"objects": []}') == []
    assert service.GeminiBackend._boxes_from_json("no json here") == []
    assert service.GeminiBackend._boxes_from_json('{"objects": [{"label": "x", "box_2d": [1, 2]}]}') == []
    backend = _gemini({"m1": "Nothing found."}, ("m1",))
    assert service.parse_boxes(backend.infer(Image.new("RGB", (10, 10)), "[refer] Give me the location of <p> cars </p>")) == []


def test_gemini_rotates_to_the_next_model_on_quota_and_remembers_it():
    quota = RuntimeError("Error code: 429 - quota exceeded for generate_content_free_tier_requests")
    backend = _gemini({"m1": quota, "m2": "A river through farmland."}, ("m1", "m2"))
    img = Image.new("RGB", (10, 10))
    assert backend.infer(img, "What is here?") == "A river through farmland."
    assert backend.client.chat.completions.calls == ["m1", "m2"]
    assert backend.last_model == "m2"
    # m1 is on cooldown now: the second call goes straight to m2.
    backend.infer(img, "And now?")
    assert backend.client.chat.completions.calls == ["m1", "m2", "m2"]


def test_gemini_every_model_dry_is_a_clear_error_not_a_crash():
    quota = RuntimeError("429 RESOURCE_EXHAUSTED")
    backend = _gemini({"m1": quota, "m2": quota}, ("m1", "m2"))
    status, body = service.respond(STORE, backend, "vqa", {"image_id": "img_2026_opt", "question": "?"})
    assert status == 500 and "rate-limited or unavailable" in body["error"]
    assert "m1" in body["error"] and "m2" in body["error"]


def test_gemini_non_quota_errors_are_not_swallowed_by_rotation():
    backend = _gemini({"m1": RuntimeError("Error code: 400 - bad image"), "m2": "unused"}, ("m1", "m2"))
    try:
        backend.infer(Image.new("RGB", (10, 10)), "?")
    except RuntimeError as exc:
        assert "bad image" in str(exc)
    else:
        raise AssertionError("a 400 must surface, not fall through to the next model")
    assert backend.client.chat.completions.calls == ["m1"]


def test_cache_serves_the_second_identical_call_without_the_model(tmp_dir=None):
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        cache = service.ResultCache(tmp)
        backend = _gemini({"m1": "Built-up area beside a river."}, ("m1",))
        args = {"image_id": "udaipur_2026_opt", "question": "Describe it."}
        s1, b1 = service.respond(STORE, backend, "vqa", args, cache)
        s2, b2 = service.respond(STORE, backend, "vqa", args, cache)
        assert s1 == s2 == 200 and b1["result"] == b2["result"]
        assert b2.get("cached") is True and "cached" not in b1
        assert backend.client.chat.completions.calls == ["m1"]  # one model call for two requests
        # Different question, different key.
        service.respond(STORE, backend, "vqa", {**args, "question": "Anything else?"}, cache)
        assert backend.client.chat.completions.calls == ["m1", "m1"]
        # Extra request fields do not change the key; only the tool's own arguments do.
        _, b4 = service.respond(STORE, backend, "vqa", {**args, "extra": 1}, cache)
        assert b4.get("cached") is True
        assert len(list(tmp.glob("vqa-udaipur_2026_opt-*.json"))) == 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


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
