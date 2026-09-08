"""API tests for the change-detection service, via FastAPI's TestClient.

Run:
    .venv/Scripts/python.exe test_api.py

Covers a successful run, a size mismatch, a missing file, a cache hit and a
query-mode request. Written as a plain script so it needs no pytest; the
individual checks are ordinary functions and work under pytest too.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from main import app

SERVICE_ROOT = Path(__file__).resolve().parent
SAMPLES = SERVICE_ROOT / "vendor" / "BIT_CD" / "samples"
SAMPLE = "test_2_0000_0000.png"
BEFORE = str(SAMPLES / "A" / SAMPLE)
AFTER = str(SAMPLES / "B" / SAMPLE)

EXPECTED_WORKFLOW = [
    "validate_inputs",
    "change_detection",
    "mask_analysis",
    "change_description",
]

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"    ok   {label}")
    else:
        _failures.append(label)
        print(f"    FAIL {label}{(' -- ' + detail) if detail else ''}")


def test_health(client: TestClient) -> None:
    print("\n[health]")
    response = client.get("/health")
    body = response.json()
    check("200", response.status_code == 200)
    check("model warmed at startup", body["model_loaded"] is True, str(body))
    check("device reported", body["device"] is not None, str(body))
    print(f"    device={body['device']} cuda={body['cuda_available']} model={body['model']}")


def test_success(client: TestClient) -> dict:
    print("\n[successful run]")
    response = client.post(
        "/change-detection",
        json={
            "before_path": BEFORE,
            "after_path": AFTER,
            "before_date": "2017-03-11",
            "after_date": "2019-08-24",
        },
    )
    body = response.json()

    check("200", response.status_code == 200, response.text[:200])
    check("status success", body["status"] == "success")
    check("query_id present", bool(body.get("query_id")))
    check("workflow matches contract", body["analysis"]["workflow"] == EXPECTED_WORKFLOW)
    check("intent is change_detection", body["analysis"]["intent"] == "change_detection")

    result = body["result"]
    check("description non-empty", bool(result["description"].strip()))
    check("region_count matches regions", result["region_count"] == len(result["regions"]))
    check("change_percentage is a number", isinstance(result["change_percentage"], (int, float)))
    check("georeferenced false for PNG", result["georeferenced"] is False)
    check("dates appear in description", "2017-03-11" in result["description"])

    steps = body["execution"]["steps"]
    check("validate_inputs is step 1", steps[0]["action"] == "validate_inputs")
    check("validate_inputs completed", steps[0]["status"] == "completed")
    check(
        "every step has action/status/duration_ms",
        all({"action", "status", "duration_ms"} <= set(step) for step in steps),
    )
    check("all steps completed", all(step["status"] == "completed" for step in steps))

    detection_step = next(s for s in steps if s["action"] == "change_detection")
    check("detection step carries model", detection_step.get("model") == "BIT-LEVIR-CD")
    check("detection step carries parameters", "parameters" in detection_step)

    details = steps[0]["details"]
    check(
        "validation covers the five mandated checks",
        {"count", "modality", "format", "metadata", "compatibility"} <= set(details),
        str(sorted(details)),
    )
    check(
        "georeferencing recorded per image",
        {"before_georeferenced", "after_georeferenced"} <= set(details["metadata"]),
    )
    check(
        "CRS and transform compatibility recorded",
        {"crs_match", "transform_match"} <= set(details["compatibility"]),
    )

    execution = body["execution"]
    check("model reported", execution["model"] == "BIT-LEVIR-CD")
    check(
        "parameters reported",
        execution["parameters"] == {"threshold": 0.5, "min_region_px": 50, "connectivity": 2},
        str(execution["parameters"]),
    )
    check("total_ms positive", execution["total_ms"] > 0)
    check("not served from cache", execution["cached"] is False)

    statistics = body["statistics"]
    check("statistics agree with result", statistics["regions_detected"] == result["region_count"])

    visualization = body["visualization"]
    check("type is temporal_comparison", visualization["type"] == "temporal_comparison")

    # The images must exist on disk and be served by the static mount.
    for key, name in (("change_mask_url", "change_mask.png"), ("overlay_url", "overlay.png")):
        url = visualization[key]
        check(f"{name} url shaped correctly", url.endswith(f"/{name}"), url)
        served = client.get(url)
        check(f"{name} served over /results", served.status_code == 200)
        check(f"{name} is a PNG", served.content[:8] == b"\x89PNG\r\n\x1a\n")

        on_disk = SERVICE_ROOT / "results" / body["query_id"] / name
        check(f"{name} written to disk", on_disk.is_file(), str(on_disk))

    with Image.open(SERVICE_ROOT / "results" / body["query_id"] / "change_mask.png") as mask:
        check("mask is single-band", mask.mode == "L", mask.mode)
        check("mask matches image size", mask.size == (256, 256), str(mask.size))
    with Image.open(SERVICE_ROOT / "results" / body["query_id"] / "overlay.png") as overlay:
        check("overlay is RGB", overlay.mode == "RGB", overlay.mode)

    print(
        f"    {result['change_percentage']}% changed, {result['region_count']} regions, "
        f"{execution['total_ms']} ms"
    )
    return body


def test_cache_hit(client: TestClient, first: dict) -> None:
    print("\n[cache hit]")
    payload = {
        "before_path": BEFORE,
        "after_path": AFTER,
        "before_date": "2017-03-11",
        "after_date": "2019-08-24",
    }
    response = client.post("/change-detection", json=payload)
    body = response.json()

    check("200", response.status_code == 200)
    check("served from cache", body["execution"]["cached"] is True)
    check("same query_id as first call", body["query_id"] == first["query_id"])
    check("identical result payload", body["result"] == first["result"])
    check(
        "cache hit is faster than the live run",
        body["execution"]["total_ms"] <= first["execution"]["total_ms"],
        f"{body['execution']['total_ms']} vs {first['execution']['total_ms']}",
    )

    # A different parameter must miss the cache and produce a fresh run.
    other = client.post("/change-detection", json={**payload, "threshold": 0.8}).json()
    check("different threshold misses the cache", other["execution"]["cached"] is False)
    check("different threshold gets a new query_id", other["query_id"] != first["query_id"])
    check(
        "higher threshold flags no more change",
        other["result"]["change_percentage"] <= first["result"]["change_percentage"],
        f"{other['result']['change_percentage']} vs {first['result']['change_percentage']}",
    )


def test_size_mismatch(client: TestClient, tmp_dir: Path) -> None:
    print("\n[size mismatch]")
    resized = tmp_dir / "resized.png"
    with Image.open(AFTER) as handle:
        handle.resize((128, 128)).save(resized)

    response = client.post(
        "/change-detection", json={"before_path": BEFORE, "after_path": str(resized)}
    )
    body = response.json()

    check("400", response.status_code == 400, str(response.status_code))
    check("status error", body["status"] == "error")
    check("code is SIZE_MISMATCH", body["error"]["code"] == "SIZE_MISMATCH", str(body["error"]))
    check("message names both sizes", "256x256" in body["error"]["message"], body["error"]["message"])

    steps = body["execution"]["steps"]
    check("partial trace returned", len(steps) == 1)
    check("validation marked failed", steps[0]["status"] == "failed")
    check("failed step still timed", steps[0]["duration_ms"] >= 0)
    check("no result key on error", "result" not in body)


def test_missing_file(client: TestClient) -> None:
    print("\n[missing file]")
    response = client.post(
        "/change-detection",
        json={"before_path": str(SAMPLES / "A" / "does_not_exist.png"), "after_path": AFTER},
    )
    body = response.json()

    check("404", response.status_code == 404, str(response.status_code))
    check("code is IMAGE_NOT_FOUND", body["error"]["code"] == "IMAGE_NOT_FOUND", str(body["error"]))
    check("validation marked failed", body["execution"]["steps"][0]["status"] == "failed")
    check("query_id present on error", bool(body.get("query_id")))

    print("\n[missing second image]")
    response = client.post("/change-detection", json={"before_path": BEFORE})
    body = response.json()
    check("400", response.status_code == 400, str(response.status_code))
    check(
        "code is MISSING_SECOND_IMAGE",
        body["error"]["code"] == "MISSING_SECOND_IMAGE",
        str(body["error"]),
    )

    print("\n[invalid image]")
    with tempfile.NamedTemporaryFile("w", suffix=".png", delete=False) as handle:
        handle.write("this is not a PNG")
        junk = handle.name
    response = client.post(
        "/change-detection", json={"before_path": junk, "after_path": AFTER}
    )
    body = response.json()
    check("400", response.status_code == 400, str(response.status_code))
    check("code is INVALID_IMAGE", body["error"]["code"] == "INVALID_IMAGE", str(body["error"]))
    Path(junk).unlink(missing_ok=True)


def test_query_mode(client: TestClient) -> None:
    print("\n[query mode]")
    response = client.post(
        "/change-detection",
        json={"before_path": BEFORE, "after_path": AFTER, "query": "Where are the changes?"},
    )
    body = response.json()

    check("200", response.status_code == 200)
    result = body["result"]
    check("query echoed", result["query"] == "Where are the changes?")
    check("intent routed to where", result["intent"] == "where", str(result.get("intent")))
    check("marked supported", result["supported"] is True)
    check("answer mentions a compass position", "north" in result["description"].lower()
          or "centre" in result["description"].lower(), result["description"][:120])

    step = next(s for s in body["execution"]["steps"] if s["action"] == "change_description")
    check("description step records query mode", step["mode"] == "query")
    check("description step records intent", step["intent"] == "where")

    print("\n[query mode, unsupported question]")
    body = client.post(
        "/change-detection",
        json={
            "before_path": BEFORE,
            "after_path": AFTER,
            "query": "What is the average rainfall here?",
        },
    ).json()
    check("still a successful run", body["status"] == "success")
    check("marked unsupported", body["result"]["supported"] is False)
    check("intent is unknown", body["result"]["intent"] == "unknown")

    print("\n[direction question invents no direction]")
    body = client.post(
        "/change-detection",
        json={
            "before_path": BEFORE,
            "after_path": AFTER,
            "query": "Has the built-up area increased or decreased?",
        },
    ).json()
    answer = body["result"]["description"]
    check("intent is direction", body["result"]["intent"] == "direction")
    check("states the limitation", "not what it changed from or to" in answer, answer[:120])


def main() -> int:
    for path in (Path(BEFORE), Path(AFTER)):
        if not path.is_file():
            print(f"sample not found: {path}", file=sys.stderr)
            return 1

    # The TestClient context manager is what triggers the lifespan handler, so
    # the startup warm-up runs exactly as it does under uvicorn.
    with TestClient(app) as client, tempfile.TemporaryDirectory() as tmp:
        test_health(client)
        first = test_success(client)
        test_cache_hit(client, first)
        test_size_mismatch(client, Path(tmp))
        test_missing_file(client)
        test_query_mode(client)

    print()
    if _failures:
        print(f"{len(_failures)} check(s) FAILED:")
        for name in _failures:
            print(f"  - {name}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
