# Agent controller

The layer between the backend and the model services. A planner turns an
English query into a validated JSON plan; a deterministic executor runs it over
HTTP against the services and two local geometry tools. The controller never
imports a model, and the planner never produces a coordinate or a count.

## Running

```
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # or .venv/bin/pip
.venv/Scripts/python tests.py
```

Mocks are on by default (`SATQUERY_MOCK=1`), so everything runs with no
services and no network. Every mock result carries `"mock": true`. To point a
tool at a real service:

```
SATQUERY_MOCK_GROUND=0
SATQUERY_GROUND_URL=https://<tunnel>/ground
```

One tool at a time, so integration problems land one at a time.

## Mock service

`mock_service.py` is the HTTP stand-in for the model services. It serves the
fixtures in `mock_data/` on the tools' default ports (8001-8004) in the team
envelope, so the controller exercises its real HTTP adapter, envelope handling
and response validation instead of the in-process mocks:

```
.venv/Scripts/python mock_service.py           # terminal 1
SATQUERY_MOCK=0 .venv/Scripts/python e2e.py    # terminal 2
```

The fixtures are the reference responses for P2/P3/P4: one file per tool,
keyed by the demo image id(s), every geometry a GeoJSON Polygon in EPSG:4326
that lies inside the source footprint. `tests.py` checks every fixture against
the same validators a real response goes through, so a service whose output
matches a fixture field-for-field will pass integration. An unknown image id
gets a 404 with `{"status": "error", "error": ...}`, which is what a real
service should return for an image it has not loaded.

## Service contract

This is what the controller sends and what it checks on the way back. It is
the contract in the implementation doc, not the pixel-bbox one in
`backend/README.md`, which describes the earlier in-process mocks. The two
differ in path, argument names and geometry format; this one is binding.

Every service is `POST /<tool>` with a JSON body, returning either a bare
result object or `{"status": "success", "result": {...}}`.

| tool | request | result |
|---|---|---|
| `vqa` | `{image_id, question}` | `{answer, confidence}` |
| `ground` | `{image_id, phrase}` | `{objects: [{label, geometry, confidence}]}` |
| `change_detect` | `{image_id_t1, image_id_t2}` | `{regions: [{type, geometry, confidence}], changed_area_km2, change_mask_id}` |
| `cross_modal` | `{optical_image_id, sar_image_id, phrase}` | `{regions: [{label, geometry, confidence}], summary}` |

Rules the controller enforces on every response, mock or real:

- every declared field present; list fields are lists
- every `geometry` is a GeoJSON `Polygon` or `MultiPolygon` in **EPSG:4326
  (lon, lat)**; anything else is rejected before it reaches the next step
- every item's geometry lies mostly inside the footprint of the image it was
  computed from — this catches a swapped lon/lat, a missing reprojection, and
  a result from the wrong tile
- every item has a numeric `confidence` in `[0, 1]`

BigEarthNet patches are stored in UTM. Reproject inside the adapter with
`rasterio.warp.transform_geom(src.crs, "EPSG:4326", geom)`. The footprint
check is there for the day someone forgets.

Paste a sample response into `tools._check_shape` and `tools._check_items` to
find out whether it conforms before writing any model code.

## Plan format

```json
{
  "steps": [
    {"id": "s1", "tool": "change_detect",
     "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
    {"id": "s2", "tool": "ground",
     "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
    {"id": "s3", "tool": "filter_by_region",
     "args": {"objects": "$s2.objects", "regions": "$s1.regions"}}
  ],
  "answer_from": "s3",
  "reasoning": "New buildings are buildings in the later image inside changed regions."
}
```

`$<step_id>.<field>` is the only way data moves between steps. `plan.validate`
rejects, with a message the model can act on: unknown tools, missing or extra
arguments, references forwards or to fields a tool does not return, references
of the wrong kind, image ids that are not loaded, literal geometry or counts,
two images of different places, a change comparison that runs backwards in
time, and an optical/SAR pair the wrong way round.

## Layout

| file | what |
|---|---|
| `geometry.py` | EPSG:4326 validation and repair, geodesic area, overlap |
| `session.py` | loaded images: id, sensor, date, footprint, cloud flag, pairing |
| `tools.py` | tool registry, HTTP and mock adapters, `filter_by_region`, `count` |
| `plan.py` | plan format and validator |
| `planner.py` | LLM query -> validated `Plan`, with retry and keyword fallback |
| `executor.py` | runs a `Plan`: `$ref` resolution, short-circuiting, partial results, confidence |
| `controller.py` | `handle_query()`, the single public entry point |
| `mock_service.py` | HTTP stand-in for the model services, serves `mock_data/` fixtures |
| `e2e.py` | end-to-end runner: real planner, mock or real tools |
| `tests.py` | run with `python tests.py` or pytest |

## Running a query end to end

```python
from session import demo_session
from controller import handle_query

result = handle_query("Find buildings constructed after 2023", demo_session())
print(result["answer"])
```

## Planner providers

The planner tries a chain of OpenAI-compatible endpoints in order and only
then falls back to keyword routing:

| position | variables | default | enabled when |
|---|---|---|---|
| primary | `SATQUERY_LLM_BASE_URL`, `SATQUERY_LLM_MODEL`, `SATQUERY_LLM_API_KEY` | Ollama, `http://localhost:11434/v1`, `gpt-oss:20b`, no key | any of the three is set |
| fallback | `SATQUERY_LLM_FALLBACK_BASE_URL`, `SATQUERY_LLM_FALLBACK_MODEL`, `SATQUERY_LLM_FALLBACK_API_KEY` | Gemini's OpenAI-compatible endpoint, `gemini-3.6-flash` | any of the three is set (the key, in practice) |

The demo setup is Ollama on a teammate's machine as primary and Gemini as the
safety net, which is this `.env`:

```
SATQUERY_LLM_BASE_URL=http://<teammate-ip>:11434/v1
SATQUERY_LLM_FALLBACK_API_KEY=<Google AI Studio key>
```

Each provider gets one retry with the validator's error appended, then the
chain moves on. If every provider fails, the keyword plan's `reasoning` ends
with `(every LLM provider failed: ...)` naming each provider and its error,
and `plan.provider` is `"keywords"`; an LLM plan carries the model name there.
With nothing set, the planner is keyword-only and fully offline, which is how
`tests.py` runs. Groq is `https://api.groq.com/openai/v1` with
`openai/gpt-oss-20b` in either slot. Gemini's free tier is about 20 requests
a day per model, which is why it is the fallback and not the primary.
