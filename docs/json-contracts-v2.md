# SatQuery — Prototype JSON Contracts v2

This supersedes the v1 contract document ("SatQuery — Updated Prototype JSON
Contracts", the shared Google Doc). v1 was written before the controller
existed. The controller is now on `main` and enforces the shapes below on every
response it receives, so this document describes what the code actually
accepts and returns. Where v2 differs from v1 the difference is listed in
section 0 with the reason.

The pieces are:

```
frontend ── POST /query ──▶ backend (Node) ── handle_query() ──▶ controller (Python)
                                                                    │  HTTP
                                             vqa · ground · change_detect · cross_modal
```

The controller never imports a model. It speaks HTTP to every service, and it
never produces a coordinate or a count itself — every number in a response
comes from a service or from a geometric computation on service output.

---

## 0. What changed from v1

| area | v1 | v2 | why |
|---|---|---|---|
| geometry | `bbox: [x_min, y_min, x_max, y_max]` in pixels | `geometry`: GeoJSON `Polygon` / `MultiPolygon` in **EPSG:4326 (lon, lat)** | Grounding tiles at 512 px, change detection resamples; the two pixel grids do not line up, so pixel boxes from different models cannot be compared. Lat/lon can. BigEarthNet patches are georeferenced GeoTIFFs, so this is one `rasterio.warp.transform_geom` call in the adapter. |
| change detection args | `image_before`, `image_after` | `image_id_t1`, `image_id_t2` | Matches the implementation doc; `t1` is always the earlier image and the validator checks this against the scene dates. |
| grounding arg | `query` | `phrase` | Avoids clashing with the user's query string, which is also called `query`. |
| cross-modal args | `optical_image: {image_id, source}`, `sar_image: {…}`, `query` | `optical_image_id`, `sar_image_id`, `phrase` | Flat ids; the controller already knows each image's source. |
| change detection result | `changes: [{id, bbox, confidence}]`, `change_count` | `regions: [{type, geometry, confidence}]`, `changed_area_km2`, `change_mask_id` | Area is a real geodesic measurement, not a count of boxes. `type` is `construction` / `vegetation_loss` / `removal`. |
| change mask | `mask_url` in the result | `change_mask_id` (opaque string) | The controller does not fetch masks. The service keeps the mask and serves it to the frontend by id (see §3.3). |
| per-item fields | `confidence` optional in practice | `confidence` **required**, number in `[0, 1]`, on every object and region | Confidence for a chain is the minimum across steps; a missing value would silently inflate it. |
| envelope | `{task, status, result}` | `{status, result}` or a bare result object — both accepted; `task` and `type` inside `result` are ignored | Less to get wrong. `status` must be `"success"` or `"ok"` if present. |
| router output | `{intent, input_type, workflow: [{step, task, input \| input_from}]}` | plan: `{steps: [{id, tool, args}], answer_from, reasoning}` with `$step.field` references | `input_from: "step_1"` names one source step and no field. The headline query (`filter_by_region` taking objects from one step and regions from another) cannot be written in it. |
| final response | `{query_id, status, analysis, result, visualization, statistics}` | `{query, plan, status, answer, confidence, confidence_note, results, trace, elapsed_s}` (§6) | The controller returns what it computed and how. Deriving `visualization` and `statistics` for the frontend is the backend's job (§9), since it depends on how P6 renders. |
| error | `{status: "error", error: {code, message}}` | `status: "partial"` plus a human-readable `answer` and `confidence_note`; the controller never raises | A dead model in step 3 still returns steps 1 and 2. The frontend always gets valid JSON with a sentence in it. |
| execution log | `{execution: {steps: [{step, action, status}]}}` | `trace: [{step, tool, status, detail?}]` | One entry per plan step, tied to the plan's step ids. |
| image metadata | `{image_id, source, modality, date, resolution_m}` | adds `bounds`, `cloudy`, `snow`, `pair_id`, `labels`; `date` → `acquired` | The planner needs footprints to know which scenes are the same place, and the cloud flag to route to SAR. |
| cross-modal | one of several tasks | a first-class tool **and** a planning rule: if the optical scene is flagged cloudy and a paired SAR scene is loaded, the planner routes there | Satisfies the multi-sensor requirement as a routing decision, which is also the demo moment. |

Not changed: the `/query` request from the frontend (§1) keeps v1's shape,
minus the `input.type` field, which the controller no longer needs.

---

## 1. Frontend → backend request

Unchanged from v1 except that `input.type` is gone. The frontend does not
choose the workflow; it says which images are on screen.

```json
{
  "query_id": "q_004",
  "query": "Find buildings constructed after 2023",
  "image_ids": ["img_2023_opt", "img_2026_opt", "img_2026_sar"]
}
```

The backend builds a `Session` from the referenced images (§2) and calls
`handle_query(query, session)`. The HTTP wrapper around `handle_query` is the
backend's; the controller's public API is the Python call.

---

## 2. Image metadata (`Scene`)

One record per loaded image. This is what the planner is shown and what the
validator checks image ids against.

```json
{
  "image_id": "img_2026_opt",
  "modality": "optical",
  "source": "sentinel-2",
  "acquired": "2026-06-15",
  "bounds": [16.3700, 48.2100, 16.3862, 48.2208],
  "resolution_m": 10,
  "cloudy": false,
  "snow": false,
  "pair_id": "img_2026_sar",
  "labels": ["Arable land", "Urban fabric"]
}
```

| field | type | notes |
|---|---|---|
| `image_id` | string | unique within the session |
| `modality` | `"optical"` or `"sar"` | required |
| `source` | string | e.g. `sentinel-2`, `sentinel-1` |
| `acquired` | ISO date | required. Decides which scene is earlier in a change comparison. Not defaulted. |
| `bounds` | `[west, south, east, north]` in EPSG:4326 | read off the GeoTIFF with `rasterio.warp.transform_bounds`. Scenes that straddle the antimeridian are refused. |
| `resolution_m` | number, optional | |
| `cloudy`, `snow` | boolean | from BigEarthNet's `contains_cloud_or_shadow` / `contains_seasonal_snow`. The recommended `metadata.parquet` excludes cloudy patches; pull from the snow/cloud parquet to demo SAR routing. |
| `pair_id` | string, optional | the co-registered scene from the other sensor. BigEarthNet's `s1_name` column gives this for free. |
| `labels` | list of strings | BigEarthNet land-cover labels; the VQA mock uses them |

Scenes whose footprints overlap by IoU ≥ 0.9 are grouped into an **area**
(`area A`, `area B`, …). The planner sees the area label next to each image;
the validator refuses any comparison between scenes in different areas.

---

## 3. Service contracts

Every service is `POST /<tool>` with a JSON body. The response is either the
result object directly, or wrapped:

```json
{ "status": "success", "result": { ... } }
```

Rules enforced on every response, mock or real, before it can reach the next
step:

1. every field listed under *result* below is present, and is a list / number /
   string as declared
2. every `geometry` is a GeoJSON `Polygon` or `MultiPolygon` in **EPSG:4326**,
   `[lon, lat]` order; anything outside ±180 / ±90 is rejected as a missed
   reprojection
3. every item's geometry lies at least 50 % inside the footprint of one of the
   images the call was given — this catches a swapped lon/lat, the wrong tile,
   and any other wrong-frame output
4. every item has `confidence`, a number in `[0, 1]`
5. invalid rings (self-intersections, holes poking past the shell) are repaired
   with `shapely.make_valid(method="structure")`; area is measured geodesically

A response that fails any rule fails the step with a message naming the item
and the rule. The remaining steps of the plan are not run; the answer says so.

BigEarthNet patches are stored in UTM. Reproject in the adapter:

```python
from rasterio.warp import transform_geom
geojson = transform_geom(src.crs, "EPSG:4326", pixel_polygon_in_src_crs)
```

Before writing model code, paste a sample response into
`controller/tools.py:_check_shape` and `_check_items` to see whether it
conforms.

### 3.1 `POST /vqa`

```json
{ "image_id": "img_2026_opt", "question": "What land cover is visible?" }
```
```json
{ "answer": "Agricultural land with a small built-up area in the north-east.",
  "confidence": 0.91 }
```

### 3.2 `POST /ground`

```json
{ "image_id": "img_2026_opt", "phrase": "buildings" }
```
```json
{ "objects": [
    { "id": "obj_001", "label": "building",
      "geometry": { "type": "Polygon",
                    "coordinates": [[[16.3716, 48.2197], [16.3729, 48.2197],
                                     [16.3729, 48.2188], [16.3716, 48.2188],
                                     [16.3716, 48.2197]]] },
      "confidence": 0.93 }
] }
```

`id` and `label` are passed through but not enforced. Apply NMS across tile
boundaries inside the adapter; duplicated detections inflate every count
downstream.

### 3.3 `POST /change_detect`

```json
{ "image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt" }
```
```json
{ "regions": [
    { "id": "change_001", "type": "construction",
      "geometry": { "type": "Polygon", "coordinates": [[ ... ]] },
      "confidence": 0.92 }
  ],
  "changed_area_km2": 0.107,
  "change_mask_id": "mask_2023_2026_a1b2" }
```

`type` is one of `construction`, `vegetation_loss`, `removal`. Polygonise the
mask into `regions`; adjacent fragments are fine, the controller merges them
before measuring coverage. `change_mask_id` is opaque to the controller. If the
frontend needs the raster, the service should serve it at a URL it can build
from the id, e.g. `GET /masks/<change_mask_id>.png`; agree that path with P6.

### 3.4 `POST /cross_modal`

```json
{ "optical_image_id": "img_cloud_opt", "sar_image_id": "img_cloud_sar",
  "phrase": "built-up and water" }
```
```json
{ "regions": [
    { "id": "region_001", "label": "built_up",
      "geometry": { "type": "Polygon", "coordinates": [[ ... ]] },
      "confidence": 0.91 },
    { "id": "region_002", "label": "water",
      "geometry": { "type": "Polygon", "coordinates": [[ ... ]] },
      "confidence": 0.94 }
  ],
  "summary": "The scene contains built-up regions and water-covered regions." }
```

The two images must be a co-registered pair in the session; the validator
refuses the call otherwise.

### 3.5 Failure

Return a non-2xx status, or a body with an `error` field:

```json
{ "status": "error", "error": { "code": "MODEL_FAILED", "message": "OOM on tile 3" } }
```

Either becomes a failed step with the message in the trace. v1's error codes
(`IMAGE_NOT_FOUND`, `MODEL_FAILED`, `PROCESSING_TIMEOUT`, …) are still a good
vocabulary for `code`; the controller passes the message through and does not
switch on the code.

### 3.6 Local tools

Two tools run inside the controller, not as services. They are listed here
because they appear in plans and traces.

| tool | args | result |
|---|---|---|
| `filter_by_region` | `objects`, `regions` (both references) | `objects` (the survivors), `kept`, `dropped` |
| `count` | `objects` (reference) | `n`, `mean_confidence` (null when `n` is 0) |

`filter_by_region` keeps an object when at least half its footprint lies inside
the merged regions. A building grazing the corner of a changed area is not a
new building.

---

## 4. Plan format

What the planner produces and the executor runs. Replaces v1's
`intent` / `workflow` contract.

```json
{
  "steps": [
    { "id": "s1", "tool": "change_detect",
      "args": { "image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt" } },
    { "id": "s2", "tool": "ground",
      "args": { "image_id": "img_2026_opt", "phrase": "buildings" } },
    { "id": "s3", "tool": "filter_by_region",
      "args": { "objects": "$s2.objects", "regions": "$s1.regions" } }
  ],
  "answer_from": "s3",
  "reasoning": "New buildings are buildings in the later image that lie inside changed regions.",
  "source": "llm",
  "provider": "gpt-oss:20b"
}
```

`$<step_id>.<field>` is the only way data moves between steps. `provider`
names what planned it: the model name for an LLM plan, `"keywords"` for the
fallback, whose `reasoning` then ends with `(every LLM provider failed: ...)`
listing each provider and its error. `source` is
`"llm"` or `"fallback"` (keyword routing, used when the LLM is unavailable or
its plan failed validation twice).

The validator rejects, with a message the model can act on:

- unknown tools; missing or unknown arguments
- step ids that are not `[A-Za-z0-9_]+`; duplicate ids; more than 6 steps
- a reference that points forwards, to itself, to a field the tool does not
  return, or to a field of the wrong kind (a number into a list argument, a
  list into a text argument)
- anything containing `$` that is not exactly a reference
- an `image_id` that is not loaded
- a literal list where a reference is required — plans never carry geometry or
  counts
- `change_detect` on two scenes from different areas, on the same scene twice,
  with `t1` later than `t2`, or with the same date
- `cross_modal` with the modalities swapped or on an unpaired scene
- `filter_by_region` whose two inputs derive, through any number of earlier
  steps, from different areas
- `answer_from` that is not a step id

---

## 5. Planner behaviour

A chain, in order: primary LLM (Ollama on a teammate's machine by default)
plan validated, with one retry carrying the validation errors verbatim; the
same against the fallback LLM (Gemini by default) if the primary is down,
rate-limited or plans invalidly twice; keyword fallback last, with every
provider's error recorded in the plan's `reasoning`. `temperature` is 0. With
no `SATQUERY_LLM_*` or `SATQUERY_LLM_FALLBACK_*` variable set, every LLM is
skipped and the keyword fallback is used, so the whole system runs offline
against the mocks.

Sensor rule: optical is the default. If the relevant optical scene is flagged
`cloudy` and a paired SAR scene of the same area is loaded, the planner uses
`cross_modal` and says so in `reasoning`.

---

## 6. Controller response

`handle_query(query, session)` returns this. It is what the backend receives.

```json
{
  "query": "Find buildings constructed after 2023",
  "plan": { "source": "fallback", "reasoning": "...", "answer_from": "s3", "steps": [ ... ] },
  "status": "ok",
  "answer": "2 object(s) fell inside the region(s) of interest.",
  "confidence": 0.79,
  "confidence_note": "limited by the weakest of 3 chained steps",
  "results": {
    "s1": { "regions": [ ... ], "changed_area_km2": 0.107,
            "change_mask_id": "mask_img_2023_opt_img_2026_opt", "mock": true },
    "s2": { "objects": [ ... ], "mock": true },
    "s3": { "objects": [ ... ], "kept": 2, "dropped": 3, "mock": true }
  },
  "trace": [
    { "step": "s1", "tool": "change_detect", "status": "ok" },
    { "step": "s2", "tool": "ground", "status": "ok" },
    { "step": "s3", "tool": "filter_by_region", "status": "ok" }
  ],
  "elapsed_s": 0.001
}
```

| field | meaning |
|---|---|
| `status` | `"ok"`, or `"partial"` if any step failed or planning itself failed |
| `answer` | one sentence, templated per tool from the step named by `answer_from`. Never written by the LLM. Prefixed with *"I'm not confident in this:"* when `confidence` is below 0.5 |
| `confidence` | the **minimum** over every confidence the plan produced — each item's, plus top-level `confidence` / `mean_confidence` fields. Never the mean. `null` if nothing produced one |
| `confidence_note` | why it is what it is: the weakest-link note, the abstain note, or the step that failed |
| `results` | every step's result keyed by step id. Steps that were skipped because their input was empty are `{ "skipped": true, "reason": "..." }`. A result that came from a mock adapter, or was computed from one, carries `"mock": true` |
| `trace` | one entry per step attempted; `status` is `ok`, `skipped` or `error`, with `detail` on the last two |
| `plan` | the validated plan that ran; `null` only if planning threw |

---

## 7. Failure behaviour

- **A step returns an empty list** (no buildings found): every step that
  depends on it is skipped, not run. The answer says *"Nothing to report"*
  rather than a confident zero.
- **A step fails** (service down, bad geometry, timeout): execution stops
  there, `status` is `"partial"`, earlier results are kept, the trace shows the
  error, and `confidence_note` names the step.
- **Confidence below 0.5**: the answer is still given, prefixed with a
  statement that the system is not confident.
- **Anything else goes wrong**: `status` is `"partial"`, `plan` is `null`,
  `answer` is *"Could not complete this query: …"*. `handle_query` never raises.

---

## 8. Traceability

`trace` is what the frontend renders as the live execution panel. Each entry
maps 1:1 to a plan step. v1's human-readable action names (`understand_query`,
`validate_inputs`, …) are not emitted; the backend can prepend them if P6 wants
the five-line version.

```json
[
  { "step": "s1", "tool": "change_detect", "status": "ok" },
  { "step": "s2", "tool": "ground", "status": "ok" },
  { "step": "s3", "tool": "filter_by_region", "status": "error",
    "detail": "filter_by_region.objects[0]: 'confidence' must be a number from 0 to 1, got '0.9'" }
]
```

---

## 9. What the backend derives for the frontend

The frontend was built against v1's unified response. The controller does not
produce `visualization` or `statistics`, because both depend on how the
frontend renders. This is a suggested mapping; adjust it with P6.

| v1 frontend field | from controller response |
|---|---|
| `query_id` | echoed from the request |
| `status` | `"success"` if `status == "ok"`, else `"partial"` |
| `analysis.workflow` | `[s.tool for s in plan.steps]` |
| `analysis.intent` | the tool of `plan.answer_from`, or `"multi_step"` if more than one step |
| `result.summary` | `answer` |
| `result.confidence` | `confidence` |
| `visualization.type` | `"bounding_boxes"` if the answer step has `objects`; `"temporal_comparison"` if the plan contains `change_detect`; `"cross_modal"` if it contains `cross_modal`; else `"single_image"` |
| `visualization.items` | the answer step's `objects` or `regions`; each has `geometry` (lon/lat) and `confidence`. Draw with a map library, or project to the image with the scene's `bounds` |
| `visualization.overlay` | `results[<change step>].change_mask_id`, fetched from the change service |
| `statistics.objects_detected` | `len(results[answer_from].objects)` |
| `statistics.changes_detected` | `len(results[<change step>].regions)` |
| `statistics.processing_time_ms` | `elapsed_s * 1000` |
| `execution.steps` | `trace` |

Pass `plan`, `trace`, `confidence_note` and the `mock` flags through unchanged
as well; the execution panel wants them, and a demo answer built on mocks
should say so.

---

## 10. Configuration

| variable | default | effect |
|---|---|---|
| `SATQUERY_MOCK` | `1` | all remote tools answer from mocks. Set `0` / `off` for real services |
| `SATQUERY_MOCK_<TOOL>` | — | per-tool override, e.g. `SATQUERY_MOCK_GROUND=0` |
| `SATQUERY_<TOOL>_URL` | `http://127.0.0.1:800N/<tool>` | vqa 8001, ground 8002, change_detect 8003, cross_modal 8004 |
| `SATQUERY_LLM_BASE_URL` | `http://localhost:11434/v1` (Ollama) | primary planner endpoint, any OpenAI-compatible URL. Point at the teammate running Ollama. Groq: `https://api.groq.com/openai/v1` |
| `SATQUERY_LLM_MODEL` | `gpt-oss:20b` | must use the endpoint's own naming (Groq: `openai/gpt-oss-20b`); check the provider's live model list |
| `SATQUERY_LLM_API_KEY` | — | not needed for Ollama. Setting any of the three `SATQUERY_LLM_*` variables enables the primary; none set means it is skipped |
| `SATQUERY_LLM_FALLBACK_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` (Gemini) | fallback planner endpoint, tried when the primary fails or plans invalidly twice |
| `SATQUERY_LLM_FALLBACK_MODEL` | `gemini-3.6-flash` | Gemini free tier is ~20 requests/day per model, hence fallback only |
| `SATQUERY_LLM_FALLBACK_API_KEY` | — | Google AI Studio key; setting it enables the fallback. With no provider enabled at all the planner is keyword-only |

Swap one tool at a time to real services so integration problems arrive one at
a time.

---

## Ownership

| contract | owner | status |
|---|---|---|
| §1 request | backend + frontend | unchanged from v1 |
| §2 scene metadata | backend (loads from BigEarthNet parquet + GeoTIFF) | new fields |
| §3.1–3.4 services | P2 / P3 / P4 / cross-modal developer | **binding; enforced by the controller** |
| §4–§5 plan | controller | internal, documented for the execution panel |
| §6–§8 response | controller → backend | binding |
| §9 derivation | backend + frontend | suggested; agree with P6 |
