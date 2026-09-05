"""The planner: turns an English query into a validated JSON Plan.

The planner is the only non-deterministic piece of the controller. It never
touches pixels and never produces a coordinate or a count itself -- it only
picks tools and wires their arguments together (see plan.py). Everything here
exists to get from free text to a Plan that plan.validate() accepts, with a
three-layer fallback so a flaky LLM degrades gracefully instead of crashing
the demo:

  1. LLM plan, validated.
  2. LLM retry, with the validation error appended verbatim.
  3. Deterministic keyword fallback (Plan.source == "fallback").
"""

from __future__ import annotations

import json
import os
import re

from plan import Plan, PlanFormatError, validate
from session import Session
from tools import render_menu

# Routing is a short-input, tiny-output task, so a small open-weights model is
# enough; temperature 0 so the same query always plans the same way.
DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
MAX_LLM_ATTEMPTS = 2

ROLE = (
    "You are the planner for a satellite-imagery question answering system. "
    "You never analyse imagery yourself and you never produce a coordinate, a "
    "geometry or a count -- you only choose tools from the menu below and wire "
    "their inputs together into a JSON plan. Every other module does the actual "
    "computation."
)

RULES = """Rules:
1. Use only image_ids that appear in "Loaded images" below. Never invent one.
2. Pass data between steps with "$<step_id>.<field>"; references may only
   point to a step earlier in the list, never to the step's own output.
3. Use the fewest steps that answer the question.
4. A question about things NEW / BUILT / REMOVED / APPEARED between two dates
   cannot be answered by one tool -- chain change_detect -> ground ->
   filter_by_region.
5. Optical is the default sensor. If the relevant optical image is cloudy and
   a SAR image of the same area and date is loaded, use the SAR image instead
   and say so in "reasoning".
6. If nothing in the menu fits the question, emit a single vqa step with the
   closest sensible question.

Respond with a single JSON object and nothing else -- no prose, no markdown
fences. The object must have this shape:
{"steps": [{"id": "s1", "tool": "<tool name>", "args": {...}}, ...],
 "answer_from": "<id of the step whose result answers the question>",
 "reasoning": "<one sentence>"}"""

# Few-shot examples. One per major routing case in the doc, so a fresh model
# only has to pattern-match, not invent structure. Kept short on purpose --
# per the implementation doc, fixes belong here as examples, not as more
# prose in RULES.
EXAMPLES = [
    (
        "Loaded images:\n- img_a: optical, sentinel-2, 2026-06-15, area A\n"
        "Question: What is in this image?",
        {"steps": [{"id": "s1", "tool": "vqa",
                    "args": {"image_id": "img_a", "question": "What is in this image?"}}],
         "answer_from": "s1", "reasoning": "A description question needs no localisation."},
    ),
    (
        "Loaded images:\n- img_a: optical, sentinel-2, 2026-06-15, area A\n"
        "Question: Find all the buildings.",
        {"steps": [{"id": "s1", "tool": "ground", "args": {"image_id": "img_a", "phrase": "buildings"}}],
         "answer_from": "s1", "reasoning": "A find/locate question is answered by grounding."},
    ),
    (
        "Loaded images:\n- img_2023: optical, sentinel-2, 2023-06-15, area A\n"
        "- img_2026: optical, sentinel-2, 2026-06-15, area A\n"
        "Question: What changed between 2023 and 2026?",
        {"steps": [{"id": "s1", "tool": "change_detect",
                    "args": {"image_id_t1": "img_2023", "image_id_t2": "img_2026"}}],
         "answer_from": "s1", "reasoning": "A between-dates comparison is a single change_detect call."},
    ),
    (
        "Loaded images:\n- img_2023_opt: optical, sentinel-2, 2023-06-15, area A\n"
        "- img_2026_opt: optical, sentinel-2, 2026-06-15, area A\n"
        "Question: Find buildings constructed after 2023.",
        {"steps": [
            {"id": "s1", "tool": "change_detect",
             "args": {"image_id_t1": "img_2023_opt", "image_id_t2": "img_2026_opt"}},
            {"id": "s2", "tool": "ground", "args": {"image_id": "img_2026_opt", "phrase": "buildings"}},
            {"id": "s3", "tool": "filter_by_region", "args": {"objects": "$s2.objects", "regions": "$s1.regions"}},
        ], "answer_from": "s3",
         "reasoning": "New buildings are buildings in the later image that lie inside changed regions."},
    ),
    (
        "Loaded images:\n- img_opt: optical, sentinel-2, 2026-07-02, area A [cloudy, paired with img_sar]\n"
        "- img_sar: sar, sentinel-1, 2026-07-02, area A [paired with img_opt]\n"
        "Question: Find the built-up areas.",
        {"steps": [{"id": "s1", "tool": "cross_modal",
                    "args": {"optical_image_id": "img_opt", "sar_image_id": "img_sar", "phrase": "built-up areas"}}],
         "answer_from": "s1",
         "reasoning": "The optical image is cloudy; a co-registered SAR image is loaded, so cross_modal is used instead."},
    ),
    (
        "Loaded images:\n- img_a: optical, sentinel-2, 2026-06-15, area A\n"
        "Question: How many vehicles are there?",
        {"steps": [
            {"id": "s1", "tool": "ground", "args": {"image_id": "img_a", "phrase": "vehicles"}},
            {"id": "s2", "tool": "count", "args": {"objects": "$s1.objects"}},
        ], "answer_from": "s2", "reasoning": "A tally requires grounding first, then counting the results."},
    ),
]

# The fenced-json pattern strips a "```json ... ```" wrapper some models add
# despite being told not to; the brace pattern is the last-resort fallback.
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BRACES = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Best-effort recovery of a JSON object from a raw completion."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    else:
        braces = _BRACES.search(text)
        if braces:
            text = braces.group(0)
    return json.loads(text)


def _prompt_messages(query: str, session: Session, retry_error: str | None) -> list[dict]:
    system = "\n\n".join([ROLE, "Tools:\n" + render_menu(), RULES])
    messages = [{"role": "system", "content": system}]
    for user_text, assistant_json in EXAMPLES:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": json.dumps(assistant_json)})

    user_text = f"Loaded images:\n{session.describe()}\nQuestion: {query}"
    if retry_error:
        # Fed back verbatim, per the implementation doc: this is what fixes
        # most failures on the second attempt without a human in the loop.
        user_text += (
            "\n\nYour previous plan was rejected for this reason, fix it and "
            f"respond with a corrected plan only:\n{retry_error}"
        )
    messages.append({"role": "user", "content": user_text})
    return messages


def _call_llm(messages: list[dict]) -> str:
    """One completion call. Raises on any failure; callers decide what to do
    with that (retry, then fall back to keywords). Imported lazily so a
    machine with no LLM configured can still import this module and run the
    keyword fallback."""
    from openai import OpenAI  # Groq, OpenRouter and Ollama are all OpenAI-compatible.

    client = OpenAI(
        base_url=os.getenv("SATQUERY_LLM_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.getenv("SATQUERY_LLM_API_KEY", "unset"),
    )
    response = client.chat.completions.create(
        model=os.getenv("SATQUERY_LLM_MODEL", DEFAULT_MODEL),
        messages=messages,
        temperature=0,  # non-negotiable: the same query must plan identically every time.
    )
    return response.choices[0].message.content or ""


def _llm_plan(query: str, session: Session) -> Plan | None:
    """Try the LLM, with one retry against the validator's own error message.
    Returns None (never raises) so plan_query can fall through to keywords."""
    # No key configured: skip straight to the keyword fallback instead of
    # spending the retry budget on a call that cannot succeed. Also lets the
    # test suite and any offline run exercise the fallback without a network
    # timeout, the same way tools.py defaults to mocks with nothing configured.
    if not os.getenv("SATQUERY_LLM_API_KEY"):
        return None

    retry_error: str | None = None
    for _ in range(MAX_LLM_ATTEMPTS):
        try:
            raw_text = _call_llm(_prompt_messages(query, session, retry_error))
            raw = _extract_json(raw_text)
            plan = Plan.from_dict(raw, source="llm")
        except (PlanFormatError, json.JSONDecodeError, Exception) as exc:
            retry_error = str(exc)
            continue

        errors = validate(plan, session)
        if not errors:
            return plan
        retry_error = "; ".join(errors)

    return None


# --- keyword fallback ---------------------------------------------------------

# Checked in order; the first match wins. This is degraded mode only -- it
# exists so a rate-limited or unreachable LLM still returns something
# reasonable, never as the primary router.
_CHANGE_WORDS = ("chang", "differ", "compar", "new", "built", "construct", "remov", "appear", "demolish")
_FIND_WORDS = ("find", "show", "locate", "highlight", "where")
_COUNT_WORDS = ("how many", "count", "tally", "number of")


def _keyword_plan(query: str, session: Session) -> Plan:
    q = query.lower()
    ids = session.ids()

    # Prefer a same-area optical pair, oldest first, for anything chained or
    # compared -- without one there is nothing sensible to fall back to.
    by_area: dict[str, list] = {}
    for image_id in ids:
        scene = session.get(image_id)
        if scene.modality == "optical":
            by_area.setdefault(session.areas()[image_id], []).append(scene)
    pair = next((sorted(scenes, key=lambda s: s.acquired) for scenes in by_area.values()
                 if len(scenes) >= 2), None)

    if pair and any(w in q for w in _CHANGE_WORDS):
        t1, t2 = pair[0].image_id, pair[-1].image_id
        if any(w in q for w in ("new", "built", "construct", "appear")):
            raw = {
                "steps": [
                    {"id": "s1", "tool": "change_detect", "args": {"image_id_t1": t1, "image_id_t2": t2}},
                    {"id": "s2", "tool": "ground", "args": {"image_id": t2, "phrase": "buildings"}},
                    {"id": "s3", "tool": "filter_by_region", "args": {"objects": "$s2.objects", "regions": "$s1.regions"}},
                ],
                "answer_from": "s3",
                "reasoning": "keyword fallback: change-then-filter chain for a new/built/appeared query",
            }
        else:
            raw = {
                "steps": [{"id": "s1", "tool": "change_detect", "args": {"image_id_t1": t1, "image_id_t2": t2}}],
                "answer_from": "s1",
                "reasoning": "keyword fallback: comparison query routed to change_detect",
            }
        return Plan.from_dict(raw, source="fallback")

    image_id = ids[0] if ids else None
    if image_id and any(w in q for w in _COUNT_WORDS):
        phrase = _guess_phrase(q) or "objects"
        raw = {
            "steps": [
                {"id": "s1", "tool": "ground", "args": {"image_id": image_id, "phrase": phrase}},
                {"id": "s2", "tool": "count", "args": {"objects": "$s1.objects"}},
            ],
            "answer_from": "s2",
            "reasoning": "keyword fallback: tally query routed to ground-then-count",
        }
        return Plan.from_dict(raw, source="fallback")

    if image_id and any(w in q for w in _FIND_WORDS):
        raw = {
            "steps": [{"id": "s1", "tool": "ground",
                       "args": {"image_id": image_id, "phrase": _guess_phrase(q) or "objects"}}],
            "answer_from": "s1",
            "reasoning": "keyword fallback: find/show/locate query routed to ground",
        }
        return Plan.from_dict(raw, source="fallback")

    # Nothing matched, or no images are loaded at all: ask the closest
    # sensible question rather than refusing to produce a plan.
    raw = {
        "steps": [{"id": "s1", "tool": "vqa",
                   "args": {"image_id": image_id or "none", "question": query}}],
        "answer_from": "s1",
        "reasoning": "keyword fallback: no rule matched, asking the question directly",
    }
    return Plan.from_dict(raw, source="fallback")


def _guess_phrase(query: str) -> str | None:
    """Pull a plausible noun phrase out of a find/count query, e.g. 'find all
    the buildings' -> 'buildings'. Best-effort only; a wrong guess is still a
    valid plan the executor can run, just not a sharp one."""
    match = re.search(r"(?:find|show|locate|count|highlight|of)\s+(?:all\s+|the\s+)*([a-z][a-z \-]*)", query)
    if not match:
        return None
    phrase = match.group(1).strip().rstrip("?.! ")
    return phrase or None


def plan_query(query: str, session: Session) -> Plan:
    """The planner's public entry point. Always returns a Plan -- it never
    raises, because a routing failure should degrade the answer, not the
    controller. Tries the LLM first (validated, retried once against its own
    error), then falls back to keyword routing."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    plan = _llm_plan(query, session)
    if plan is not None:
        return plan
    return _keyword_plan(query, session)
