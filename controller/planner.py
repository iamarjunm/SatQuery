"""The planner: turns an English query into a validated JSON Plan.

The planner is the only non-deterministic piece of the controller. It never
touches pixels and never produces a coordinate or a count itself -- it only
picks tools and wires their arguments together (see plan.py). Everything here
exists to get from free text to a Plan that plan.validate() accepts, with a
fallback chain so a flaky LLM degrades gracefully instead of crashing the
demo:

  1. Primary LLM plan, validated; one retry with the validation error
     appended verbatim.
  2. The same against the fallback LLM (a hosted provider with its own
     quota, Gemini by default) when the primary is down or rate-limited.
  3. Deterministic keyword fallback (Plan.source == "fallback"), whose
     reasoning records why every LLM was skipped, so a degraded answer is
     never mistaken for a planned one.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from plan import Plan, PlanFormatError, validate
from session import Session
from tools import render_menu

# Routing is a short-input, tiny-output task; any capable chat model will do.
# Every provider is OpenAI-compatible, so each is just a base URL, a key and a
# model name. The primary defaults to Ollama (a teammate's machine, no key);
# the fallback defaults to Gemini's hosted endpoint, which needs a key.
# temperature 0 so the same query always plans the same way.
DEFAULT_MODEL = "gpt-oss:20b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_FALLBACK_MODEL = "gemini-3.6-flash"
DEFAULT_FALLBACK_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MAX_LLM_ATTEMPTS = 2
LLM_TIMEOUT_S = 60  # same hard ceiling as a tool call; a hung provider must not hang the demo


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible endpoint in the chain. `name` is "primary" or
    "fallback" and only appears in diagnostics."""
    name: str
    base_url: str
    api_key: str
    model: str

    def describe(self) -> str:
        return f"{self.name} {self.model} @ {self.base_url}"


def _provider(name: str, prefix: str, default_url: str, default_model: str) -> Provider | None:
    """Build one provider from <prefix>_BASE_URL / _MODEL / _API_KEY. Setting
    any of the three enables it, so a keyless Ollama needs only a URL or a
    model name, and a hosted endpoint needs only its key. Nothing set means
    the provider is skipped, which keeps the test suite and any offline run
    on the keyword fallback without a network timeout, the same way tools.py
    defaults to mocks with nothing configured."""
    url, model, key = (os.getenv(f"{prefix}_{k}") for k in ("BASE_URL", "MODEL", "API_KEY"))
    if not (url or model or key):
        return None
    # Ollama ignores the key but the OpenAI client insists on a non-empty one.
    return Provider(name, base_url=url or default_url, api_key=key or "ollama", model=model or default_model)


def providers() -> list[Provider]:
    """The chain, in the order it is tried, read from the environment each
    call so a key or URL can be changed without restarting anything.

    Primary:  SATQUERY_LLM_*            defaults to Ollama on localhost; point
                                        BASE_URL at the teammate running it.
    Fallback: SATQUERY_LLM_FALLBACK_*   defaults to Gemini; needs its API key.
    """
    chain = [
        _provider("primary", "SATQUERY_LLM", DEFAULT_BASE_URL, DEFAULT_MODEL),
        _provider("fallback", "SATQUERY_LLM_FALLBACK", DEFAULT_FALLBACK_BASE_URL, DEFAULT_FALLBACK_MODEL),
    ]
    return [p for p in chain if p is not None]

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


def _call_llm(messages: list[dict], provider: Provider) -> str:
    """One completion call against one provider. Raises on any failure;
    callers decide what to do with that (retry, next provider, keywords).
    Imported lazily so a machine with no LLM configured can still import this
    module and run the keyword fallback."""
    from openai import OpenAI  # Gemini, Groq, OpenRouter and Ollama are all OpenAI-compatible.

    # max_retries=1: a quota error or a dead host should hand over to the
    # next provider in seconds, not sit in the client's own backoff loop.
    client = OpenAI(base_url=provider.base_url, api_key=provider.api_key,
                    timeout=LLM_TIMEOUT_S, max_retries=1)
    response = client.chat.completions.create(
        model=provider.model,
        messages=messages,
        temperature=0,  # non-negotiable: the same query must plan identically every time.
    )
    return response.choices[0].message.content or ""


def _short(exc: Exception) -> str:
    """One line of an exception, enough to tell a 429 from a refused
    connection from a malformed plan, without the provider's whole payload."""
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text[:160]}"


def _plan_with(provider: Provider, query: str, session: Session) -> tuple[Plan | None, str]:
    """Try one provider, with one retry against the validator's own error
    message. Returns (plan, "") on success or (None, reason) otherwise."""
    retry_error: str | None = None
    for _ in range(MAX_LLM_ATTEMPTS):
        try:
            raw_text = _call_llm(_prompt_messages(query, session, retry_error), provider)
            raw = _extract_json(raw_text)
            plan = Plan.from_dict(raw, source="llm", provider=provider.model)
        except (PlanFormatError, json.JSONDecodeError, Exception) as exc:
            retry_error = _short(exc)
            continue

        errors = validate(plan, session)
        if not errors:
            return plan, ""
        retry_error = "; ".join(errors)

    return None, retry_error or "no attempt made"


def _llm_plan(query: str, session: Session) -> tuple[Plan | None, list[str]]:
    """Walk the provider chain. Returns the first valid plan, plus one line per
    provider that failed, so the caller can say why it degraded. Never raises.
    With nothing configured this returns immediately, which lets the test suite
    and any offline run exercise the keyword fallback without a network timeout,
    the same way tools.py defaults to mocks with nothing configured."""
    failures: list[str] = []
    for provider in providers():
        plan, reason = _plan_with(provider, query, session)
        if plan is not None:
            return plan, failures
        failures.append(f"{provider.describe()}: {reason}")
    return None, failures


# --- keyword fallback ---------------------------------------------------------

# Checked in order; the first match wins. This is degraded mode only -- it
# exists so a rate-limited or unreachable LLM still returns something
# reasonable, never as the primary router.
_CHANGE_WORDS = ("chang", "differ", "compar", "new", "built", "construct", "remov", "appear", "demolish")
_FIND_WORDS = ("find", "show", "locate", "highlight", "where")
_COUNT_WORDS = ("how many", "count", "tally", "number of")
_CROSS_MODAL_WORDS = ("sar", "radar", "both sensors", "optical and radar", "cross-modal", "cross modal")


def _keyword_plan(query: str, session: Session) -> Plan:
    q = query.lower()
    ids = session.ids()

    # Sensor selection as a planning decision (Sec 5 rule), checked first so an
    # explicit both-sensors request wins over an incidental _CHANGE_WORDS
    # match ("built-up" contains "built"). change_detect and cross_modal are
    # not actually in tension in practice -- one compares two dates of the
    # same sensor, the other fuses two sensors of (presumably) the same date.
    optical_sar_pairs = []
    for candidate_id in ids:
        scene = session.get(candidate_id)
        if scene.modality != "optical":
            continue
        counterpart = session.counterpart(candidate_id)
        if counterpart is not None and counterpart.modality == "sar":
            optical_sar_pairs.append((scene, counterpart))

    if optical_sar_pairs:
        explicit_request = any(w in q for w in _CROSS_MODAL_WORDS)
        # Sec 5's cloud trigger is about the scene a plain query would
        # otherwise land on (the same default used below, ids[0]) -- not
        # "a cloudy pair exists somewhere in the session". Scanning the
        # whole session would hijack unrelated find/tally/change queries
        # into cross_modal just because some other cloudy scene happens to
        # be loaded alongside them.
        default_id = ids[0] if ids else None
        default_pair = next((p for p in optical_sar_pairs if p[0].image_id == default_id), None)
        cloud_triggered = default_pair is not None and default_pair[0].cloudy
        if explicit_request or cloud_triggered:
            optical_scene, sar_scene = default_pair or optical_sar_pairs[0]
            raw = {
                "steps": [{"id": "s1", "tool": "cross_modal",
                           "args": {"optical_image_id": optical_scene.image_id,
                                    "sar_image_id": sar_scene.image_id,
                                    "phrase": _guess_phrase(q) or "built-up areas"}}],
                "answer_from": "s1",
                "reasoning": (
                    "keyword fallback: default optical scene is cloudy and a "
                    "paired SAR scene is loaded, routed to cross_modal"
                    if cloud_triggered else
                    "keyword fallback: query names both sensors, routed to cross_modal"
                ),
            }
            return Plan.from_dict(raw, source="fallback", provider="keywords")

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
        return Plan.from_dict(raw, source="fallback", provider="keywords")

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
        return Plan.from_dict(raw, source="fallback", provider="keywords")

    if image_id and any(w in q for w in _FIND_WORDS):
        raw = {
            "steps": [{"id": "s1", "tool": "ground",
                       "args": {"image_id": image_id, "phrase": _guess_phrase(q) or "objects"}}],
            "answer_from": "s1",
            "reasoning": "keyword fallback: find/show/locate query routed to ground",
        }
        return Plan.from_dict(raw, source="fallback", provider="keywords")

    # Nothing matched, or no images are loaded at all: ask the closest
    # sensible question rather than refusing to produce a plan.
    raw = {
        "steps": [{"id": "s1", "tool": "vqa",
                   "args": {"image_id": image_id or "none", "question": query}}],
        "answer_from": "s1",
        "reasoning": "keyword fallback: no rule matched, asking the question directly",
    }
    return Plan.from_dict(raw, source="fallback", provider="keywords")


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

    plan, failures = _llm_plan(query, session)
    if plan is not None:
        return plan
    fallback = _keyword_plan(query, session)
    if not failures:
        return fallback
    # Say why. "keyword fallback: ..." alone looks like a design choice; with
    # the provider errors attached it reads as the degradation it is.
    why = " | ".join(failures)
    return Plan(
        steps=fallback.steps,
        answer_from=fallback.answer_from,
        reasoning=f"{fallback.reasoning} (every LLM provider failed: {why})",
        source="fallback",
        provider="keywords",
    )
