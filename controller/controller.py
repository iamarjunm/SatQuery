"""The controller's single public entry point.

Everything else in this package (planner, executor, tools, plan, session)
exists to make handle_query() possible: turn a query into a plan, run the
plan, and describe the result in one sentence a non-expert can read.
"""

from __future__ import annotations

import time
from typing import Any

from executor import ABSTAIN_THRESHOLD, run
from planner import plan_query
from session import Session


def _describe(plan, results: dict[str, dict]) -> str:
    """A human-readable sentence built from the step named by answer_from.
    Kept as simple templating per tool, since the LLM never gets to phrase
    the answer -- that would let it fabricate a number no tool computed."""
    step = plan.step(plan.answer_from)
    result = results.get(plan.answer_from, {})
    if step is None or not result:
        return "No answer could be produced."
    if result.get("skipped"):
        return f"Nothing to report: {result.get('reason', 'an earlier step returned nothing')}."

    if step.tool == "vqa":
        return str(result.get("answer", "No answer."))
    if step.tool == "ground":
        n = len(result.get("objects", []))
        return f"Found {n} matching object{'s' if n != 1 else ''}."
    if step.tool == "change_detect":
        area = result.get("changed_area_km2")
        n = len(result.get("regions", []))
        return f"Detected {n} changed region{'s' if n != 1 else ''} covering {area} km^2."
    if step.tool == "cross_modal":
        return str(result.get("summary", "No summary."))
    if step.tool == "filter_by_region":
        return f"{result.get('kept', 0)} object(s) fell inside the region(s) of interest."
    if step.tool == "count":
        n, mean = result.get("n", 0), result.get("mean_confidence")
        return f"Counted {n} object{'s' if n != 1 else ''}" + (
            f" (mean confidence {mean:.2f})." if mean is not None else "."
        )
    return "Done."


def handle_query(query: str, session: Session) -> dict[str, Any]:
    """Plan the query, run it, and return the response shape the frontend
    expects. Always returns a valid dict -- an unexpected failure anywhere in
    planning or execution degrades to status="partial" instead of raising,
    per the doc's "every failure mode returns valid JSON" rule.
    """
    started = time.monotonic()
    try:
        plan = plan_query(query, session)
        outcome = run(plan, session)
        answer = _describe(plan, outcome["results"])
        if outcome["confidence"] is not None and outcome["confidence"] < ABSTAIN_THRESHOLD:
            answer = f"I'm not confident in this: {answer}"
    except Exception as exc:  # noqa: BLE001 - last-resort guard, see docstring
        return {
            "query": query,
            "plan": None,
            "status": "partial",
            "answer": f"Could not complete this query: {exc}",
            "confidence": None,
            "confidence_note": "",
            "results": {},
            "trace": [],
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    return {
        "query": query,
        "plan": plan.to_dict(),
        "status": outcome["status"],
        "answer": answer,
        "confidence": outcome["confidence"],
        "confidence_note": outcome["confidence_note"],
        "results": outcome["results"],
        "trace": outcome["trace"],
        "elapsed_s": round(time.monotonic() - started, 3),
    }
