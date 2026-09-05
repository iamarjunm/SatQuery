"""The executor: runs a validated Plan step by step.

Deterministic, plain Python -- unlike the planner, nothing here guesses.
Responsibilities (per the implementation doc):
  - resolve every "$step.field" reference before the call it belongs to
  - short-circuit a step whose list input came back empty, instead of
    running a dependent step over nothing
  - keep partial results and an honest note if a step fails, rather than
    crashing the whole query
  - report confidence as the minimum across steps, never the mean, and
    say so when that minimum falls below the abstain threshold
  - emit one trace event per step, for the caller to render as a live log
"""

from __future__ import annotations

from typing import Any

from plan import Plan, Step, parse_ref
from session import Session
from tools import LIST_KINDS, TOOLS, ToolError, call

# Below this, the answer says it is not sure rather than guessing.
ABSTAIN_THRESHOLD = 0.5


def _resolve_args(step: Step, results: dict[str, dict], mocked: set[str]) -> tuple[dict, bool]:
    """Substitute every "$step.field" argument with the value it names.

    Returns (resolved_args, used_mocked_input) -- the second lets the caller
    mark a step's own result as mock-derived once any of its inputs were,
    so that provenance survives a chain of otherwise-real steps.
    """
    resolved: dict[str, Any] = {}
    used_mocked = False
    for name, value in step.args.items():
        ref = parse_ref(value)
        if ref is None:
            resolved[name] = value
            continue
        target_id, field = ref
        resolved[name] = results[target_id][field]
        if target_id in mocked:
            used_mocked = True
    return resolved, used_mocked


def _confidences(tool: str, result: dict) -> list[float]:
    """Every confidence value a tool's result actually carries. Item-level
    confidences (objects/regions) count individually, not just the top-level
    field, so one bad detection in an otherwise-strong batch still pulls the
    reported number down."""
    spec = TOOLS[tool]
    scores: list[float] = []
    for field, ret in spec.returns.items():
        value = result.get(field)
        if ret.kind in LIST_KINDS and isinstance(value, list):
            scores.extend(
                item["confidence"] for item in value
                if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))
            )
        elif field in ("confidence", "mean_confidence") and isinstance(value, (int, float)):
            scores.append(float(value))
    return scores


def _empty_upstream(step: Step, results: dict[str, dict]) -> str | None:
    """If a list-kind argument resolves to an empty list, name the field --
    the step is skipped rather than called with nothing to work on."""
    spec = TOOLS.get(step.tool)
    if spec is None:
        return None
    for name, value in step.args.items():
        ref = parse_ref(value)
        if ref is None:
            continue
        arg = spec.args.get(name)
        if arg is None or arg.kind not in LIST_KINDS:
            continue
        target_id, field = ref
        if results.get(target_id, {}).get(field) == []:
            return name
    return None


def run(plan: Plan, session: Session) -> dict[str, Any]:
    """Execute every step in order and return results, a trace, and an
    overall status. Never raises -- a tool failure ends the walk early and is
    reported as a partial result, per the doc's "partial results" rule."""
    results: dict[str, dict] = {}
    trace: list[dict] = []
    mocked: set[str] = set()
    confidences: list[float] = []
    status = "ok"
    note = ""

    for step in plan.steps:
        empty_arg = _empty_upstream(step, results)
        if empty_arg is not None:
            # Nothing to compute on. Recording this as its own result field
            # (rather than skipping the entry entirely) means a later step
            # or answer_from that points here still finds a well-shaped,
            # empty value instead of a KeyError.
            results[step.id] = {"skipped": True, "reason": f"'{empty_arg}' was empty"}
            mocked.add(step.id)
            trace.append({"step": step.id, "tool": step.tool, "status": "skipped",
                           "detail": f"upstream '{empty_arg}' returned nothing"})
            continue

        try:
            args, used_mocked = _resolve_args(step, results, mocked)
            result = call(step.tool, args, session, mocked_inputs=used_mocked)
        except ToolError as exc:
            status = "partial"
            note = f"stopped at step {step.id} ({step.tool}): {exc}"
            trace.append({"step": step.id, "tool": step.tool, "status": "error", "detail": str(exc)})
            break

        results[step.id] = result
        if used_mocked or result.get("mock"):
            mocked.add(step.id)
        confidences.extend(_confidences(step.tool, result))
        trace.append({"step": step.id, "tool": step.tool, "status": "ok"})

    confidence = min(confidences) if confidences else None
    confidence_note = note
    if confidence is not None:
        if len(confidences) > 1 or len(plan.steps) > 1:
            confidence_note = confidence_note or f"limited by the weakest of {len(plan.steps)} chained steps"
        if confidence < ABSTAIN_THRESHOLD:
            confidence_note = (
                f"confidence {confidence:.2f} is below the abstain threshold; "
                "treat this answer as unsure"
            ) if not note else note

    return {
        "results": results,
        "trace": trace,
        "status": status,
        "confidence": confidence,
        "confidence_note": confidence_note,
    }
