"""Plan representation and validation.

A plan is the only thing the planner is allowed to produce. It names tools and
wires their arguments together; it never contains a coordinate, a count or an
answer. Everything below exists so that a malformed or nonsensical plan is
rejected with a message specific enough to hand straight back to the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from session import Scene, Session
from tools import LIST_KINDS, TOOLS

MAX_STEPS = 6

# Step ids are restricted so that every id can appear in a reference.
STEP_ID = re.compile(r"[A-Za-z0-9_]+")

# $<step_id>.<field> is the only data-passing mechanism between steps.
REF = re.compile(r"\$([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)")

PlanSource = Literal["llm", "fallback"]


class PlanFormatError(ValueError):
    """The payload is not shaped like a plan at all."""


def parse_ref(value: Any) -> tuple[str, str] | None:
    """Return (step_id, field) if `value` is exactly a reference, else None."""
    if not isinstance(value, str):
        return None
    match = REF.fullmatch(value)
    return (match.group(1), match.group(2)) if match else None


def looks_like_ref(value: Any) -> bool:
    """A '$' anywhere in a value that is not exactly a reference is a mistyped
    or embedded reference, not prompt text. It must be rejected, or the literal
    string '$s1.answer' would be sent to a model as its question."""
    return isinstance(value, str) and "$" in value and parse_ref(value) is None


@dataclass(frozen=True)
class Step:
    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "tool": self.tool, "args": self.args}


@dataclass(frozen=True)
class Plan:
    steps: list[Step]
    answer_from: str
    reasoning: str = ""
    source: PlanSource = "llm"
    # Which planner produced it: the model name for an LLM plan, "keywords"
    # for the fallback. Diagnostic only; the frontend keys off `source`.
    provider: str = ""

    @classmethod
    def from_dict(cls, raw: Any, *, source: PlanSource = "llm", provider: str = "") -> "Plan":
        if not isinstance(raw, dict):
            raise PlanFormatError(f"expected a JSON object, got {type(raw).__name__}")

        raw_steps = raw.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlanFormatError("'steps' must be a non-empty list")

        steps = []
        for i, item in enumerate(raw_steps):
            if not isinstance(item, dict):
                raise PlanFormatError(f"steps[{i}] must be an object")
            step_id, tool = item.get("id"), item.get("tool")
            if not isinstance(step_id, str) or not STEP_ID.fullmatch(step_id):
                raise PlanFormatError(
                    f"steps[{i}] needs an 'id' made of letters, digits and underscores, "
                    f"got {step_id!r}"
                )
            if not isinstance(tool, str) or not tool:
                raise PlanFormatError(f"steps[{i}] is missing a string 'tool'")
            args = item.get("args", {})
            if not isinstance(args, dict):
                raise PlanFormatError(f"steps[{i}] 'args' must be an object")
            steps.append(Step(id=step_id, tool=tool, args=args))

        answer_from = raw.get("answer_from")
        if not isinstance(answer_from, str) or not answer_from:
            raise PlanFormatError("'answer_from' must name the step holding the final result")

        return cls(
            steps=steps,
            answer_from=answer_from,
            reasoning=str(raw.get("reasoning", "")),
            source=source,
            provider=provider,
        )

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "provider": self.provider,
            "reasoning": self.reasoning,
            "answer_from": self.answer_from,
            "steps": [s.to_dict() for s in self.steps],
        }


def validate(plan: Plan, session: Session) -> list[str]:
    """Return every problem with the plan, phrased for the model to act on.

    Collecting all errors rather than raising on the first one means a single
    retry can fix several mistakes at once.
    """
    errors: list[str] = []

    if len(plan.steps) > MAX_STEPS:
        errors.append(f"plan has {len(plan.steps)} steps; the maximum is {MAX_STEPS}")

    seen: set[str] = set()
    provenance: dict[str, list[Scene]] = {}
    for position, step in enumerate(plan.steps):
        if step.id in seen:
            errors.append(f"duplicate step id '{step.id}'; every step needs a unique id")
        seen.add(step.id)

        spec = TOOLS.get(step.tool)
        if spec is None:
            errors.append(
                f"step {step.id}: tool '{step.tool}' does not exist. "
                f"Available tools: {', '.join(TOOLS)}"
            )
            continue

        missing = spec.required_args - set(step.args)
        if missing:
            errors.append(
                f"step {step.id}: {step.tool} is missing required args: {', '.join(sorted(missing))}"
            )
        unknown = set(step.args) - set(spec.args)
        if unknown:
            errors.append(
                f"step {step.id}: {step.tool} does not accept args: {', '.join(sorted(unknown))}. "
                f"It accepts: {', '.join(spec.args)}"
            )

        for name, value in step.args.items():
            arg = spec.args.get(name)
            if arg is None:
                continue
            errors.extend(_check_arg(plan, session, position, step, name, arg.kind, value))

        provenance[step.id] = _provenance(step, provenance, session)
        errors.extend(_check_semantics(step, provenance[step.id], session))

    if plan.step(plan.answer_from) is None:
        errors.append(
            f"answer_from '{plan.answer_from}' is not a step in this plan. "
            f"Step ids are: {', '.join(s.id for s in plan.steps)}"
        )

    return errors


def _check_arg(
    plan: Plan, session: Session, position: int, step: Step, name: str, kind: str, value: Any
) -> list[str]:
    ref = parse_ref(value)

    if ref is not None:
        return _check_ref(plan, position, step, name, kind, ref)

    if looks_like_ref(value):
        return [
            f"step {step.id}: '{name}' = {value!r} looks like a reference but is not one. "
            f"A reference is the whole value, in the form $<step_id>.<field>, e.g. $s1.objects"
        ]

    if kind == "image_id":
        if not isinstance(value, str):
            return [f"step {step.id}: '{name}' must be an image_id string, got {type(value).__name__}"]
        if not session.has(value):
            return [
                f"step {step.id}: image '{value}' is not loaded. "
                f"Use one of: {', '.join(session.ids()) or 'none'}"
            ]
        return []

    if kind == "text":
        if not isinstance(value, str) or not value.strip():
            return [f"step {step.id}: '{name}' must be a non-empty string"]
        return []

    if kind in LIST_KINDS:
        # Never a literal. Geometry and counts come from tools, not from the
        # plan; a literal list would carry numbers nothing ever computed and
        # would sidestep both the provenance and the footprint checks.
        return [
            f"step {step.id}: '{name}' must be a reference to an earlier step's output, "
            f"like $<step_id>.{kind}. Plans do not contain geometry or counts directly."
        ]

    return []


def _compatible(arg_kind: str, return_kind: str) -> bool:
    """Objects and regions are both lists of geometry items and may feed each
    other. Text feeds text. Nothing else lines up."""
    if arg_kind in LIST_KINDS:
        return return_kind in LIST_KINDS
    return arg_kind == return_kind


def _check_ref(
    plan: Plan, position: int, step: Step, name: str, kind: str, ref: tuple[str, str]
) -> list[str]:
    target_id, target_field = ref
    earlier = {s.id: s for s in plan.steps[:position]}

    if target_id == step.id:
        return [f"step {step.id}: '{name}' refers to the step's own output, ${target_id}.{target_field}"]

    if target_id not in earlier:
        if plan.step(target_id) is not None:
            return [
                f"step {step.id}: '{name}' refers to ${target_id}.{target_field}, "
                f"but {target_id} runs later. References must point backwards."
            ]
        return [
            f"step {step.id}: '{name}' refers to step '{target_id}', which does not exist. "
            f"Earlier steps are: {', '.join(earlier) or 'none'}"
        ]

    if kind == "image_id":
        return [f"step {step.id}: '{name}' must be a literal image_id, not a reference"]

    target_spec = TOOLS.get(earlier[target_id].tool)
    if target_spec is None:
        return []  # the unknown tool is already reported on its own step

    returned = target_spec.returns.get(target_field)
    if returned is None:
        return [
            f"step {step.id}: '{name}' refers to ${target_id}.{target_field}, but "
            f"{target_spec.name} returns: {', '.join(target_spec.returns)}"
        ]
    if not _compatible(kind, returned.kind):
        usable = [f for f, r in target_spec.returns.items() if _compatible(kind, r.kind)]
        return [
            f"step {step.id}: '{name}' needs {kind}, but ${target_id}.{target_field} "
            f"is a {returned.kind}. {target_spec.name} fields usable here: "
            f"{', '.join(usable) or 'none'}"
        ]
    return []


# --- semantics ---------------------------------------------------------------

def _literal_scenes(step: Step, session: Session) -> list[Scene]:
    """Scenes named directly by a step's image_id arguments."""
    spec = TOOLS.get(step.tool)
    if spec is None:
        return []
    scenes = []
    for name, arg in spec.args.items():
        value = step.args.get(name)
        if arg.kind == "image_id" and isinstance(value, str) and session.has(value):
            scenes.append(session.get(value))
    return scenes


def _provenance(step: Step, upstream: dict[str, list[Scene]], session: Session) -> list[Scene]:
    """Every scene a step's geometry derives from: the ones it names directly,
    plus, through geometry references, the ones its inputs derived from.
    Followed transitively so an intermediate local tool cannot launder where
    geometry came from. Text references do not carry geometry, so a question
    phrased from an earlier answer does not drag that answer's image along."""
    spec = TOOLS.get(step.tool)
    scenes = _literal_scenes(step, session)
    for name, value in step.args.items():
        arg = spec.args.get(name) if spec else None
        if arg is None or arg.kind not in LIST_KINDS:
            continue
        ref = parse_ref(value)
        if ref is not None and ref[0] in upstream:
            scenes.extend(upstream[ref[0]])
    seen: set[str] = set()
    return [s for s in scenes if not (s.image_id in seen or seen.add(s.image_id))]


def _check_semantics(step: Step, derived_from: list[Scene], session: Session) -> list[str]:
    """Cross-argument checks.

    Every argument can be individually valid while the combination is nonsense:
    two images of different places, a change comparison running backwards in
    time, or an optical/SAR pair the wrong way round. None of those fail at
    runtime — they produce confident, wrong answers.

    The rules are keyed on tool names here rather than attached to the ToolSpec
    because they need the plan's provenance graph, which tools.py does not know
    about.
    """
    if step.tool == "change_detect":
        return _check_change_detect(step, session)
    if step.tool == "cross_modal":
        return _check_cross_modal(step, session)
    if step.tool == "filter_by_region":
        return _check_same_area(
            step, derived_from, session,
            "filter_by_region compares geometry, so both inputs must come from the same place.",
        )
    return []


def _check_same_area(step: Step, scenes: list[Scene], session: Session, why: str) -> list[str]:
    areas = session.areas()
    by_area: dict[str, list[str]] = {}
    for scene in scenes:
        by_area.setdefault(areas[scene.image_id], []).append(scene.image_id)
    if len(by_area) <= 1:
        return []
    parts = "; ".join(f"{label}: {', '.join(ids)}" for label, ids in sorted(by_area.items()))
    return [
        f"step {step.id}: {step.tool} is combining inputs that cover different areas "
        f"({parts}). {why}"
    ]


def _resolve_pair(step: Step, session: Session, first: str, second: str):
    a, b = step.args.get(first), step.args.get(second)
    if not (isinstance(a, str) and isinstance(b, str)):
        return None
    if not (session.has(a) and session.has(b)):
        return None  # already reported as an unknown image
    return session.get(a), session.get(b)


def _check_change_detect(step: Step, session: Session) -> list[str]:
    pair = _resolve_pair(step, session, "image_id_t1", "image_id_t2")
    if pair is None:
        return []
    t1, t2 = pair

    if t1.image_id == t2.image_id:
        return [f"step {step.id}: change_detect needs two different images, both are {t1.image_id}"]

    errors = _check_same_area(
        step, [t1, t2], session, "change_detect needs two images of the same place."
    )
    if t1.acquired > t2.acquired:
        errors.append(
            f"step {step.id}: image_id_t1 must be the earlier image, but {t1.image_id} "
            f"({t1.acquired.isoformat()}) is later than {t2.image_id} "
            f"({t2.acquired.isoformat()}). Swap them."
        )
    elif t1.acquired == t2.acquired:
        errors.append(
            f"step {step.id}: {t1.image_id} and {t2.image_id} were both acquired on "
            f"{t1.acquired.isoformat()}; there is no time span to compare."
        )
    return errors


def _check_cross_modal(step: Step, session: Session) -> list[str]:
    pair = _resolve_pair(step, session, "optical_image_id", "sar_image_id")
    if pair is None:
        return []
    optical, sar = pair

    errors = []
    if optical.modality != "optical":
        errors.append(
            f"step {step.id}: optical_image_id must name an optical image, but "
            f"{optical.image_id} is {optical.modality}"
        )
    if sar.modality != "sar":
        errors.append(
            f"step {step.id}: sar_image_id must name a SAR image, but {sar.image_id} "
            f"is {sar.modality}"
        )
    errors.extend(_check_same_area(
        step, [optical, sar], session, "cross_modal needs a co-registered pair."
    ))
    return errors
