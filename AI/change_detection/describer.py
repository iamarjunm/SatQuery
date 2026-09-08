"""Template-driven natural language over a mask_analysis result.

No model is involved. Every number in the output is read from the analysis
dict, and the only arithmetic performed is summation and rounding to one
decimal place. If a fact is not in the dict, it does not appear in the text --
which is why, for example, the size-filter threshold is described as "large
enough to report" rather than by its value: `min_region_px` is a parameter of
`analyse_mask`, not part of what it returns.

The graded vocabulary lives in the band tables below so the wording can be
tuned in one place rather than hunted through the sentence templates.
"""

from __future__ import annotations

import re

# change_percentage (percent) -> magnitude word. Upper bounds, exclusive.
MAGNITUDE_BANDS = (
    (1.0, "negligible"),
    (10.0, "limited"),
    (25.0, "moderate"),
    (float("inf"), "extensive"),
)

# mean region confidence (0-1) -> qualifier. Lower bounds, inclusive.
CONFIDENCE_BANDS = (
    (0.85, "high"),
    (0.65, "moderate"),
    (0.0, "low"),
)

# At or below this percentage, `direction` treats the scene as unchanged.
UNCHANGED_THRESHOLD_PCT = 1.0

# How the dominant positions are picked: take positions in descending order of
# their combined area until they account for this share of the changed area,
# and never name more than MAX_POSITIONS of them.
POSITION_COVERAGE = 0.7
MAX_POSITIONS = 3

# Intent patterns, tested in order. `direction` deliberately precedes
# `how_much` so that "how much has it increased" is answered with the
# no-direction caveat rather than a bare area figure.
INTENT_PATTERNS = (
    (
        "direction",
        re.compile(
            r"increas|decreas|unchanged|grow|grew|grown|shrink|shrank|shrunk"
            r"|expand|declin|reduc|gain|loss|lost|more or less|up or down",
            re.IGNORECASE,
        ),
    ),
    (
        "where",
        re.compile(
            r"\bwhere\b|\bwhich (?:area|part|region|side)|\bwhat (?:area|part)"
            r"|\blocation\b|\blocated\b|\bwhereabouts\b|\bwhich portion",
            re.IGNORECASE,
        ),
    ),
    (
        "how_much",
        re.compile(
            r"how much|how many|how big|how large|how extensive|what percent"
            r"|percentage|\bproportion\b|\bhow far\b|\bextent\b|\bcount\b"
            r"|\bnumber of\b|\barea\b|\bsize\b",
            re.IGNORECASE,
        ),
    ),
    (
        "what_changed",
        re.compile(
            r"what (?:has )?chang|what.{0,20}\bchang|describ|summar|overview"
            r"|tell me|what happen|what do you see|analys|analyz",
            re.IGNORECASE,
        ),
    ),
)

UNSUPPORTED_MESSAGE = (
    "I cannot answer that from this analysis. I can describe what changed, "
    "say where the changes are, report how much of the scene changed, and "
    "explain whether the scene changed at all."
)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """"1 region" / "3 regions", without the (s) hedge."""
    if count == 1:
        return f"{count} {singular}"
    return f"{count} {plural or singular + 's'}"


def _join(items: list[str]) -> str:
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _capitalise(text: str) -> str:
    """Uppercase the opening letter without touching the rest.

    str.capitalize() would lowercase everything after it, which would wreck
    compass words and region ids further along the sentence.
    """
    return text[:1].upper() + text[1:] if text else text


def _magnitude(change_percentage: float) -> str:
    for upper_bound, word in MAGNITUDE_BANDS:
        if change_percentage < upper_bound:
            return word
    return MAGNITUDE_BANDS[-1][1]


def _confidence_word(mean_confidence: float) -> str:
    for lower_bound, word in CONFIDENCE_BANDS:
        if mean_confidence >= lower_bound:
            return word
    return CONFIDENCE_BANDS[-1][1]


def _mean_confidence(regions: list[dict]) -> float:
    return sum(region["confidence"] for region in regions) / len(regions)


def _dominant_positions(regions: list[dict]) -> tuple[list[str], int]:
    """Return (positions worth naming, how many distinct positions exist).

    Positions are ranked by the combined area of the regions in them, so one
    large region outweighs several specks -- "where the largest regions are",
    rather than where the most regions are.
    """
    totals: dict[str, int] = {}
    for region in regions:
        totals[region["position"]] = totals.get(region["position"], 0) + region["area_px"]

    total_area = sum(totals.values())
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)

    chosen: list[str] = []
    accumulated = 0
    for position, area in ranked:
        chosen.append(position)
        accumulated += area
        if accumulated / total_area >= POSITION_COVERAGE or len(chosen) >= MAX_POSITIONS:
            break

    return chosen, len(totals)


def _area_phrase(analysis: dict) -> str:
    """Total changed area, in m² when georeferenced and pixels otherwise."""
    regions = analysis["regions"]
    if analysis["georeferenced"]:
        total = round(sum(region["area_sq_m"] for region in regions), 1)
        return f"approximately {total:,.1f} m²"
    total_px = sum(region["area_px"] for region in regions)
    return f"{total_px:,} {'pixel' if total_px == 1 else 'pixels'}"


def _date_prefix(before_date, after_date) -> str:
    """Only stated when both endpoints are known."""
    if before_date is None or after_date is None:
        return ""
    return f"Between {before_date} and {after_date}, "


def _coordinate_phrase(region: dict, georeferenced: bool) -> str:
    """Centroid as stored in the analysis dict -- never re-rounded."""
    if georeferenced:
        longitude, latitude = region["centroid_latlon"]
        return f"lon {longitude}, lat {latitude}"
    x, y = region["centroid"]
    return f"pixel x={x}, y={y}"


def describe_change(analysis: dict, before_date=None, after_date=None) -> str:
    """Describe an analysis result as a single natural-language paragraph.

    Args:
        analysis: the dict returned by `mask_analysis.analyse_mask`.
        before_date: date of the earlier image. Stated only if `after_date`
            is also given.
        after_date: date of the later image.

    Returns one paragraph. Magnitude, position and confidence are rendered as
    graded words rather than raw figures; the percentage is the only number
    quoted directly, at one decimal place.
    """
    change_percentage = analysis["change_percentage"]
    regions = analysis["regions"]
    region_count = analysis["region_count"]
    prefix = _date_prefix(before_date, after_date)
    magnitude = _magnitude(change_percentage)

    # No regions survived filtering. The percentage may still be non-zero:
    # scattered pixels can clear the threshold without forming a region.
    if region_count == 0:
        opening = (
            f"{prefix}the scene shows {magnitude} change: "
            f"{change_percentage:.1f}% of pixels are above the detection threshold"
        )
        if change_percentage > 0:
            return _capitalise(
                f"{opening}, but no regions large enough to report were found. "
                "The flagged pixels are scattered rather than forming coherent areas."
            )
        return _capitalise(f"{opening}, and no distinct change regions were found.")

    sentences = [
        f"{prefix}the scene shows {magnitude} change: "
        f"{change_percentage:.1f}% of pixels changed."
    ]

    positions, distinct_positions = _dominant_positions(regions)
    position_text = _join(positions)

    if region_count == 1:
        sentences.append(
            f"This resolves to a single change region, in the {position_text}."
        )
        sentences.append(f"It covers {_area_phrase(analysis)}.")
    else:
        if distinct_positions == 1:
            placement = f"all of them in the {position_text}"
        else:
            placement = f"mainly in the {position_text}"
        sentences.append(
            f"This resolves to {_plural(region_count, 'distinct change region')}, "
            f"{placement}."
        )
        sentences.append(f"Together they cover {_area_phrase(analysis)}.")

    confidence = _confidence_word(_mean_confidence(regions))
    subject = "this region" if region_count == 1 else "these regions"
    sentences.append(f"Detection confidence across {subject} is {confidence}.")

    # The date prefix supplies its own capital; without it the paragraph
    # would open on "the scene ...".
    return _capitalise(" ".join(sentences))


def _answer_where(analysis: dict) -> str:
    regions = analysis["regions"]
    if not regions:
        return (
            "No change regions were identified, so there is no location to report."
        )

    georeferenced = analysis["georeferenced"]
    positions, distinct_positions = _dominant_positions(regions)

    if distinct_positions == 1:
        opening = f"All change is in the {_join(positions)}."
    else:
        opening = (
            f"By area, change is concentrated in the {_join(positions)}."
        )

    highlighted = regions[:MAX_POSITIONS]
    described = [
        f"{region['id']} in the {region['position']} "
        f"({_plural(region['area_px'], 'pixel')}, centred at "
        f"{_coordinate_phrase(region, georeferenced)})"
        for region in highlighted
    ]

    if len(regions) == 1:
        detail = f"The only region is {described[0]}."
    elif len(regions) <= MAX_POSITIONS:
        detail = f"The individual regions, largest first, are {_join(described)}."
    else:
        detail = (
            f"The {_plural(len(highlighted), 'largest individual region')}, "
            f"in order, {'is' if len(highlighted) == 1 else 'are'} {_join(described)}."
        )

    if not georeferenced:
        detail += (
            " Coordinates are in pixels: the source image carries no CRS, "
            "so no latitude and longitude are available."
        )

    return f"{opening} {detail}"


def _answer_how_much(analysis: dict) -> str:
    change_percentage = analysis["change_percentage"]
    region_count = analysis["region_count"]

    if region_count == 0:
        return (
            f"{change_percentage:.1f}% of pixels are above the detection threshold, "
            "but no regions large enough to report were found."
        )

    across = (
        "a single region"
        if region_count == 1
        else _plural(region_count, "distinct region")
    )
    return (
        f"{change_percentage:.1f}% of the scene changed, across {across} "
        f"covering {_area_phrase(analysis)} in total."
    )


def _answer_direction(analysis: dict) -> str:
    change_percentage = analysis["change_percentage"]
    magnitude = _magnitude(change_percentage)

    if change_percentage <= UNCHANGED_THRESHOLD_PCT:
        verdict = (
            f"The scene is essentially unchanged: only {change_percentage:.1f}% "
            f"of pixels differ, which counts as {magnitude} change."
        )
    else:
        verdict = (
            f"The scene did change: {change_percentage:.1f}% of pixels differ, "
            f"which counts as {magnitude} change."
        )

    return (
        f"{verdict} I cannot say whether that is an increase or a decrease. "
        "This is a binary change-detection model: it reports that a pixel "
        "differs between the two dates, not what it changed from or to. "
        "Answering that would need a semantic model that classifies land cover "
        "in each image and compares the labels."
    )


def answer_question(analysis: dict, question: str) -> dict:
    """Route a question to one of four intents over the analysis dict.

    Returns `{"answer": str, "intent": str, "supported": bool}`. Unrecognised
    questions come back with `supported: False`, intent `"unknown"`, and a
    message listing what can be answered -- never a guess.
    """
    text = question or ""

    for intent, pattern in INTENT_PATTERNS:
        if not pattern.search(text):
            continue

        if intent == "what_changed":
            answer = describe_change(analysis)
        elif intent == "where":
            answer = _answer_where(analysis)
        elif intent == "how_much":
            answer = _answer_how_much(analysis)
        else:
            answer = _answer_direction(analysis)

        return {"answer": answer, "intent": intent, "supported": True}

    return {"answer": UNSUPPORTED_MESSAGE, "intent": "unknown", "supported": False}
