"""Smoke test for the detect_change -> analyse_mask pipeline.

Run:
    .venv/Scripts/python.exe test_change_model.py [sample_name]

Runs both stages on one sample pair and prints the probability map's statistics
followed by the analysis dict. `sample_name` defaults to a LEVIR-CD tile with
substantial real change; pass any filename present in both
vendor/BIT_CD/samples/A and .../samples/B.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np

from change_model import detect_change
from describer import answer_question, describe_change
from mask_analysis import analyse_mask

SAMPLES = Path(__file__).resolve().parent / "vendor" / "BIT_CD" / "samples"
DEFAULT_SAMPLE = "test_2_0000_0000.png"

# Stand-in acquisition dates; the sample PNGs carry no metadata.
BEFORE_DATE = "2017-03-11"
AFTER_DATE = "2019-08-24"

QUESTIONS = (
    "What changed between these two images?",
    "Where are the changes located?",
    "How much of the area changed?",
    "Has the built-up area increased, decreased or remained unchanged?",
    "What is the average rainfall here?",
)


def _wrap(text: str, width: int = 78, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAMPLE
    before = SAMPLES / "A" / name
    after = SAMPLES / "B" / name

    for path in (before, after):
        if not path.is_file():
            print(f"sample not found: {path}", file=sys.stderr)
            return 1

    print(f"before: {before}")
    print(f"after:  {after}")

    # --- stage 1: inference -------------------------------------------------
    result = detect_change(before, after)
    prob = result["probability"]

    print()
    print("--- detect_change ---")
    print(f"model:         {result['model_name']}")
    print(f"original_size: {result['original_size']} (width, height)")
    print(f"inference:     {result['inference_ms']:.1f} ms")
    print(f"shape:   {prob.shape}")
    print(f"dtype:   {prob.dtype}")
    print(f"min:     {prob.min():.6f}")
    print(f"max:     {prob.max():.6f}")
    print(f"mean:    {prob.mean():.6f}")
    print(f"changed: {np.count_nonzero(prob > 0.5) / prob.size * 100:.2f}% of pixels > 0.5")

    # --- stage 2: region analysis -------------------------------------------
    analysis = analyse_mask(prob, after)

    print()
    print("--- analyse_mask ---")
    print(json.dumps(analysis, indent=2))

    if not analysis["georeferenced"]:
        print()
        print("note: samples are plain PNGs with no CRS, so bbox_latlon,")
        print("      centroid_latlon and area_sq_m are omitted by design.")

    # --- stage 3: description -----------------------------------------------
    print()
    print("--- describe_change ---")
    print(_wrap(describe_change(analysis, BEFORE_DATE, AFTER_DATE)))

    print()
    print("(without dates)")
    print(_wrap(describe_change(analysis)))

    # --- stage 4: question answering ----------------------------------------
    print()
    print("--- answer_question ---")
    for question in QUESTIONS:
        response = answer_question(analysis, question)
        status = "supported" if response["supported"] else "UNSUPPORTED"
        print()
        print(f"Q: {question}")
        print(f"   [intent: {response['intent']} | {status}]")
        print(_wrap(response["answer"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
