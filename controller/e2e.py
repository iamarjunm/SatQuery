"""End-to-end smoke run: real planner LLM (if a key is configured), mock or
real tools, full handle_query() path. Prints one block per query so a human
can see which planner produced the plan and what came back.

Run: .venv/Scripts/python e2e.py [query ...]

Reads controller/.env (gitignored) so keys never live in a shell profile:
    SATQUERY_LLM_BASE_URL=http://<teammate>:11434/v1   # primary: Ollama
    SATQUERY_LLM_FALLBACK_API_KEY=...                  # fallback: Gemini
    # optional: SATQUERY_LLM_MODEL, SATQUERY_LLM_FALLBACK_MODEL, SATQUERY_MOCK_*
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(Path(__file__).with_name(".env"))

from controller import handle_query  # noqa: E402  (after .env so the planner sees the key)
from planner import providers  # noqa: E402
from session import demo_session, manifest_session  # noqa: E402

DEFAULT_QUERIES = [
    "Find buildings constructed after 2023",
    "How many ships are in the harbour?",
    "What changed between the two images?",
    "Describe the land cover in the latest image",
    "Show the flooded areas using the radar image",
]


def main(queries: list[str]) -> int:
    chain = providers()
    key = bool(chain)
    if chain:
        for p in chain:
            print(f"planner : {p.describe()}")
    else:
        print("planner : keyword fallback only (no SATQUERY_LLM_* / SATQUERY_LLM_FALLBACK_* set)")
    print(f"mocks   : SATQUERY_MOCK={os.getenv('SATQUERY_MOCK', '<unset -> on>')}")
    manifest = Path(__file__).with_name("images") / "manifest.json"
    session = (lambda: manifest_session(manifest)) if manifest.is_file() else demo_session
    print(f"images  : {'real chips from images/manifest.json' if manifest.is_file() else 'synthetic demo session'}")
    print(session().describe() + "\n")

    llm_planned = 0
    for query in queries:
        result = handle_query(query, session())
        plan = result.get("plan", {})
        source = plan.get("source")
        llm_planned += source == "llm"
        print("=" * 78)
        print(f"Q: {query}")
        print(f"status={result.get('status')}  plan.source={source}  provider={plan.get('provider')}  "
              f"confidence={result.get('confidence')}  s={result.get('elapsed_s')}")
        for step in plan.get("steps", []):
            print(f"  {step.get('id')}: {step.get('tool')} {json.dumps(step.get('args'))}")
        if plan.get("reasoning"):
            print(f"  reasoning: {plan['reasoning']}")
        print(f"A: {result.get('answer')}")
        for err in result.get("errors", []) or []:
            print(f"  error: {err}")
    print("=" * 78)
    print(f"{llm_planned}/{len(queries)} queries planned by the LLM")
    if key and llm_planned == 0:
        print("Providers are configured but every query fell back to keywords. "
              "The reasoning line of each plan above says why each provider failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or DEFAULT_QUERIES))
