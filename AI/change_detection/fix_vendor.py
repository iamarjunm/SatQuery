#!/usr/bin/env python3
"""Patch vendor/BIT_CD so it runs on modern Python / NumPy / torchvision.

BIT_CD was written against ~2021 dependencies and does not import under a
current stack. Two source-level fixes are needed after a fresh clone:

  1. models/resnet.py imports ``load_state_dict_from_url`` from
     ``torchvision.models.utils``, a private module removed in torchvision
     0.13. The function lives in ``torch.hub``.

  2. Several files use the ``np.str`` / ``np.int`` / ``np.float`` / ``np.bool``
     aliases for Python builtins. These were deprecated in NumPy 1.20 and
     removed in 1.24, so they raise AttributeError on any current NumPy.

Both fixes are idempotent -- running this twice is a no-op -- so it is safe to
call from a provisioning script or re-run after ``git pull``. Nothing else in
the vendor tree is touched; to revert, run ``git checkout .`` inside it.

Usage:
    python fix_vendor.py            # apply fixes
    python fix_vendor.py --check    # report only; exit 1 if fixes are needed
    python fix_vendor.py --vendor path/to/BIT_CD
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DEFAULT_VENDOR = Path(__file__).resolve().parent / "vendor" / "BIT_CD"

# --- Fix 1 -----------------------------------------------------------------
# torchvision.models.utils was removed in torchvision 0.13.
IMPORT_PATTERN = re.compile(
    r"^(\s*)from\s+torchvision\.models\.utils\s+import\s+load_state_dict_from_url[ \t]*$",
    re.MULTILINE,
)
IMPORT_REPLACEMENT = r"\1from torch.hub import load_state_dict_from_url"

# --- Fix 2 -----------------------------------------------------------------
# Aliases removed in NumPy 1.24. The trailing \b keeps sized dtypes such as
# np.int8, np.float32 and np.bool_ untouched -- those are still valid.
ALIASES = {
    "str": "str",
    "int": "int",
    "float": "float",
    "bool": "bool",
    "object": "object",
    "complex": "complex",
    "long": "int",
    "unicode": "str",
}
ALIAS_PATTERN = re.compile(r"\b(?:np|numpy)\.(" + "|".join(ALIASES) + r")\b(?!\s*=)")


def _read(path: Path) -> tuple[str, str]:
    r"""Return (text normalized to \n, the file's dominant line ending).

    The working tree is CRLF on Windows (git core.autocrlf=true), so endings
    are detected on read and restored on write. Without this, every patched
    file would come back as a whole-file diff.
    """
    raw = path.read_bytes()
    crlf = raw.count(b"\r\n")
    lf = raw.count(b"\n") - crlf
    newline = "\r\n" if crlf and crlf >= lf else "\n"
    return raw.decode("utf-8").replace("\r\n", "\n"), newline


def _write(path: Path, text: str, newline: str) -> None:
    with open(path, "w", encoding="utf-8", newline=newline) as fh:
        fh.write(text)


def fix_torchvision_import(vendor: Path) -> list[str]:
    """Rewrite the load_state_dict_from_url import. Returns changed files."""
    changed = []
    for path in sorted(vendor.rglob("*.py")):
        original, newline = _read(path)
        patched = IMPORT_PATTERN.sub(IMPORT_REPLACEMENT, original)
        if patched != original:
            _write(path, patched, newline)
            rel = path.relative_to(vendor)
            changed.append(f"{rel}: torchvision.models.utils -> torch.hub")
    return changed


def fix_numpy_aliases(vendor: Path) -> list[str]:
    """Replace removed NumPy scalar aliases with builtins. Returns changes."""
    changed = []
    for path in sorted(vendor.rglob("*.py")):
        original, newline = _read(path)
        hits: list[str] = []

        def _sub(match: re.Match) -> str:
            alias = match.group(1)
            hits.append(alias)
            return ALIASES[alias]

        patched = ALIAS_PATTERN.sub(_sub, original)
        if patched != original:
            _write(path, patched, newline)
            detail = ", ".join(f"np.{a} -> {ALIASES[a]}" for a in dict.fromkeys(hits))
            changed.append(f"{path.relative_to(vendor)}: {detail}")
    return changed


def scan(vendor: Path) -> list[str]:
    """Report files still needing fixes, without modifying anything."""
    pending = []
    for path in sorted(vendor.rglob("*.py")):
        text, _ = _read(path)
        rel = path.relative_to(vendor)
        if IMPORT_PATTERN.search(text):
            pending.append(f"{rel}: stale torchvision.models.utils import")
        for match in ALIAS_PATTERN.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            pending.append(f"{rel}:{line}: removed alias {match.group(0)}")
    return pending


def check_checkpoint(vendor: Path) -> None:
    """The LEVIR checkpoint is committed to the upstream repo, so a fresh
    clone already has it. Warn rather than fail if it is missing."""
    ckpt = vendor / "checkpoints" / "BIT_LEVIR" / "best_ckpt.pt"
    if ckpt.is_file():
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        print(f"  checkpoint present: {ckpt.relative_to(vendor)} ({size_mb:.0f} MB)")
    else:
        print(f"  WARNING: {ckpt.relative_to(vendor)} is missing.")
        print("  Download it from the link in the vendor README and place it there.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--vendor",
        type=Path,
        default=DEFAULT_VENDOR,
        help="path to the BIT_CD clone (default: vendor/BIT_CD)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report needed fixes without applying them; exit 1 if any",
    )
    args = parser.parse_args()

    vendor = args.vendor.resolve()
    if not vendor.is_dir():
        print(f"error: vendor path not found: {vendor}", file=sys.stderr)
        print("Clone it first:", file=sys.stderr)
        print(
            "  git clone https://github.com/justchenhao/BIT_CD.git vendor/BIT_CD",
            file=sys.stderr,
        )
        return 2

    print(f"vendor: {vendor}")

    if args.check:
        pending = scan(vendor)
        if pending:
            print(f"{len(pending)} fix(es) needed:")
            for item in pending:
                print(f"  {item}")
            return 1
        print("  all fixes already applied")
        check_checkpoint(vendor)
        return 0

    changes = fix_torchvision_import(vendor) + fix_numpy_aliases(vendor)
    if changes:
        print(f"applied {len(changes)} fix(es):")
        for item in changes:
            print(f"  {item}")
    else:
        print("  no changes needed (already patched)")
    check_checkpoint(vendor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
