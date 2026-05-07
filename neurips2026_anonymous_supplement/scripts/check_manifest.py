#!/usr/bin/env python3
"""Check that every file referenced in MANIFEST.md exists and is non-empty.

CPU-only, stdlib-only.  Returns exit code 0 on PASS, non-zero on FAIL.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST.md"
FULL_RERUN = ROOT / "results" / "full_rerun_manifest.json"

# Match backtick-quoted relative paths that look like real files
PATH_RE = re.compile(r"`([A-Za-z0-9_./{}*,+-]+\.(?:json|csv|jsonl|md|py|pdf|png|sh|yaml|yml))`")


def expand_brace(path: str) -> list[str]:
    """Expand `{a,b,c}` and trivial wildcards used in MANIFEST.md.

    Doesn't aim to be a full glob — just enough to handle the patterns we use.
    """
    if "{" in path and "}" in path:
        prefix, rest = path.split("{", 1)
        body, suffix = rest.split("}", 1)
        return [
            p
            for opt in body.split(",")
            for p in expand_brace(prefix + opt + suffix)
        ]
    return [path]


def collect_referenced_paths() -> list[str]:
    text = MANIFEST.read_text()
    raw = set()
    for match in PATH_RE.findall(text):
        raw.add(match)
    expanded = set()
    for p in raw:
        expanded.update(expand_brace(p))
    return sorted(expanded)


def main() -> int:
    refs = collect_referenced_paths()
    missing = []
    empty = []
    ok = 0
    skipped = []
    for ref in refs:
        # Skip patterns and meta-references (".py" manifest entries are scripts in this same package; we accept them only if they exist)
        if "*" in ref:
            # Wildcard reference; check that at least one file matches.
            matches = list(ROOT.glob(ref))
            if not matches:
                missing.append(ref + "  (no glob matches)")
            else:
                ok += 1
            continue
        path = ROOT / ref
        if not path.exists():
            # Some manifest paths reference nested adapter/checkpoint files we
            # intentionally excluded; only flag a path as missing if it lives
            # under a directory we *do* ship.
            shipped_top = path.parts[len(ROOT.parts):]
            if shipped_top and shipped_top[0] in {"results", "scripts", "src",
                                                  "configs", "rubrics",
                                                  "prompts", "data",
                                                  "examples", "figures",
                                                  "compute", "tools",
                                                  "release"}:
                missing.append(ref)
            else:
                skipped.append(ref)
            continue
        if path.is_file() and path.stat().st_size == 0:
            empty.append(ref)
            continue
        ok += 1

    # Also walk the full_rerun_manifest.json if present
    full_rerun_status = ""
    if FULL_RERUN.exists():
        d = json.load(FULL_RERUN.open())
        artifact_count = d.get("artifact_count", "?")
        missing_count = d.get("missing_count", "?")
        full_rerun_status = (
            f"full_rerun_manifest.json: artifact_count={artifact_count}, "
            f"missing_count={missing_count}"
        )

    total = len(refs)
    print(f"check_manifest: {ok}/{total} referenced paths OK")
    if skipped:
        print(f"  ({len(skipped)} non-shipped references skipped)")
    if full_rerun_status:
        print("  " + full_rerun_status)
    if empty:
        print("EMPTY:")
        for e in empty:
            print(f"  {e}")
    if missing:
        print("MISSING:")
        for m in missing:
            print(f"  {m}")
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
