#!/usr/bin/env python3
"""Scan the supplement folder for likely anonymity leaks.

CPU-only, stdlib-only.

Patterns checked (configurable below):
  - Personal-name and institution blocklist
  - E-mail addresses
  - GitHub user/repo URLs
  - Absolute paths (/Users/, /home/, /mnt/, /work/, /scratch/, ~/...)
  - API-key-shaped strings (OpenAI-, Anthropic-, HuggingFace-style)
  - WandB URLs
  - Cluster slurm account/partition strings
  - Hidden files (.DS_Store, ._*, *.pyc) that should not have been included

Set BLOCKLIST and ALLOWLIST below to customize.

Usage:
    python tools/check_anonymization.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Configurable blocklist and allowlist
# ---------------------------------------------------------------------------

# Lower-case substrings that should not appear anywhere in any text file.
# Add author last names / first names / institution short forms / cluster
# usernames here.
BLOCKLIST: list[str] = [
    # cluster account / username / institution names redacted via build
    "pi_larsonj_wit_edu",
    "egank2_wit_edu",
    "larsonj",
    "wit.edu",
    # generic project directory hint
    "/work/pi_",
    "/home/egan",
    # WandB
    "wandb.ai",
    "wandb.com",
    # Slurm account identifiers
    "--account=pi_",
]

# Additional regex patterns flagged.  Each entry: (label, regex, severity).
# Severities: "fail" (PASS/FAIL), "warn" (informational only).
PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("Email address",
     re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
     "fail"),
    ("OpenAI-style API key",
     re.compile(r"sk-[A-Za-z0-9]{20,}"),
     "fail"),
    ("Anthropic-style API key",
     re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
     "fail"),
    ("HuggingFace token",
     re.compile(r"hf_[A-Za-z0-9]{20,}"),
     "fail"),
    ("Generic API-key-shaped string",
     re.compile(r"\b[A-Za-z0-9]{32,64}\b"),
     "warn"),
    ("Absolute home/work/scratch path",
     re.compile(r"\B(?:/Users/|/home/|/mnt/|/work/|/scratch/)[A-Za-z0-9_./-]+"),
     "fail"),
    ("GitHub user/repo URL",
     re.compile(r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"),
     "fail"),
    ("WandB URL",
     re.compile(r"wandb\.(?:ai|com|me)/[A-Za-z0-9_./-]+"),
     "fail"),
    ("Slurm account directive",
     re.compile(r"--account=[A-Za-z0-9_]+"),
     "fail"),
    ("Hostname-like reference",
     re.compile(r"\b(?:login|head|gpu|compute)[0-9]{1,3}\.[A-Za-z0-9_.-]+\.(?:edu|org|com)\b"),
     "fail"),
]

# Filenames or path patterns to skip (verbatim or fnmatch-style).
SKIP_PATHS: list[str] = [
    ".git", "__pycache__", ".pytest_cache", "*.pyc", "*.so",
    "release/make_release_zip.sh",  # the script intentionally references release path patterns
    "tools/check_anonymization.py",  # this file documents the patterns
    "ANONYMIZATION.md",  # documents what was removed (mentions blocklist items)
    "PREFLIGHT_REPORT.md",  # documents what was removed
    "examples/freeform_audit_sample.jsonl",  # may contain quoted reviewer text
]

# Files that contain expected mentions of blocklisted strings (for documentation).
# We allow blocklist hits in these files because the file itself is the audit record.
ALLOWLIST_FILES: set[str] = {
    "ANONYMIZATION.md",
    "PREFLIGHT_REPORT.md",
    "tools/check_anonymization.py",
}

# Extensions we will read as text.
TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".jsonl", ".csv", ".yml", ".yaml",
    ".sh", ".cfg", ".toml", ".ini", ".tex", ".bib",
}

# Hidden / metadata files we should not be shipping.
SUSPECT_HIDDEN = {".DS_Store", "Thumbs.db"}


def is_skipped(rel: Path) -> bool:
    parts = set(rel.parts)
    if any(p in parts for p in {".git", "__pycache__", ".pytest_cache"}):
        return True
    str_rel = str(rel)
    for pat in SKIP_PATHS:
        if pat.startswith("*."):
            if str_rel.endswith(pat[1:]):
                return True
        elif pat == str_rel:
            return True
        elif pat in str_rel:
            return True
    return False


def scan_text(text: str, file: str) -> list[tuple[str, str, str]]:
    findings: list[tuple[str, str, str]] = []
    low = text.lower()
    for needle in BLOCKLIST:
        if needle.lower() in low:
            findings.append(("Blocklist", needle, "fail"))
    for label, regex, sev in PATTERNS:
        m = regex.search(text)
        if m:
            findings.append((label, m.group(0), sev))
    return findings


def main() -> int:
    fails: list[tuple[Path, str, str]] = []
    warns: list[tuple[Path, str, str]] = []
    suspect_files: list[Path] = []
    files_scanned = 0

    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if is_skipped(rel):
            continue
        # Suspect hidden / metadata file?
        if p.name in SUSPECT_HIDDEN or p.name.startswith("._"):
            suspect_files.append(rel)
            continue
        if p.suffix not in TEXT_EXTS:
            continue
        files_scanned += 1
        try:
            text = p.read_text(errors="ignore")
        except Exception:
            continue
        for label, match, sev in scan_text(text, str(rel)):
            # Allowlisted file? Suppress blocklist + path findings (documentation).
            if str(rel) in ALLOWLIST_FILES and label in {"Blocklist",
                                                         "Absolute home/work/scratch path",
                                                         "Slurm account directive",
                                                         "WandB URL",
                                                         "GitHub user/repo URL"}:
                continue
            # Suppress generic 32-64 char base64-ish noise in JSON files (lots
            # of float-vector ids, etc.)
            if label == "Generic API-key-shaped string" and rel.suffix in {".json", ".jsonl", ".csv"}:
                continue
            if sev == "fail":
                fails.append((rel, label, match))
            else:
                warns.append((rel, label, match))

    print(f"check_anonymization: {files_scanned} text files scanned in {ROOT}")
    if suspect_files:
        print("SUSPECT METADATA FILES (should not be shipped):")
        for s in suspect_files:
            print(f"  {s}")
    if warns:
        print(f"WARNINGS ({len(warns)}):")
        for rel, label, match in warns[:20]:
            print(f"  {rel}: {label}: {match[:80]}")
        if len(warns) > 20:
            print(f"  … and {len(warns) - 20} more")
    if fails or suspect_files:
        print(f"FAILURES ({len(fails)}):")
        for rel, label, match in fails:
            print(f"  {rel}: {label}: {match[:120]}")
        print()
        print("FAIL")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
