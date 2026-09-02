#!/usr/bin/env python3
"""Mirror src/fdlib to the Kaggle Dataset `arifshekh/fdlib`.

A Kaggle kernel cannot clone a private GitHub repo without embedding a credential, so
the library travels as a dataset instead: uploaded here, attached to every notebook,
and put on sys.path by `fdlib_bootstrap`. One library, one definition of
preprocessing, identical on both machines -- which is the whole point of section 4.

Every notebook prints the fdlib version and content hash it actually ran against, so
a results CSV can always be traced back to the exact code that produced it.

Usage:
    python scripts/sync_fdlib.py            # create or version-up
    python scripts/sync_fdlib.py --message "add CORAL"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "fdlib"
SLUG = "arifshekh/fdlib"


def content_hash(pkg: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*.py")):
        h.update(p.relative_to(pkg).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def sh(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", default=None)
    args = ap.parse_args()

    if not SRC.is_dir():
        sys.exit(f"missing {SRC}")

    digest = content_hash(SRC)
    msg = args.message or f"fdlib {digest}"

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / "upload"
        stage.mkdir()
        shutil.copytree(SRC, stage / "fdlib",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (stage / "VERSION.json").write_text(json.dumps(
            {"content_hash": digest, "package": "fdlib"}, indent=2))
        (stage / "dataset-metadata.json").write_text(json.dumps(
            {"title": "fdlib", "id": SLUG, "licenses": [{"name": "CC0-1.0"}]}, indent=2))

        code, out = sh(["kaggle", "datasets", "status", SLUG])
        exists = "ready" in out.lower() or "error" not in out.lower() and code == 0

        if exists:
            code, out = sh(["kaggle", "datasets", "version", "-p", str(stage),
                            "-m", msg, "--dir-mode", "zip"])
        else:
            code, out = sh(["kaggle", "datasets", "create", "-p", str(stage),
                            "--dir-mode", "zip"])
        print(out.strip())
        if code != 0 and "already exists" not in out.lower():
            return 1

    print(f"\nfdlib content hash: {digest}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "fdlib_version.json").write_text(
        json.dumps({"content_hash": digest, "message": msg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
