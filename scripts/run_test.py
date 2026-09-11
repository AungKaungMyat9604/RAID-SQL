#!/usr/bin/env python3
"""Locked Spider-test evaluation (run only after Spider-dev gate)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_eval import main  # noqa: E402


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--split" not in argv:
        argv = ["--split", "test", *argv]
    print(
        "WARNING: Locked Spider-test claim run. Prefer Spider-dev EX ≳ 77–78% first.",
        flush=True,
    )
    raise SystemExit(main(argv))
