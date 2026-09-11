#!/usr/bin/env python3
"""Convenience wrapper: Spider-dev evaluation."""

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
        argv = ["--split", "dev", *argv]
    raise SystemExit(main(argv))
