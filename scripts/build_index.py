#!/usr/bin/env python3
"""Build Chroma few-shot index over Spider train."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from raid_sql.retrieve import FewShotRetriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build RAID-SQL Spider train index")
    parser.add_argument("--dataset", default=None, help="Spider data directory")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild collection")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.dataset:
        settings.spider_data_dir = args.dataset  # type: ignore[misc]
    data_dir = settings.spider_dir
    print(f"Indexing train from {data_dir} → {settings.chroma_dir}")
    n = FewShotRetriever(settings).build_index(data_dir, reset=args.reset)
    print(f"Done. Indexed {n} examples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
