#!/usr/bin/env python3
"""Score predicted_sql.txt against gold using Spider execution accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from raid_sql.spider_data import execution_match, load_spider_split, resolve_db_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Spider EX from SQL files")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--pred", required=True, help="predicted_sql.txt")
    parser.add_argument("--output", default=None, help="optional summary JSON path")
    args = parser.parse_args(argv)

    settings = get_settings()
    data_dir = Path(args.dataset) if args.dataset else settings.spider_dir
    examples = load_spider_split(data_dir, args.split)
    preds = [
        ln.strip()
        for ln in Path(args.pred).read_text(encoding="utf-8").splitlines()
        if ln.strip() or True
    ]
    # keep empty lines aligned
    preds = Path(args.pred).read_text(encoding="utf-8").splitlines()
    n = min(len(preds), len(examples))
    correct = 0
    for i in range(n):
        ex = examples[i]
        gold = ex.get("query") or ex.get("SQL") or ""
        pred = preds[i].strip()
        db_path = resolve_db_path(data_dir, ex["db_id"], split=args.split)
        if execution_match(db_path, pred, gold):
            correct += 1
    summary = {
        "split": args.split,
        "n": n,
        "correct": correct,
        "execution_accuracy": correct / n if n else None,
    }
    print(json.dumps(summary, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
