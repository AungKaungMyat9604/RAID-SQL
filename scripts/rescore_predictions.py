#!/usr/bin/env python3
"""Recompute ex_match in predictions.jsonl using current execution_match()."""

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
    parser = argparse.ArgumentParser(description="Rescore predictions.jsonl ex_match")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=ROOT / "outputs" / "test_raid_v2_values",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args(argv)

    pred_path = args.output_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"Missing {pred_path}", file=sys.stderr)
        return 1

    settings = get_settings()
    examples = load_spider_split(settings.spider_dir, args.split)

    lines: list[str] = []
    changed = 0
    correct = 0
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        i = int(row["index"])
        ex = examples[i]
        gold = ex.get("query") or ex.get("SQL") or row.get("gold_sql") or ""
        db_id = row.get("db_id") or ex.get("db_id") or ""
        pred = (row.get("predicted_sql") or "").strip()
        db = resolve_db_path(settings.spider_dir, db_id, split=args.split)
        new_match = execution_match(db, pred, gold)
        old_match = bool(row.get("ex_match"))
        if new_match != old_match:
            changed += 1
        row["ex_match"] = new_match
        if new_match:
            correct += 1
        lines.append(json.dumps(row, ensure_ascii=False))

    pred_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    n = len(lines)
    print(
        f"Rescored {n} rows in {pred_path}\n"
        f"  EX={correct}/{n} ({100 * correct / n:.2f}%)\n"
        f"  ex_match flips: {changed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
