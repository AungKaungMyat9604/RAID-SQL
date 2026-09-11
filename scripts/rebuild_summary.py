#!/usr/bin/env python3
"""Rebuild summary.json / predicted_sql.txt from predictions.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from raid_sql.metrics import RunLedger
from raid_sql.spider_data import (
    execution_match,
    load_spider_split,
    resolve_db_path,
    write_sql_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild RAID-SQL summary from predictions.jsonl"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=ROOT / "outputs" / "dev_raid",
        help="Run output directory containing predictions.jsonl",
    )
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args(argv)

    settings = get_settings()
    out_dir = args.output_dir
    if not (out_dir / "predictions.jsonl").exists():
        print(f"Missing {out_dir / 'predictions.jsonl'}", file=sys.stderr)
        return 1

    # Prefer the model actually recorded in predictions (not whatever .env says now).
    model_label = settings.chat_model
    try:
        from collections import Counter

        counts: Counter[str] = Counter()
        with (out_dir / "predictions.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                for stage in json.loads(line).get("stages") or []:
                    m = (stage.get("model") or "").strip()
                    if m and m != "text-embedding-004":
                        counts[m] += 1
        if counts:
            model_label = counts.most_common(1)[0][0]
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    ledger = RunLedger(
        out_dir,
        model=model_label,
        din_cost=settings.din_sql_paper_cost_per_query_usd,
        din_latency_sec=settings.din_sql_paper_latency_sec,
    )
    ledger.reload_examples_from_disk()
    write_sql_file(
        out_dir / "predicted_sql.txt",
        [e.predicted_sql or "SELECT" for e in ledger.examples],
    )
    write_sql_file(
        out_dir / "gold.sql",
        [e.gold_sql or "" for e in ledger.examples],
        [e.db_id for e in ledger.examples],
    )
    # Infer flags from last prediction if present in a prior summary
    flags = {}
    old = out_dir / "summary.json"
    if old.exists():
        try:
            flags = json.loads(old.read_text()).get("flags") or {}
        except json.JSONDecodeError:
            flags = {}

    path = ledger.write_summary(split=args.split, extra={"flags": flags} if flags else None)
    data = json.loads(path.read_text())

    # Headline EX: Spider official execution_match (not stale ex_match in jsonl).
    examples = load_spider_split(settings.spider_dir, args.split)
    official_correct = 0
    for e in ledger.examples:
        i = e.index
        ex = examples[i]
        gold = ex.get("query") or ex.get("SQL") or e.gold_sql or ""
        db_id = e.db_id or ex.get("db_id") or ""
        pred = (e.predicted_sql or "").strip()
        db = resolve_db_path(settings.spider_dir, db_id, split=args.split)
        if execution_match(db, pred, gold):
            official_correct += 1
    n = len(ledger.examples)
    data["ex_correct"] = official_correct
    data["ex_total_scored"] = n
    data["execution_accuracy"] = official_correct / n if n else None
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    print(
        f"Rebuilt {path}\n"
        f"  n={data.get('n_examples')}  "
        f"EX={data.get('execution_accuracy')}  "
        f"({data.get('ex_correct')}/{data.get('ex_total_scored')})\n"
        f"  mean_cost=${data.get('mean_cost_usd'):.4f}  "
        f"mean_latency_s={data.get('mean_latency_sec'):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
