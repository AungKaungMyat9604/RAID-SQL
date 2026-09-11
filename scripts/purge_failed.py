#!/usr/bin/env python3
"""Remove failed/empty API predictions so --resume can redo them."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def is_failed_prediction(row: dict) -> bool:
    """True if prediction should be redone (empty SQL, fences, or DNS/API collapse)."""
    sql = (row.get("predicted_sql") or "").strip().rstrip(";")
    if not sql or sql.upper() == "SELECT" or len(sql) < 15:
        return True
    # GPT-4o often wrapped SQL in markdown fences; those runs are invalid.
    if "```" in sql or "`sql" in sql.lower():
        return True
    # Unbalanced parens = truncated generation
    if sql.count("(") != sql.count(")"):
        return True
    # API/DNS collapse: stages failed and essentially no spend
    cost = float(row.get("total_cost_usd") or 0.0)
    for s in row.get("stages") or []:
        err = (s.get("error") or "").lower()
        if any(
            x in err
            for x in (
                "nodename nor servname",
                "errno 8",
                "failed to resolve",
                "name or service not known",
                "getaddrinfo",
                "connection reset",
                "connection refused",
                "network is unreachable",
            )
        ):
            return True
    # Zero-cost + non-executable usually means every LLM call failed instantly
    if cost <= 0.0 and row.get("exec_ok") is False:
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Purge failed RAID-SQL predictions for resume"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=ROOT / "outputs" / "test_raid",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print counts; do not rewrite files",
    )
    args = parser.parse_args(argv)

    pred_path = args.output_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"Missing {pred_path}", file=sys.stderr)
        return 1

    rows = [
        json.loads(line)
        for line in pred_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # keep latest per index
    by_index: dict[int, dict] = {}
    for row in rows:
        by_index[int(row["index"])] = row

    good = {i: r for i, r in by_index.items() if not is_failed_prediction(r)}
    failed = sorted(set(by_index) - set(good))

    print(f"total unique: {len(by_index)}")
    print(f"keep (good): {len(good)}")
    print(f"purge (failed/empty/API): {len(failed)}")
    if failed:
        print(f"failed indices: {failed[0]}..{failed[-1]} (n={len(failed)})")

    if args.dry_run:
        print("dry-run: no files changed")
        return 0

    bak = pred_path.with_suffix(".jsonl.bak_before_purge")
    shutil.copy2(pred_path, bak)
    print(f"backup → {bak}")

    ordered = [good[i] for i in sorted(good)]
    pred_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered),
        encoding="utf-8",
    )
    print(f"rewrote {pred_path} with {len(ordered)} good rows")
    print(
        "\nNext:\n"
        f"  cd {ROOT}\n"
        "  source .venv/bin/activate\n"
        f"  python scripts/run_test.py --output {args.output_dir} --resume\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
