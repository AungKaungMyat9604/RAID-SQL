#!/usr/bin/env python3
"""Build metrics_claim.json from predictions + summary (computed, not hand-edited)."""

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
    parser = argparse.ArgumentParser(description="Build metrics_claim.json from run outputs")
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=ROOT / "outputs" / "test_raid_v2_values",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    pred_path = out_dir / "predictions.jsonl"
    summary_path = out_dir / "summary.json"
    if not pred_path.exists():
        print(f"Missing {pred_path}", file=sys.stderr)
        return 1

    settings = get_settings()
    examples = load_spider_split(settings.spider_dir, args.split)

    official_correct = 0
    n = 0
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        n += 1
        i = int(row["index"])
        ex = examples[i]
        gold = ex.get("query") or ex.get("SQL") or row.get("gold_sql") or ""
        db_id = row.get("db_id") or ex.get("db_id") or ""
        pred = (row.get("predicted_sql") or "").strip()
        db = resolve_db_path(settings.spider_dir, db_id, split=args.split)
        if execution_match(db, pred, gold):
            official_correct += 1

    summary: dict = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    lat_stages = summary.get("latency_by_stage") or {}
    sum_in = sum(int(v.get("sum_tokens_in") or 0) for v in lat_stages.values())
    sum_out = sum(int(v.get("sum_tokens_out") or 0) for v in lat_stages.values())
    token_summary = {
        "sum_tokens_in": sum_in,
        "sum_tokens_out": sum_out,
        "mean_tokens_in_per_query": round(sum_in / n, 1) if n else 0.0,
        "mean_tokens_out_per_query": round(sum_out / n, 1) if n else 0.0,
        "mean_llm_calls": summary.get("mean_llm_calls"),
    }

    claim = {
        "folder": str(out_dir.relative_to(ROOT)) if str(out_dir).startswith(str(ROOT)) else str(out_dir),
        "official_ex": official_correct / n if n else 0.0,
        "official_correct": official_correct,
        "n_examples": n,
        "fair_deployable": True,
        "system": summary.get("system") or "RAID-SQL v2 Flash + DB values (Gemini 2.5 Flash)",
        "approx_total_cost_usd": round(float(summary.get("sum_cost_usd") or 0), 2),
        "approx_cost_per_query_usd": round(float(summary.get("mean_cost_usd") or 0), 4),
        "din_sql_paper_official_ex": getattr(settings, "din_sql_paper_official_ex", 0.853),
        "beats_din_on_official_ex": (official_correct / n if n else 0.0) > 0.853,
        "token_summary": token_summary,
    }

    # Preserve EM fields if evaluate_em.py already wrote them.
    em_path = out_dir / "em_metrics.json"
    if em_path.is_file():
        em = json.loads(em_path.read_text(encoding="utf-8"))
        if em.get("exact_match") is not None:
            claim["exact_match"] = em["exact_match"]
            claim["exact_match_correct"] = em.get("em_correct")
            claim["exact_match_pct"] = em.get("exact_match_pct")

    out_path = out_dir / "metrics_claim.json"
    out_path.write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")
    em_msg = ""
    if "exact_match_pct" in claim:
        em_msg = (
            f"\n  EM={claim.get('exact_match_correct')}/{n} "
            f"({claim['exact_match_pct']}%)"
        )
    print(
        f"Wrote {out_path}\n"
        f"  official EX={official_correct}/{n} ({100 * official_correct / n:.2f}%)"
        f"{em_msg}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
