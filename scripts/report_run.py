#!/usr/bin/env python3
"""Pretty-print cost/latency tables from a run summary.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report RAID-SQL cost/latency")
    parser.add_argument("summary", type=Path, help="path to summary.json")
    args = parser.parse_args(argv)
    data = json.loads(args.summary.read_text(encoding="utf-8"))

    print(f"split={data.get('split')} model={data.get('model')} n={data.get('n_examples')}")
    ex = data.get("execution_accuracy")
    if ex is not None:
        print(
            f"EX={ex:.4%} ({data.get('ex_correct')}/{data.get('ex_total_scored')})"
        )
    print(
        f"mean_cost=${data.get('mean_cost_usd', 0):.4f} "
        f"sum_cost=${data.get('sum_cost_usd', 0):.4f} "
        f"mean_est_cost=${data.get('mean_est_cost_usd', 0):.4f}"
    )
    lat = data.get("latency_ms") or {}
    print(
        f"latency_ms mean={lat.get('mean', 0):.0f} "
        f"p50={lat.get('p50', 0):.0f} p95={lat.get('p95', 0):.0f} "
        f"({data.get('mean_latency_sec', 0):.2f}s mean)"
    )
    vs = data.get("vs_din_sql_paper") or {}
    print(
        f"vs DIN-SQL paper: cost_ratio={vs.get('cost_ratio_raid_over_din')} "
        f"latency_ratio={vs.get('latency_ratio_raid_over_din')} "
        f"save=${vs.get('cost_savings_usd_per_query')} "
        f"save_s={vs.get('latency_savings_sec_per_query')}"
    )
    print("\nBy stage (cost / latency mean):")
    for stage, row in sorted((data.get("cost_by_stage") or {}).items()):
        lat_row = (data.get("latency_by_stage") or {}).get(stage, {})
        print(
            f"  {stage:12s}  n={row.get('count', 0):4d}  "
            f"sum$={row.get('sum_cost_usd', 0):.4f}  "
            f"mean_lat={lat_row.get('mean', 0):.0f}ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
