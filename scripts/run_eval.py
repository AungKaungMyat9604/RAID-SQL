#!/usr/bin/env python3
"""Run RAID-SQL on Spider and score execution accuracy with full telemetry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT, get_settings
from raid_sql.metrics import RunLedger
from raid_sql.pipeline import RAIDPipeline
from raid_sql.schema import load_schema_store
from raid_sql.spider_data import (
    execution_match,
    load_spider_split,
    resolve_db_path,
    write_sql_file,
)


def _load_done_indices(pred_jsonl: Path) -> set[int]:
    done: set[int] = set()
    if not pred_jsonl.exists():
        return done
    for line in pred_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        done.add(int(row["index"]))
    return done


def _is_api_collapse(result) -> bool:
    """True when LLM/network failed and we only have a placeholder SELECT."""
    sql = (result.predicted_sql or "").strip().rstrip(";")
    if sql.upper() not in {"SELECT", "SELECT *"} and len(sql) >= 15:
        return False
    for s in result.stages:
        err = (getattr(s, "error", None) or "").lower()
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
    # Zero-cost placeholder after all stages tried
    if float(getattr(result, "total_cost_usd", 0.0) or 0.0) <= 0.0:
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RAID-SQL Spider evaluation")
    parser.add_argument("--dataset", default=None, help="Spider data dir")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-rag", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--no-sc", action="store_true")
    parser.add_argument("--no-db-values", action="store_true")
    parser.add_argument(
        "--max-consecutive-collapses",
        type=int,
        default=3,
        help="Abort after N consecutive DNS/API collapses (0=never)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.dataset:
        settings.spider_data_dir = args.dataset  # type: ignore[misc]
    if args.no_rag:
        settings.use_rag = False  # type: ignore[misc]
    if args.no_repair:
        settings.use_execution_repair = False  # type: ignore[misc]
    if args.no_sc:
        settings.use_self_consistency = False  # type: ignore[misc]
    if args.no_db_values:
        settings.include_db_values = False  # type: ignore[misc]

    data_dir = settings.spider_dir
    out_dir = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "outputs" / f"{args.split}_raid"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_spider_split(data_dir, args.split)
    if args.offset:
        examples = examples[args.offset :]
    if args.limit is not None:
        examples = examples[: args.limit]

    pred_jsonl = out_dir / "predictions.jsonl"
    done = _load_done_indices(pred_jsonl) if args.resume else set()
    if not args.resume and pred_jsonl.exists():
        pred_jsonl.unlink()
        for name in ("cost_ledger.jsonl", "predicted_sql.txt", "summary.json"):
            p = out_dir / name
            if p.exists():
                p.unlink()

    ledger = RunLedger(
        out_dir,
        model=settings.chat_model,
        din_cost=settings.din_sql_paper_cost_per_query_usd,
        din_latency_sec=settings.din_sql_paper_latency_sec,
    )
    # If resuming, seed cost + priors from existing predictions (full file)
    if args.resume and pred_jsonl.exists():
        prior = ledger.load_examples_from_predictions()
        if prior:
            ledger.run_cost_usd = prior[-1].run_cost_usd_so_far
            print(
                f"Resume: {len(prior)} already done, run_spend=${ledger.run_cost_usd:.4f}",
                flush=True,
            )

    schema_store = load_schema_store(data_dir, args.split)
    pipe = RAIDPipeline(
        settings,
        schema_store=schema_store,
        ledger=ledger,
        split=args.split,
    )

    print(
        f"RAID-SQL split={args.split} provider={settings.chat_provider} "
        f"model={settings.chat_model} embed={settings.embedding_provider}/"
        f"{settings.embedding_model} n={len(examples)} rag={settings.use_rag} "
        f"repair={settings.use_execution_repair} sc={settings.use_self_consistency} "
        f"out={out_dir}",
        flush=True,
    )

    gold_sqls: list[str] = []
    db_ids: list[str] = []
    correct = 0
    scored = 0
    consecutive_collapses = 0

    for i, ex in enumerate(examples):
        abs_index = args.offset + i
        if abs_index in done:
            continue
        question = ex["question"]
        db_id = ex["db_id"]
        gold = ex.get("query") or ex.get("SQL") or ""
        gold_sqls.append(gold)
        db_ids.append(db_id)

        if settings.max_budget_usd is not None and ledger.run_cost_usd >= settings.max_budget_usd:
            print(
                f"Budget stop: spent ${ledger.run_cost_usd:.4f} >= ${settings.max_budget_usd}",
                flush=True,
            )
            break

        print(f"[{i+1}/{len(examples)}] idx={abs_index} db={db_id}", flush=True)
        result = pipe.run_one(
            index=abs_index,
            question=question,
            db_id=db_id,
            gold_sql=gold,
        )

        if _is_api_collapse(result):
            consecutive_collapses += 1
            print(
                f"  API/DNS collapse ({consecutive_collapses}) — "
                f"sql={result.predicted_sql!r} cost=${result.total_cost_usd:.4f}",
                flush=True,
            )
            # Still record so purge_failed can see the error, then abort if persistent
            try:
                db_path = resolve_db_path(data_dir, db_id, split=args.split)
                result.ex_match = execution_match(db_path, result.predicted_sql, gold)
            except Exception:  # noqa: BLE001
                result.ex_match = False
            ledger.append_example(result)
            if (
                args.max_consecutive_collapses > 0
                and consecutive_collapses >= args.max_consecutive_collapses
            ):
                print(
                    f"Abort: {consecutive_collapses} consecutive API/DNS collapses. "
                    f"Fix network, then: python scripts/purge_failed.py {out_dir} "
                    f"&& python scripts/run_test.py --output {out_dir} --resume",
                    flush=True,
                )
                break
            continue
        consecutive_collapses = 0

        try:
            db_path = resolve_db_path(data_dir, db_id, split=args.split)
            match = execution_match(db_path, result.predicted_sql, gold)
            result.ex_match = match
            scored += 1
            if match:
                correct += 1
        except Exception as exc:  # noqa: BLE001
            result.ex_match = False
            print(f"  EX error: {exc}", flush=True)

        ledger.append_example(result)
        print(
            f"  class={result.predicted_class} ex={result.ex_match} "
            f"cost=${result.total_cost_usd:.4f} (est ${result.total_est_cost_usd:.4f}) "
            f"lat={result.total_latency_ms:.0f}ms (est {result.total_est_latency_ms:.0f}ms) "
            f"calls={result.n_llm_calls} run_spend=${ledger.run_cost_usd:.4f}",
            flush=True,
        )
        running = correct / scored if scored else 0.0
        print(f"  running EX={running:.4%} ({correct}/{scored})", flush=True)

    # Always rebuild SQL files + summary from full predictions.jsonl
    # (fixes resume: in-memory ledger only has this session's new examples)
    ledger.reload_examples_from_disk()
    if ledger.examples:
        write_sql_file(
            out_dir / "predicted_sql.txt",
            [e.predicted_sql or "SELECT" for e in ledger.examples],
        )
        write_sql_file(
            out_dir / "gold.sql",
            [e.gold_sql or "" for e in ledger.examples],
            [e.db_id for e in ledger.examples],
        )
    else:
        ledger.write_predicted_sql()

    summary_path = ledger.write_summary(
        split=args.split,
        extra={
            "flags": {
                "use_rag": settings.use_rag,
                "use_execution_repair": settings.use_execution_repair,
                "use_self_consistency": settings.use_self_consistency,
                "include_db_values": settings.include_db_values,
            }
        },
    )
    print(
        f"Wrote {summary_path}  n={len(ledger.examples)} "
        f"EX={summary_path and json.loads(summary_path.read_text()).get('execution_accuracy')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
