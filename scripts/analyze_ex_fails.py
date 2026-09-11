#!/usr/bin/env python3
"""Taxonomy EX fails in predictions.jsonl; optional overlap vs a baseline run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from raid_sql.parse import (
    is_incomplete_sql,
    needs_semantic_set_op_repair,
)
from raid_sql.spider_data import (
    execute_for_ex,
    execution_match,
    execution_match_strict,
    load_spider_split,
    resolve_db_path,
    rows_match_ex,
    rows_match_ex_strict,
)



def _load_preds(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[int(r["index"])] = r
    return rows


def classify_fail(
    *,
    index: int,
    pred_sql: str,
    gold_sql: str,
    question: str,
    db_id: str,
    spider_dir: Path,
    split: str,
) -> tuple[str, str]:
    pred = (pred_sql or "").strip()
    gold = (gold_sql or "").strip()
    try:
        db = resolve_db_path(spider_dir, db_id, split=split)
        ok_p, rp, ep = execute_for_ex(db, pred)
        ok_g, rg, _eg = execute_for_ex(db, gold)
    except Exception as exc:  # noqa: BLE001
        return "exec_setup_error", str(exc)[:120]

    if is_incomplete_sql(pred) or (
        pred.upper() in {"SELECT", "SELECT *"} and len(pred) < 20
    ):
        return "incomplete_sql", ""
    if not ok_p:
        return "exec_error", (ep or "")[:120]
    if not ok_g:
        return "gold_exec_error", ""
    assert rp is not None and rg is not None
    if rows_match_ex(rp, rg) and not rows_match_ex_strict(rp, rg):
        return "column_order", ""
    if needs_semantic_set_op_repair(question, pred):
        return "set_op_miss", ""
    if "INTERSECT" in gold.upper() and "INTERSECT" not in pred.upper():
        return "miss_intersect", ""
    if (
        "NOT IN" in gold.upper()
        and "NOT IN" not in pred.upper()
        and "EXCEPT" not in pred.upper()
    ):
        return "miss_not_in", ""
    if len(rp) > len(rg):
        return "superset_rows", ""
    if len(rp) < len(rg):
        return "subset_rows", ""
    return "wrong_values_or_filter", ""


def _pattern_tags(pred: str, gold: str, question: str) -> list[str]:
    tags: list[str] = []
    pu, gu, q = pred.upper(), gold.upper(), (question or "").lower()
    if re.search(r"SELECT\s+\*", pred, re.I) and "ORDER BY" in pu:
        tags.append("select_star_order")
    gm = re.search(r"SELECT\s+(.*?)\s+FROM", gold, re.I | re.S)
    pm = re.search(r"SELECT\s+(.*?)\s+FROM", pred, re.I | re.S)
    if gm and pm and "GROUP BY" in gu:
        gsel, psel = gm.group(1), pm.group(1)
        if re.match(r"\s*count\s*\(", gsel, re.I) and not re.match(
            r"\s*count\s*\(", psel, re.I
        ):
            tags.append("gold_count_first")
        if not re.match(r"\s*count\s*\(", gsel, re.I) and re.match(
            r"\s*count\s*\(", psel, re.I
        ):
            tags.append("gold_key_first")
    if "COUNT(DISTINCT" in pu and "COUNT(DISTINCT" not in gu:
        tags.append("bad_count_distinct")
    if re.search(r"\bDISTINCT\b", pred, re.I) and not re.search(
        r"\bDISTINCT\b", gold, re.I
    ):
        tags.append("extra_distinct")
    if "NOT IN" in gu and "NOT IN" not in pu:
        tags.append("miss_not_in_tag")
    if "Customer_Interactions" in pred and "customers_and_services" in gold.lower():
        tags.append("wrong_bridge_gov")
    return tags


def analyze(
    pred_path: Path,
    *,
    baseline_path: Optional[Path],
    split: str,
    sample_per_kind: int,
    scoring: str = "official",
) -> dict[str, Any]:
    settings = get_settings()
    spider = settings.spider_dir
    examples = load_spider_split(spider, split)
    rows = _load_preds(pred_path)

    scored: list[tuple[int, dict[str, Any], bool, bool]] = []
    for i in sorted(rows):
        r = rows[i]
        ex = examples[i] if i < len(examples) else {}
        gold = ex.get("query") or ex.get("SQL") or r.get("gold_sql") or ""
        db_id = r.get("db_id") or ex.get("db_id") or ""
        db = resolve_db_path(spider, db_id, split=split)
        pred = (r.get("predicted_sql") or "").strip()
        official_ok = execution_match(db, pred, gold)
        strict_ok = execution_match_strict(db, pred, gold)
        scored.append((i, r, official_ok, strict_ok))

    if scoring == "strict":
        fails = [(i, r, o, s) for i, r, o, s in scored if not s]
        correct = sum(1 for _, _, _, s in scored if s)
    else:
        fails = [(i, r, o, s) for i, r, o, s in scored if not o]
        correct = sum(1 for _, _, o, _ in scored if o)

    kind_counts: Counter[str] = Counter()
    db_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {}
    fail_indices: list[int] = []

    for i, r, official_ok, strict_ok in fails:
        ex = examples[i] if i < len(examples) else {}
        gold = ex.get("query") or ex.get("SQL") or r.get("gold_sql") or ""
        q = r.get("question") or ex.get("question") or ""
        db_id = r.get("db_id") or ex.get("db_id") or ""
        pred = (r.get("predicted_sql") or "").strip()
        cls = (r.get("predicted_class") or "?").strip('"')

        kind, detail = classify_fail(
            index=i,
            pred_sql=pred,
            gold_sql=gold,
            question=q,
            db_id=db_id,
            spider_dir=spider,
            split=split,
        )

        if kind == "column_order" and scoring == "official":
            continue

        fail_indices.append(i)
        kind_counts[kind] += 1
        db_counts[db_id] += 1
        class_counts[cls] += 1
        for t in _pattern_tags(pred, gold, q):
            tag_counts[t] += 1

        bucket = samples.setdefault(kind, [])
        if len(bucket) < sample_per_kind:
            bucket.append(
                {
                    "index": i,
                    "db_id": db_id,
                    "class": cls,
                    "question": q[:160],
                    "predicted_sql": pred[:200],
                    "gold_sql": (gold or "")[:200],
                    "detail": detail,
                }
            )

    strict_correct = sum(1 for _, _, _, s in scored if s)
    official_correct = sum(1 for _, _, o, _ in scored if o)
    n = len(scored)
    residual_fails = len(fail_indices)
    column_order_only_pass = sum(
        1 for _, _, o, s in scored if o and not s
    )

    out: dict[str, Any] = {
        "pred_path": str(pred_path),
        "split": split,
        "scoring": scoring,
        "n": n,
        "correct": correct,
        "fails": residual_fails if scoring == "official" else len(fails),
        "ex": (correct / n) if n else 0.0,
        "official_ex": {
            "correct": official_correct,
            "fails": n - official_correct,
            "rate": official_correct / n if n else 0.0,
        },
        "strict_ex_legacy": {
            "correct": strict_correct,
            "fails": n - strict_correct,
            "rate": round(strict_correct / n, 4) if n else 0.0,
        },
        "column_order_only_excluded": column_order_only_pass,
        "fail_kinds": kind_counts.most_common(),
        "fail_by_db": db_counts.most_common(20),
        "fail_by_class": class_counts.most_common(),
        "pattern_tags": tag_counts.most_common(),
        "fail_indices": fail_indices,
        "samples": samples,
        "can_vs_cannot": {
            "can": [
                "normalize_select_star_order_by (SELECT * … ORDER BY col → SELECT col)",
                "strip_count_from_frequency_sort (question-gated drop COUNT from SELECT)",
                "DNS/API collapse abort + retries",
                "incomplete-SQL / soft debug stops",
            ],
            "cannot_without_gold_or_breaks": [
                "flip/remove reorder_select_for_group_by (offline net −41)",
                "drop all DISTINCT (net −30)",
                "COUNT(DISTINCT)→COUNT(*) always (net −16)",
                "prefer baseline SQL when different (net −7)",
                "auto NOT IN / bridge-table rewrite without zero-break proof",
            ],
        },
    }

    if baseline_path and baseline_path.exists():
        base = _load_preds(baseline_path)
        overlap = sorted(set(rows) & set(base))
        both_ok = both_fail = gain = regress = 0
        gain_ids: list[int] = []
        regress_ids: list[int] = []
        for i in overlap:
            a = bool(base[i].get("ex_match"))
            b = bool(rows[i].get("ex_match"))
            if a and b:
                both_ok += 1
            elif not a and not b:
                both_fail += 1
            elif not a and b:
                gain += 1
                gain_ids.append(i)
            else:
                regress += 1
                regress_ids.append(i)
        out["vs_baseline"] = {
            "baseline_path": str(baseline_path),
            "overlap": len(overlap),
            "both_ok": both_ok,
            "both_fail": both_fail,
            "gain": gain,
            "regress": regress,
            "net": gain - regress,
            "gain_indices_sample": gain_ids[:40],
            "regress_indices_sample": regress_ids[:40],
        }

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze RAID-SQL EX failures")
    parser.add_argument(
        "pred_dir",
        type=Path,
        nargs="?",
        default=ROOT / "outputs" / "test_raid_v2",
        help="Run output dir containing predictions.jsonl",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "outputs" / "test_raid" / "predictions.jsonl",
        help="Baseline predictions.jsonl for gain/regress (empty to skip)",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--sample", type=int, default=5)
    parser.add_argument(
        "--scoring",
        choices=["official", "strict"],
        default="official",
        help="official = Spider EX (271 residual fails); strict = jsonl ex_match",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip overlap comparison",
    )
    args = parser.parse_args(argv)

    pred_path = args.pred_dir
    if pred_path.is_dir():
        out_dir = pred_path
        pred_file = pred_path / "predictions.jsonl"
    else:
        pred_file = pred_path
        out_dir = pred_path.parent

    if not pred_file.exists():
        print(f"Missing {pred_file}", file=sys.stderr)
        return 1

    baseline = None if args.no_baseline else args.baseline
    if baseline is not None and not baseline.exists():
        print(f"Warning: baseline missing {baseline}; skipping overlap", flush=True)
        baseline = None

    print(f"Analyzing {pred_file} …", flush=True)
    report = analyze(
        pred_file,
        baseline_path=baseline,
        split=args.split,
        sample_per_kind=args.sample,
        scoring=args.scoring,
    )
    out_path = out_dir / "fail_analysis.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(
        f"n={report['n']} EX={report['ex']:.4%} fails={report['fails']}",
        flush=True,
    )
    print("fail_kinds:", report["fail_kinds"][:8], flush=True)
    if "vs_baseline" in report:
        vb = report["vs_baseline"]
        print(
            f"vs baseline: gain={vb['gain']} regress={vb['regress']} net={vb['net']}",
            flush=True,
        )
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
