#!/usr/bin/env python3
"""Classify Spider-test EX failures for Flash-only ceiling analysis."""

from __future__ import annotations

import json
import signal
import sqlite3
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from raid_sql.parse import is_incomplete_sql, needs_semantic_set_op_repair  # noqa: E402
from raid_sql.spider_data import resolve_db_path  # noqa: E402


class QueryTimeout(Exception):
    pass


def _alarm_handler(signum, frame):  # noqa: ANN001, ARG001
    raise QueryTimeout("timeout")


def norm(c):
    return round(c, 6) if isinstance(c, float) else c


def exec_sql(db_path: Path, sql: str, timeout_s: int = 2, limit_rows: int = 2000):
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return False, None, "empty"
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout_s)
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        cur = conn.execute(sql)
        rows = cur.fetchmany(limit_rows)
        more = cur.fetchone() is not None
        return True, [tuple(norm(c) for c in r) for r in rows], ("truncated" if more else None)
    except QueryTimeout:
        return False, None, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)[:160]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        if conn is not None:
            conn.close()


def permute_match(pred_rows, gold_rows) -> bool:
    if pred_rows == gold_rows:
        return True
    if not pred_rows or not gold_rows or len(pred_rows) != len(gold_rows):
        return False
    if len(pred_rows[0]) != len(gold_rows[0]):
        return sorted(tuple(sorted(map(str, r))) for r in pred_rows) == sorted(
            tuple(sorted(map(str, r))) for r in gold_rows
        )
    ncols = len(pred_rows[0])
    if ncols > 4:
        return sorted(tuple(reversed(r)) for r in pred_rows) == sorted(gold_rows)
    gset = sorted(gold_rows)
    for perm in permutations(range(ncols)):
        if sorted(tuple(r[i] for i in perm) for r in pred_rows) == gset:
            return True
    return False


def cell_bag(a, b) -> bool:
    bag = lambda rs: sorted((str(type(c).__name__), str(c)) for r in rs for c in r)
    return bag(a) == bag(b)


def gold_hardness(sql: str) -> str:
    s = f" {(sql or '').upper()} "
    if any(k in s for k in [" INTERSECT ", " UNION ", " EXCEPT ", " IN ", " NOT IN "]):
        return "NESTED"
    if " JOIN " in s:
        return "NON-NESTED"
    return "EASY"


def main() -> int:
    spider = (ROOT.parent / "Few-shot-NL2SQL-with-prompting" / "data").resolve()
    pred_path = ROOT / "outputs" / "test_raid" / "predictions.jsonl"
    summary = json.loads((ROOT / "outputs" / "test_raid" / "summary.json").read_text())

    by_i = {
        int(r["index"]): r
        for r in (json.loads(l) for l in pred_path.read_text().splitlines() if l.strip())
    }
    rows = [by_i[i] for i in sorted(by_i)]
    correct = [r for r in rows if r.get("ex_match")]
    fails = [r for r in rows if not r.get("ex_match")]
    n = len(rows)
    need = int(0.853 * n + 0.999) - len(correct)
    print(f"n={n} EX={len(correct)/n:.4f} fails={len(fails)} need={need}", flush=True)
    print(
        "exec_ok among fails:",
        Counter(bool(r.get("exec_ok")) for r in fails),
        flush=True,
    )

    fail_modes: Counter = Counter()
    fail_by_class: dict = defaultdict(Counter)
    rec, hard_ex = [], []
    db_cache: dict = {}

    for i, r in enumerate(fails):
        if i % 20 == 0:
            print(f"  {i}/{len(fails)}", flush=True)
        cls = (r.get("predicted_class") or "?").strip('"')
        pred = r.get("predicted_sql") or ""
        gold = r.get("gold_sql") or ""
        q = r.get("question") or ""
        mode = "semantic_other"

        if is_incomplete_sql(pred):
            mode = "exec_or_incomplete"
        elif r.get("exec_ok") is False:
            mode = "exec_or_incomplete"
        else:
            try:
                db_id = r["db_id"]
                if db_id not in db_cache:
                    db_cache[db_id] = resolve_db_path(spider, db_id, split="test")
                ok_p, rows_p, e_p = exec_sql(db_cache[db_id], pred)
                ok_g, rows_g, e_g = exec_sql(db_cache[db_id], gold)
                if not ok_p:
                    mode = "exec_or_incomplete"
                elif not ok_g:
                    mode = "gold_exec_fail"
                elif e_p == "truncated" or e_g == "truncated":
                    if needs_semantic_set_op_repair(q, pred):
                        mode = "set_op_miss"
                    elif "NOT IN" in gold.upper() and "NOT IN" not in pred.upper():
                        mode = "not_in_miss"
                    elif " JOIN " in f" {gold.upper()} " and " JOIN " not in f" {pred.upper()} ":
                        mode = "missing_join"
                    else:
                        mode = "semantic_other"
                elif rows_p == rows_g:
                    mode = "eval_false_negative"
                elif permute_match(rows_p, rows_g):
                    mode = "column_order"
                elif cell_bag(rows_p, rows_g):
                    mode = "same_cells_diff_shape"
                elif needs_semantic_set_op_repair(q, pred) or (
                    any(op in gold.upper() for op in ("INTERSECT", "EXCEPT", "UNION"))
                    and not any(op in pred.upper() for op in ("INTERSECT", "EXCEPT", "UNION"))
                ):
                    mode = "set_op_miss"
                elif (
                    "NOT IN" in gold.upper()
                    and "NOT IN" not in pred.upper()
                    and "EXCEPT" not in pred.upper()
                ):
                    mode = "not_in_miss"
                elif " JOIN " in f" {gold.upper()} " and " JOIN " not in f" {pred.upper()} ":
                    mode = "missing_join"
                elif " JOIN " in f" {pred.upper()} " and " JOIN " in f" {gold.upper()} ":
                    mode = "wrong_join_or_filter"
                else:
                    try:
                        if set(rows_p) >= set(rows_g) or set(rows_g) >= set(rows_p):
                            mode = "result_superset_subset"
                    except TypeError:
                        pass
            except Exception:  # noqa: BLE001
                mode = "db_error"

        fail_modes[mode] += 1
        fail_by_class[cls][mode] += 1
        item = {
            "index": r["index"],
            "class": cls,
            "mode": mode,
            "db": r["db_id"],
            "q": q[:140],
            "pred": pred[:180],
            "gold": gold[:180],
        }
        easy = mode in {
            "column_order",
            "same_cells_diff_shape",
            "exec_or_incomplete",
            "set_op_miss",
            "eval_false_negative",
            "not_in_miss",
        }
        if easy and len(rec) < 12:
            rec.append(item)
        if (not easy) and len(hard_ex) < 12:
            hard_ex.append(item)

    easy_flip = sum(
        fail_modes[m]
        for m in [
            "column_order",
            "same_cells_diff_shape",
            "exec_or_incomplete",
            "set_op_miss",
            "eval_false_negative",
            "not_in_miss",
        ]
    )
    medium = (
        fail_modes["missing_join"]
        + fail_modes["result_superset_subset"]
        + fail_modes["wrong_join_or_filter"]
    )
    hard = fail_modes["semantic_other"] + fail_modes["db_error"] + fail_modes["gold_exec_fail"]

    misc = sum(
        1
        for r in fails
        if (r.get("predicted_class") or "").strip('"') != gold_hardness(r.get("gold_sql"))
    )
    class_tot = Counter((r.get("predicted_class") or "?").strip('"') for r in rows)
    class_fail = Counter((r.get("predicted_class") or "?").strip('"') for r in fails)
    db_fail = Counter(r["db_id"] for r in fails)
    db_tot = Counter(r["db_id"] for r in rows)
    gold_ops: Counter = Counter()
    for r in fails:
        g = f" {(r.get('gold_sql') or '').upper()} "
        for op in [
            "INTERSECT",
            "EXCEPT",
            "UNION",
            "NOT IN",
            "GROUP BY",
            "ORDER BY",
            "JOIN",
            "HAVING",
            "LIMIT",
        ]:
            if op in g:
                gold_ops[op] += 1

    catch_cons = int(0.55 * easy_flip + 0.20 * medium + 0.05 * hard)
    catch_opt = int(0.85 * easy_flip + 0.40 * medium + 0.12 * hard)
    stage_cost = {
        k: v.get("sum_cost_usd", 0)
        for k, v in (summary.get("latency_by_stage") or {}).items()
        if isinstance(v, dict)
    }

    out = {
        "n": n,
        "correct": len(correct),
        "fails": len(fails),
        "ex": len(correct) / n,
        "need_flips": need,
        "fail_modes": fail_modes.most_common(),
        "fail_by_class": {c: dict(v) for c, v in fail_by_class.items()},
        "easy_flip": easy_flip,
        "medium": medium,
        "hard": hard,
        "ex_if_all_easy": (len(correct) + easy_flip) / n,
        "ex_conservative": (len(correct) + catch_cons) / n,
        "ex_optimistic": (len(correct) + catch_opt) / n,
        "catch_cons": catch_cons,
        "catch_opt": catch_opt,
        "top_fail_dbs": [(db, db_fail[db], db_tot[db]) for db, _ in db_fail.most_common(12)],
        "gold_ops_in_fails": dict(gold_ops.most_common()),
        "misc_class_in_fails": misc,
        "class_totals": dict(class_tot),
        "class_fails": dict(class_fail),
        "class_acc": {c: 1 - class_fail[c] / class_tot[c] for c in class_tot},
        "summary": {
            "mean_cost": summary.get("mean_cost_usd"),
            "mean_latency_sec": summary.get("mean_latency_sec"),
            "mean_llm_calls": summary.get("mean_llm_calls"),
            "sum_cost": summary.get("sum_cost_usd"),
        },
        "stage_cost": stage_cost,
        "recoverable_examples": rec,
        "hard_examples": hard_ex,
        "exec_ok_false_fails": sum(1 for r in fails if r.get("exec_ok") is False),
    }
    out_path = ROOT / "outputs" / "test_raid" / "fail_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in [
        "need_flips", "easy_flip", "medium", "hard", "ex_if_all_easy",
        "ex_conservative", "ex_optimistic", "fail_modes", "class_acc",
        "misc_class_in_fails", "exec_ok_false_fails",
    ]}, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
