#!/usr/bin/env python3
"""Flash-only SELECT-column-order rewrite on existing predictions.

Goal: recover Spider official EX column-order fails without full regenerate.
Keeps original SQL if rewrite is empty, identical, or fails to execute.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from raid_sql.llm import create_chat_client
from raid_sql.parse import extract_sql
from raid_sql.spider_data import (
    execution_match,
    execute_for_ex,
    load_spider_split,
    resolve_db_path,
    write_sql_file,
)

_SELECT_SPLIT = re.compile(
    r"(?is)^(.*?)\bselect\s+(distinct\s+)?(.+?)\s+from\b(.*)$"
)


def _select_arity(sql: str) -> int:
    m = _SELECT_SPLIT.match((sql or "").strip())
    if not m:
        return 0
    body = m.group(3)
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return len(parts)


def build_prompt(question: str, sql: str) -> str:
    return f"""You fix ONLY SELECT column order for Spider official execution match.

Question:
{question}

Current SQL:
{sql}

Rules:
1) Change ONLY the order of expressions in the SELECT list (and keep DISTINCT if present).
2) Do NOT change tables, JOINs, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, UNION/INTERSECT/EXCEPT, or literals.
3) Metric-first (aggregate then key) when the question leads with how many / number of / count the number / what is the number / average / maximum / minimum, AND the question does NOT say "each".
4) If the question says "each" / "for each", put the GROUP BY key (entity) BEFORE the aggregate.
5) Otherwise follow the order attributes are listed in the question.
6) Return ONLY one executable SQLite SQL statement. No commentary.

SQL:
"""


def _load_preds(path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[int(r["index"])] = r
    return rows


def _norm_sql(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().rstrip(";")).lower()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SELECT-order rewrite pass")
    parser.add_argument(
        "--flash-dir",
        type=Path,
        default=ROOT / "outputs" / "test_raid_v2_values_v3",
        help="Source predictions dir (Flash baseline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "test_raid_v2_values_v3_selorder",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--min-cols",
        type=int,
        default=2,
        help="Only rewrite when SELECT has at least this many columns",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=4,
        help="Only rewrite when SELECT has at most this many columns",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on new rewrites this run (0 = all)",
    )
    args = parser.parse_args(argv)

    settings = get_settings().model_copy(
        update={
            "chat_provider": "gemini",
            "gemini_model": "gemini-2.5-flash",
        }
    )
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_preds = out_dir / "predictions.jsonl"
    out_log = out_dir / "rewrite_log.jsonl"

    flash_path = args.flash_dir / "predictions.jsonl"
    if not flash_path.exists():
        print(f"Missing {flash_path}", file=sys.stderr)
        return 1

    flash = _load_preds(flash_path)
    examples = load_spider_split(settings.spider_dir, args.split)
    if args.resume:
        done = _load_preds(out_preds)
    else:
        done = {}
        for p in (out_preds, out_log):
            if p.exists():
                p.unlink()

    llm = create_chat_client(settings)
    print(
        f"SELECT-order rewrite model={settings.chat_model} "
        f"src={args.flash_dir} out={out_dir} "
        f"cols=[{args.min_cols},{args.max_cols}] "
        f"done={len(done)}/{len(flash)}",
        flush=True,
    )

    # Indices to consider: multi-col SELECT in range; always keep others as copy
    todo: list[int] = []
    skipped_copy = 0
    for i in sorted(flash):
        if i in done:
            continue
        sql = (flash[i].get("predicted_sql") or "").strip()
        n_cols = _select_arity(sql)
        if args.min_cols <= n_cols <= args.max_cols:
            todo.append(i)
        else:
            # copy through unchanged
            row = dict(flash[i])
            row["select_order_rewrote"] = False
            row["select_order_skipped"] = True
            with out_preds.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            done[i] = row
            skipped_copy += 1

    if skipped_copy:
        print(f"Copied {skipped_copy} single/odd-col preds unchanged", flush=True)

    if args.limit and args.limit > 0:
        todo = todo[: args.limit]

    gain = loss = keep = 0
    n_new = 0
    for n_done, i in enumerate(todo, start=1):
        ex = examples[i]
        question = ex["question"]
        db_id = ex["db_id"]
        gold = ex.get("query") or ex.get("SQL") or ""
        base_sql = (flash[i].get("predicted_sql") or "").strip()
        base_ex = bool(flash[i].get("ex_match"))

        print(
            f"[rewrite {n_done}/{len(todo)}] idx={i} db={db_id} base_ex={base_ex}",
            flush=True,
        )
        prompt = build_prompt(question, base_sql)
        try:
            text, metrics = llm.generate(
                prompt,
                stage="select_order_rewrite",
                temperature=0.0,
                max_output_tokens=400,
            )
            new_sql = extract_sql(text).replace("\n", " ").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  LLM error: {exc}", flush=True)
            new_sql = base_sql
            metrics = None

        used = base_sql
        rewrote = False
        reason = "keep_base"
        if new_sql and _norm_sql(new_sql) != _norm_sql(base_sql):
            db = resolve_db_path(settings.spider_dir, db_id, split=args.split)
            ok_exec, _, err = execute_for_ex(db, new_sql)
            if ok_exec:
                used = new_sql
                rewrote = True
                reason = "rewrote"
            else:
                reason = f"exec_fail:{err}"

        try:
            db = resolve_db_path(settings.spider_dir, db_id, split=args.split)
            new_ex = execution_match(db, used, gold)
        except Exception:  # noqa: BLE001
            new_ex = False

        if new_ex and not base_ex:
            gain += 1
        elif base_ex and not new_ex:
            loss += 1
        else:
            keep += 1

        row = dict(flash[i])
        row["predicted_sql"] = used
        row["ex_match"] = new_ex
        row["select_order_rewrote"] = rewrote
        row["select_order_reason"] = reason
        row["select_order_base_sql"] = base_sql
        if metrics is not None:
            row["select_order_cost_usd"] = metrics.cost_usd
            row["total_cost_usd"] = float(row.get("total_cost_usd") or 0) + float(
                metrics.cost_usd or 0
            )

        with out_preds.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with out_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "rewrote": rewrote,
                        "reason": reason,
                        "base_ex": base_ex,
                        "new_ex": new_ex,
                        "cost": None if metrics is None else metrics.cost_usd,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        done[i] = row
        n_new += 1
        print(
            f"  rewrote={rewrote} ex={new_ex} (base={base_ex}) "
            f"running_flips +{gain}/-{loss} net={gain - loss}",
            flush=True,
        )

    # Merge any missing as copies (safety)
    all_rows = []
    for i in sorted(flash):
        if i in done:
            all_rows.append(done[i])
        else:
            all_rows.append(dict(flash[i]))

    # Rewrite predictions file sorted (resume appends out of order possible)
    # Reload from disk for accuracy
    final = _load_preds(out_preds)
    for i in sorted(flash):
        if i not in final:
            final[i] = dict(flash[i])
            final[i]["select_order_rewrote"] = False
    ordered = [final[i] for i in sorted(final)]
    out_preds.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered),
        encoding="utf-8",
    )
    write_sql_file(
        out_dir / "predicted_sql.txt",
        [(r.get("predicted_sql") or "").replace("\n", " ").strip() for r in ordered],
    )

    ok = sum(1 for r in ordered if r.get("ex_match"))
    base_ok = sum(1 for r in flash.values() if r.get("ex_match"))
    n = len(ordered)
    summary = {
        "n": n,
        "execution_accuracy": ok / n if n else 0.0,
        "correct": ok,
        "baseline_dir": str(args.flash_dir),
        "baseline_correct": base_ok,
        "baseline_ex": base_ok / len(flash) if flash else 0.0,
        "delta_correct": ok - base_ok,
        "rewrite_gain": gain,
        "rewrite_loss": loss,
        "rewrite_net": gain - loss,
        "n_rewritten_calls": n_new,
        "beats_din_85_3": (ok / n) >= 0.853 if n else False,
        "note": "Flash SELECT-order rewrite on v3 preds; exec-safe fallback to base",
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
