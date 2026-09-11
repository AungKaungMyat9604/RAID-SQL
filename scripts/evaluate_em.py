#!/usr/bin/env python3
"""Score Spider exact set match (EM) on a locked prediction package.

Uses the official Spider exact-set-match logic vendored under
``third_party/spider_eval/`` (Yu et al., 2018). Values are ignored in EM,
matching the Spider script defaults (DISABLE_VALUE / DISABLE_DISTINCT).

Example (locked Flash package)::

    python scripts/evaluate_em.py outputs/test_raid_v2_values
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPIDER_EVAL = ROOT / "third_party" / "spider_eval"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Spider scripts import ``process_sql`` as a top-level module.
if str(SPIDER_EVAL) not in sys.path:
    sys.path.insert(0, str(SPIDER_EVAL))

from config import get_settings  # noqa: E402


def _normalize_gold_line(line: str) -> tuple[str, str]:
    """Split ``sql\\tdb_id``, repairing rare tabs inside the SQL field."""
    parts = line.split("\t")
    if len(parts) < 2:
        raise ValueError(f"gold line missing tab/db_id: {line[:120]!r}")
    db_id = parts[-1].strip()
    sql = "\t".join(parts[:-1]).replace("\t", " ").strip()
    return sql, db_id


def _load_pairs(
    out_dir: Path,
    *,
    pred_path: Path | None,
    gold_path: Path | None,
) -> list[tuple[str, str, str]]:
    pred_file = pred_path or (out_dir / "predicted_sql.txt")
    gold_file = gold_path or (out_dir / "gold.sql")
    if not pred_file.is_file():
        raise FileNotFoundError(pred_file)
    if not gold_file.is_file():
        raise FileNotFoundError(gold_file)

    preds = pred_file.read_text(encoding="utf-8").splitlines()
    golds = gold_file.read_text(encoding="utf-8").splitlines()
    if len(preds) != len(golds):
        raise ValueError(
            f"length mismatch: pred={len(preds)} gold={len(golds)} "
            f"({pred_file} vs {gold_file})"
        )
    pairs: list[tuple[str, str, str]] = []
    for pred, gold_line in zip(preds, golds):
        g_sql, db_id = _normalize_gold_line(gold_line)
        pairs.append((pred.strip(), g_sql, db_id))
    return pairs


def compute_exact_match(
    pairs: list[tuple[str, str, str]],
    *,
    data_dir: Path,
    split: str,
) -> dict:
    """Return EM summary dict for (pred, gold_sql, db_id) rows."""
    try:
        import nltk  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "EM scoring requires nltk. Install with: pip install nltk"
        ) from exc

    # Ensure tokenizer models exist (quiet).
    try:
        from nltk import word_tokenize  # noqa: F401

        word_tokenize("SELECT 1")
    except LookupError:
        import nltk

        nltk.download("punkt", quiet=True)
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass

    from evaluation import (  # type: ignore
        Evaluator,
        build_foreign_key_map_from_json,
        build_valid_col_units,
        rebuild_sql_col,
        rebuild_sql_val,
    )
    from process_sql import Schema, get_schema, get_sql  # type: ignore

    tables_name = "test_tables.json" if split == "test" else "tables.json"
    tables_path = data_dir / tables_name
    if not tables_path.is_file():
        tables_path = data_dir / "tables.json"
    if not tables_path.is_file():
        raise FileNotFoundError(
            f"Missing Spider tables JSON under {data_dir} "
            f"(expected {tables_name} or tables.json)"
        )

    db_root = data_dir / ("test_database" if split == "test" else "database")
    if not db_root.is_dir():
        # fall back: some checkouts keep all DBs under database/
        db_root = data_dir / "database"
    if not db_root.is_dir():
        raise FileNotFoundError(f"Missing Spider database directory under {data_dir}")

    kmaps = build_foreign_key_map_from_json(str(tables_path))
    evaluator = Evaluator()
    levels = ["easy", "medium", "hard", "extra", "all"]
    scores = {lv: {"count": 0, "exact": 0} for lv in levels}
    gold_parse_errors = 0
    pred_parse_errors = 0

    for pred, gold, db_id in pairs:
        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.is_file():
            raise FileNotFoundError(db_path)
        schema = Schema(get_schema(str(db_path)))
        try:
            g_sql = get_sql(schema, gold)
        except Exception:
            gold_parse_errors += 1
            scores["all"]["count"] += 1
            continue

        hardness = evaluator.eval_hardness(g_sql)
        scores[hardness]["count"] += 1
        scores["all"]["count"] += 1

        try:
            p_sql = get_sql(schema, pred)
        except Exception:
            pred_parse_errors += 1
            continue

        kmap = kmaps[db_id]
        g_valid = build_valid_col_units(g_sql["from"]["table_units"], schema)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid, g_sql, kmap)
        p_valid = build_valid_col_units(p_sql["from"]["table_units"], schema)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid, p_sql, kmap)

        exact = int(evaluator.eval_exact_match(p_sql, g_sql))
        scores[hardness]["exact"] += exact
        scores["all"]["exact"] += exact

    n = scores["all"]["count"]
    em_correct = scores["all"]["exact"]
    rate = (em_correct / n) if n else None
    return {
        "metric": "spider_exact_set_match",
        "split": split,
        "n": n,
        "em_correct": em_correct,
        "exact_match": rate,
        "exact_match_pct": round(100.0 * rate, 2) if rate is not None else None,
        "gold_parse_errors": gold_parse_errors,
        "pred_parse_errors": pred_parse_errors,
        "by_hardness": {
            lv: {
                "count": scores[lv]["count"],
                "exact": scores[lv]["exact"],
                "rate": (
                    scores[lv]["exact"] / scores[lv]["count"]
                    if scores[lv]["count"]
                    else None
                ),
            }
            for lv in levels
        },
        "source": (
            "third_party/spider_eval (taoyds/spider evaluation.py exact set match; "
            "values ignored per Spider EM)"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Spider exact set match (EM) for a prediction package"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=ROOT / "outputs" / "test_raid_v2_values",
        help="run folder with predicted_sql.txt and gold.sql",
    )
    parser.add_argument("--dataset", default=None, help="Spider data root")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--pred", type=Path, default=None, help="override predicted_sql.txt")
    parser.add_argument("--gold", type=Path, default=None, help="override gold.sql")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON path (default: <output_dir>/em_metrics.json)",
    )
    parser.add_argument(
        "--update-claim",
        action="store_true",
        help="also merge EM fields into metrics_claim.json when present",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    data_dir = Path(args.dataset) if args.dataset else settings.spider_dir
    out_dir = args.output_dir
    pairs = _load_pairs(out_dir, pred_path=args.pred, gold_path=args.gold)
    summary = compute_exact_match(pairs, data_dir=data_dir, split=args.split)

    out_path = args.output or (out_dir / "em_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(
        f"Wrote {out_path}\n"
        f"  EM={summary['em_correct']}/{summary['n']} "
        f"({summary['exact_match_pct']}%)",
        file=sys.stderr,
    )

    if args.update_claim:
        claim_path = out_dir / "metrics_claim.json"
        if claim_path.is_file():
            claim = json.loads(claim_path.read_text(encoding="utf-8"))
            claim["exact_match"] = summary["exact_match"]
            claim["exact_match_correct"] = summary["em_correct"]
            claim["exact_match_pct"] = summary["exact_match_pct"]
            claim_path.write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")
            print(f"Updated {claim_path}", file=sys.stderr)
        else:
            print(f"No {claim_path} to update", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
