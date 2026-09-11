#!/usr/bin/env python3
"""Compute confusion matrices from locked test_raid_v2_values predictions."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from raid_sql.spider_data import execution_match, load_spider_split, resolve_db_path
DEFAULT_PRED = ROOT / "outputs" / "test_raid_v2_values" / "predictions.jsonl"
DEFAULT_OUT = ROOT / "outputs" / "test_raid_v2_values" / "confusion_matrices.json"


def gold_hardness(sql: str) -> str:
    """DIN-style proxy hardness from gold SQL structure (Spider has no hardness field)."""
    s = f" {(sql or '').upper()} "
    if any(k in s for k in [" INTERSECT ", " UNION ", " EXCEPT ", " IN ", " NOT IN "]):
        return "NESTED"
    if " JOIN " in s:
        return "NON-NESTED"
    return "EASY"


def norm_class(raw: str | None) -> str:
    return (raw or "?").strip().strip('"').upper()


def main() -> int:
    pred_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PRED
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT

    settings = get_settings()
    examples = load_spider_split(settings.spider_dir, "test")

    official_correct = 0
    n = 0

    classes = ["EASY", "NON-NESTED", "NESTED"]
    classifier: dict[str, dict[str, int]] = {p: {g: 0 for g in classes} for p in classes}
    ex_by_predicted: dict[str, dict[str, int]] = {
        c: {"correct": 0, "wrong": 0} for c in classes
    }
    ex_by_gold: dict[str, dict[str, int]] = {c: {"correct": 0, "wrong": 0} for c in classes}

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
        db = resolve_db_path(settings.spider_dir, db_id, split="test")
        if execution_match(db, pred, gold):
            official_correct += 1

        pred_c = norm_class(row.get("predicted_class"))
        gold_c = gold_hardness(row.get("gold_sql", ""))
        if pred_c not in classes:
            pred_c = "NESTED" if "NEST" in pred_c else "EASY"
        classifier[pred_c][gold_c] += 1
        ex_ok = bool(row.get("ex_match"))
        key = "correct" if ex_ok else "wrong"
        ex_by_predicted[pred_c][key] += 1
        ex_by_gold[gold_c][key] += 1

    diag = sum(classifier[c][c] for c in classes)
    result = {
        "source": str(pred_path.relative_to(ROOT)),
        "n_examples": n,
        "official_ex": {
            "correct": official_correct,
            "wrong": n - official_correct,
            "rate": official_correct / n if n else 0.0,
        },
        "ex_in_jsonl": {
            "correct": sum(ex_by_predicted[c]["correct"] for c in classes),
            "wrong": sum(ex_by_predicted[c]["wrong"] for c in classes),
        },
        "difficulty_classifier": {
            "description": "Predicted class (Flash) vs gold proxy hardness from gold SQL",
            "classes": classes,
            "matrix_rows_predicted_cols_gold": classifier,
            "accuracy": round(diag / n, 4),
            "accuracy_pct": round(100 * diag / n, 1),
            "diagonal_correct": diag,
        },
        "ex_by_predicted_class_bag_metric": ex_by_predicted,
        "ex_by_gold_hardness_bag_metric": ex_by_gold,
        "mermaid_note": "Use matrix_rows_predicted_cols_gold for the 3x3 classifier confusion matrix",
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
