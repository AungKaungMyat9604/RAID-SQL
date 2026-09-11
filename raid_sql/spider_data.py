"""Spider dataset helpers and execution-accuracy comparison."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional


def sql_skeleton(sql: str) -> str:
    text = sql or ""
    text = re.sub(r"'([^']|'')*'", "_", text)
    text = re.sub(r'"([^"]|"")*"', "_", text)
    text = re.sub(r"\b\d+(\.\d+)?\b", "_", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_spider_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    if split == "train":
        examples: list[dict[str, Any]] = []
        path = data_dir / "train_spider.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Spider train file: {path}")
        examples.extend(json.loads(path.read_text(encoding="utf-8")))
        others = data_dir / "train_others.json"
        if others.exists():
            examples.extend(json.loads(others.read_text(encoding="utf-8")))
        return examples

    if split == "dev":
        path = data_dir / "dev.json"
    elif split == "test":
        path = data_dir / "test.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    if not path.exists():
        raise FileNotFoundError(f"Missing Spider {split} file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_db_path(data_dir: Path, db_id: str, *, split: str | None = None) -> Path:
    data_dir = Path(data_dir)
    candidates: list[Path] = []
    if split == "test":
        candidates.extend(
            [
                data_dir / "test_database" / db_id / f"{db_id}.sqlite",
                data_dir / "test_database" / db_id / f"{db_id}.db",
            ]
        )
    candidates.extend(
        [
            data_dir / "database" / db_id / f"{db_id}.sqlite",
            data_dir / "database" / db_id / f"{db_id}.db",
            data_dir / "test_database" / db_id / f"{db_id}.sqlite",
            data_dir / "test_database" / db_id / f"{db_id}.db",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"SQLite DB not found for db_id={db_id} (split={split}) under {data_dir}"
    )


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _rows_as_multiset(rows: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Row bag with stable column order (for inspection / fail taxonomy)."""
    return sorted(tuple(_normalize_cell(c) for c in row) for row in rows)


def _rows_for_ex_compare(rows: Iterable[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    """Spider official EX: multiset of rows, column-order insensitive within each row."""
    return sorted(
        tuple(sorted(str(_normalize_cell(c)) for c in row))
        for row in rows
    )


def rows_match_ex(rows_p: list[tuple[Any, ...]], rows_g: list[tuple[Any, ...]]) -> bool:
    return _rows_for_ex_compare(rows_p) == _rows_for_ex_compare(rows_g)


def rows_match_ex_strict(
    rows_p: list[tuple[Any, ...]], rows_g: list[tuple[Any, ...]]
) -> bool:
    """Legacy strict match — column order within a row must match."""
    return rows_p == rows_g
def execute_for_ex(
    db_path: Path,
    sql: str,
    *,
    timeout_seconds: float = 30.0,
) -> tuple[bool, Optional[list[tuple[Any, ...]]], Optional[str]]:
    sql = (sql or "").strip().rstrip(";")
    if not sql:
        return False, None, "Empty SQL"
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout_seconds)
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        cur = conn.execute(sql)
        rows = cur.fetchall()
        return True, _rows_as_multiset(rows), None
    except Exception as exc:  # noqa: BLE001
        return False, None, str(exc)
    finally:
        if conn is not None:
            conn.close()


def execution_match(db_path: Path, pred_sql: str, gold_sql: str) -> bool:
    """Spider official execution accuracy (ignores SELECT column order within rows)."""
    ok_p, rows_p, _ = execute_for_ex(db_path, pred_sql)
    ok_g, rows_g, _ = execute_for_ex(db_path, gold_sql)
    if not ok_p or not ok_g or rows_p is None or rows_g is None:
        return False
    return rows_match_ex(rows_p, rows_g)


def execution_match_strict(db_path: Path, pred_sql: str, gold_sql: str) -> bool:
    """Strict tuple match — column order within each result row must agree."""
    ok_p, rows_p, _ = execute_for_ex(db_path, pred_sql)
    ok_g, rows_g, _ = execute_for_ex(db_path, gold_sql)
    if not ok_p or not ok_g or rows_p is None or rows_g is None:
        return False
    return rows_match_ex_strict(rows_p, rows_g)


def write_sql_file(
    path: Path, sqls: list[str], db_ids: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, sql in enumerate(sqls):
        clean = (sql or "").replace("\n", " ").strip().rstrip(";")
        if db_ids is not None:
            lines.append(f"{clean}\t{db_ids[i]}")
        else:
            lines.append(clean)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
