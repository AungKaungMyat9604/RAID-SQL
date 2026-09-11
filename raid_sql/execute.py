"""SQLite execution helpers."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from raid_sql.metrics import StageMetrics


def execute_query(
    db_path: str | Path,
    sql_query: str,
    *,
    max_rows: int = 50,
    timeout_seconds: float = 30.0,
    est_latency_ms: float = 5.0,
) -> tuple[dict[str, Any], StageMetrics]:
    path = Path(db_path)
    metrics = StageMetrics(
        stage="sqlite",
        model="",
        est_latency_ms=est_latency_ms,
        est_cost_usd=0.0,
    )
    t0 = time.perf_counter()
    if not path.exists():
        metrics.ok = False
        metrics.error = f"Database not found: {path}"
        metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "results": None,
            "error": metrics.error,
            "row_count": 0,
            "result_key": None,
            "bag_key": None,
        }, metrics

    sql = (sql_query or "").strip().rstrip(";")
    if not sql:
        metrics.ok = False
        metrics.error = "Empty SQL query"
        metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "results": None,
            "error": metrics.error,
            "row_count": 0,
            "result_key": None,
            "bag_key": None,
        }, metrics

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(path), timeout=timeout_seconds)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout = {int(timeout_seconds * 1000)}")
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        truncated = len(rows) > max_rows
        rows = rows[:max_rows]
        results = [tuple(row) for row in rows]
        # Stable key for voting (sorted multiset of row tuples as strings)
        key = str(sorted(str(r) for r in results))
        # Order-insensitive bag: same cells regardless of SELECT column order
        bag_key = str(
            sorted(tuple(sorted(str(c) for c in r)) for r in results)
        )
        metrics.ok = True
        metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": True,
            "results": results,
            "error": None,
            "row_count": len(results),
            "truncated": truncated,
            "result_key": key,
            "bag_key": bag_key,
        }, metrics
    except Exception as exc:  # noqa: BLE001
        metrics.ok = False
        metrics.error = str(exc)[:500]
        metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "results": None,
            "error": metrics.error,
            "row_count": 0,
            "truncated": False,
            "result_key": None,
            "bag_key": None,
        }, metrics
    finally:
        if conn is not None:
            conn.close()
