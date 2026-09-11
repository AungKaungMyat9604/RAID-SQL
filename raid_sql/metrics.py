"""Cost and latency telemetry: estimates + measurements."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


DIN_SQL_PAPER_COST = 0.50
DIN_SQL_PAPER_LATENCY_SEC = 60.0


def estimate_tokens(text: str) -> int:
    """Rough char-based token estimate (~4 chars/token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_per_m: float,
    output_per_m: float,
) -> float:
    return (input_tokens / 1_000_000.0) * input_per_m + (
        output_tokens / 1_000_000.0
    ) * output_per_m


@dataclass
class StageMetrics:
    stage: str
    model: str = ""
    prompt_chars: int = 0
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    est_cost_usd: float = 0.0
    est_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    retries: int = 0
    ok: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExampleMetrics:
    index: int
    db_id: str
    question: str
    predicted_class: str = ""
    predicted_sql: str = ""
    gold_sql: str = ""
    stages: list[StageMetrics] = field(default_factory=list)
    n_llm_calls: int = 0
    n_sqlite_execs: int = 0
    n_repairs: int = 0
    n_sc_samples: int = 0
    exec_ok: bool = False
    ex_match: Optional[bool] = None
    total_est_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    total_est_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    run_cost_usd_so_far: float = 0.0

    def finalize(self, run_cost_so_far: float) -> None:
        self.total_est_cost_usd = sum(s.est_cost_usd for s in self.stages)
        self.total_cost_usd = sum(s.cost_usd for s in self.stages)
        self.total_est_latency_ms = sum(s.est_latency_ms for s in self.stages)
        self.total_latency_ms = sum(s.latency_ms for s in self.stages)
        self.n_llm_calls = sum(
            1
            for s in self.stages
            if s.stage
            in {
                "schema_link",
                "classify",
                "generate",
                "text_debug",
                "repair",
                "sc_sample",
            }
        )
        self.n_sqlite_execs = sum(1 for s in self.stages if s.stage == "sqlite")
        self.n_repairs = sum(1 for s in self.stages if s.stage == "repair")
        self.n_sc_samples = sum(1 for s in self.stages if s.stage == "sc_sample")
        self.run_cost_usd_so_far = run_cost_so_far + self.total_cost_usd

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExampleMetrics":
        stages = [
            StageMetrics(**s) if isinstance(s, dict) else s
            for s in (data.get("stages") or [])
        ]
        return cls(
            index=int(data.get("index", 0)),
            db_id=str(data.get("db_id", "")),
            question=str(data.get("question", "")),
            predicted_class=str(data.get("predicted_class", "")),
            predicted_sql=str(data.get("predicted_sql", "")),
            gold_sql=str(data.get("gold_sql", "")),
            stages=stages,
            n_llm_calls=int(data.get("n_llm_calls", 0)),
            n_sqlite_execs=int(data.get("n_sqlite_execs", 0)),
            n_repairs=int(data.get("n_repairs", 0)),
            n_sc_samples=int(data.get("n_sc_samples", 0)),
            exec_ok=bool(data.get("exec_ok", False)),
            ex_match=data.get("ex_match"),
            total_est_cost_usd=float(data.get("total_est_cost_usd", 0.0)),
            total_cost_usd=float(data.get("total_cost_usd", 0.0)),
            total_est_latency_ms=float(data.get("total_est_latency_ms", 0.0)),
            total_latency_ms=float(data.get("total_latency_ms", 0.0)),
            run_cost_usd_so_far=float(data.get("run_cost_usd_so_far", 0.0)),
        )


class LatencyPrior:
    """Exponential moving average of stage latency for estimates."""

    def __init__(self, alpha: float = 0.2, default_ms: float = 1500.0) -> None:
        self.alpha = alpha
        self.default_ms = default_ms
        self._ema: dict[str, float] = {}

    def get(self, stage: str) -> float:
        return self._ema.get(stage, self.default_ms)

    def update(self, stage: str, latency_ms: float) -> None:
        prev = self._ema.get(stage, self.default_ms)
        self._ema[stage] = self.alpha * latency_ms + (1 - self.alpha) * prev


class RunLedger:
    """Append-only cost/latency ledger + run summary helpers."""

    def __init__(
        self,
        out_dir: Path,
        *,
        model: str,
        din_cost: float = DIN_SQL_PAPER_COST,
        din_latency_sec: float = DIN_SQL_PAPER_LATENCY_SEC,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.din_cost = din_cost
        self.din_latency_sec = din_latency_sec
        self.ledger_path = self.out_dir / "cost_ledger.jsonl"
        self.predictions_path = self.out_dir / "predictions.jsonl"
        self.priors = LatencyPrior()
        self.examples: list[ExampleMetrics] = []
        self.run_cost_usd = 0.0
        self.started_at = time.time()

    def append_stage(self, example_index: int, stage: StageMetrics) -> None:
        row = {
            "ts": time.time(),
            "example_index": example_index,
            **stage.to_dict(),
        }
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if stage.ok and stage.latency_ms > 0:
            self.priors.update(stage.stage, stage.latency_ms)

    def append_example(self, ex: ExampleMetrics) -> None:
        ex.finalize(self.run_cost_usd)
        self.run_cost_usd = ex.run_cost_usd_so_far
        self.examples.append(ex)
        with self.predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")

    def load_examples_from_predictions(self) -> list[ExampleMetrics]:
        """Load all examples from predictions.jsonl (dedupe by index, keep latest)."""
        if not self.predictions_path.exists():
            return []
        by_index: dict[int, ExampleMetrics] = {}
        for line in self.predictions_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ex = ExampleMetrics.from_dict(row)
            by_index[ex.index] = ex
        examples = [by_index[i] for i in sorted(by_index)]
        if examples:
            self.run_cost_usd = examples[-1].run_cost_usd_so_far
        return examples

    def reload_examples_from_disk(self) -> None:
        """Replace in-memory examples with the full on-disk predictions set."""
        self.examples = self.load_examples_from_predictions()

    def write_predicted_sql(self) -> Path:
        path = self.out_dir / "predicted_sql.txt"
        lines = [
            (e.predicted_sql or "").replace("\n", " ").strip() for e in self.examples
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def write_summary(self, *, split: str, extra: Optional[dict[str, Any]] = None) -> Path:
        latencies = [e.total_latency_ms for e in self.examples]
        costs = [e.total_cost_usd for e in self.examples]
        ex_flags = [e.ex_match for e in self.examples if e.ex_match is not None]
        n_ex = len(ex_flags)
        n_correct = sum(1 for x in ex_flags if x)

        by_stage: dict[str, list[StageMetrics]] = {}
        for e in self.examples:
            for s in e.stages:
                by_stage.setdefault(s.stage, []).append(s)

        def agg_latency(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
            ordered = sorted(vals)
            p95_i = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
            return {
                "mean": statistics.fmean(vals),
                "p50": statistics.median(vals),
                "p95": ordered[p95_i],
            }

        latency_by_stage = {
            stage: {
                **agg_latency([s.latency_ms for s in items]),
                "count": len(items),
                "sum_cost_usd": sum(s.cost_usd for s in items),
                "sum_tokens_in": sum(s.input_tokens for s in items),
                "sum_tokens_out": sum(s.output_tokens for s in items),
            }
            for stage, items in by_stage.items()
        }
        cost_by_stage = {
            stage: {
                "sum_cost_usd": sum(s.cost_usd for s in items),
                "sum_est_cost_usd": sum(s.est_cost_usd for s in items),
                "mean_cost_usd": statistics.fmean([s.cost_usd for s in items])
                if items
                else 0.0,
                "count": len(items),
            }
            for stage, items in by_stage.items()
        }

        mean_cost = statistics.fmean(costs) if costs else 0.0
        mean_lat_ms = statistics.fmean(latencies) if latencies else 0.0
        mean_lat_sec = mean_lat_ms / 1000.0

        summary: dict[str, Any] = {
            "split": split,
            "model": self.model,
            "n_examples": len(self.examples),
            "ex_correct": n_correct,
            "ex_total_scored": n_ex,
            "execution_accuracy": (n_correct / n_ex) if n_ex else None,
            "sum_cost_usd": sum(costs),
            "mean_cost_usd": mean_cost,
            "mean_est_cost_usd": statistics.fmean([e.total_est_cost_usd for e in self.examples])
            if self.examples
            else 0.0,
            "latency_ms": agg_latency(latencies),
            "mean_latency_sec": mean_lat_sec,
            "mean_llm_calls": statistics.fmean([e.n_llm_calls for e in self.examples])
            if self.examples
            else 0.0,
            "elapsed_wall_sec": time.time() - self.started_at,
            "vs_din_sql_paper": {
                "paper_cost_per_query_usd": self.din_cost,
                "paper_latency_sec": self.din_latency_sec,
                "cost_ratio_raid_over_din": (mean_cost / self.din_cost)
                if self.din_cost
                else None,
                "latency_ratio_raid_over_din": (mean_lat_sec / self.din_latency_sec)
                if self.din_latency_sec
                else None,
                "cost_savings_usd_per_query": self.din_cost - mean_cost,
                "latency_savings_sec_per_query": self.din_latency_sec - mean_lat_sec,
            },
            "latency_by_stage": latency_by_stage,
            "cost_by_stage": cost_by_stage,
        }
        if extra:
            summary.update(extra)

        path = self.out_dir / "summary.json"
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (self.out_dir / "latency_by_stage.json").write_text(
            json.dumps(latency_by_stage, indent=2), encoding="utf-8"
        )
        (self.out_dir / "cost_by_stage.json").write_text(
            json.dumps(cost_by_stage, indent=2), encoding="utf-8"
        )
        return path
