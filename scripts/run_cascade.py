#!/usr/bin/env python3
"""Flash → stronger-model cascade (Gemini 2.5 Pro or GPT-4).

Routing policies:
  hardness — cascade NON-NESTED + NESTED (Flash keeps EASY). Fair / blind.
  nested   — cascade NESTED only. Fair / blind.
  exec     — cascade when Flash SQL is incomplete or exec_ok=False. Fair / blind.
  fails_hard — cascade gold-EX fails that are NON-NESTED/NESTED (EASY fails stay
               Flash). **Oracle** routing (uses gold ex_match) — report as ceiling,
               not a deployable system claim.

Usage::

  python scripts/run_cascade.py \\
    --flash-dir outputs/test_raid_v2_values \\
    --output outputs/test_raid_v2_values_pro_fails_hard \\
    --policy fails_hard --model gemini-2.5-pro --resume
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_settings
from raid_sql.metrics import RunLedger
from raid_sql.parse import is_incomplete_sql
from raid_sql.pipeline import RAIDPipeline
from raid_sql.schema import load_schema_store
from raid_sql.spider_data import (
    execution_match,
    load_spider_split,
    resolve_db_path,
    write_sql_file,
)


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


def _norm_class(predicted_class: str) -> str:
    c = (predicted_class or "").strip().strip('"').upper()
    if "NON-NESTED" in c:
        return "NON-NESTED"
    if "NESTED" in c:
        return "NESTED"
    if "EASY" in c:
        return "EASY"
    return c or "?"


def _is_easy_class(predicted_class: str) -> bool:
    return _norm_class(predicted_class) == "EASY"


def select_cascade_indices(
    flash: dict[int, dict[str, Any]],
    *,
    policy: str,
) -> list[int]:
    """Select cascade indices. ``fails_hard`` uses gold ex_match (oracle)."""
    policy = (policy or "hardness").lower().strip()
    out: list[int] = []
    for i in sorted(flash):
        r = flash[i]
        sql = (r.get("predicted_sql") or "").strip()
        cls = _norm_class(str(r.get("predicted_class") or ""))
        if policy == "hardness":
            if not _is_easy_class(str(r.get("predicted_class") or "")):
                out.append(i)
        elif policy == "nested":
            if cls == "NESTED":
                out.append(i)
        elif policy == "exec":
            if r.get("exec_ok") is False or is_incomplete_sql(sql):
                out.append(i)
        elif policy == "fails_hard":
            # Oracle: gold EX fail + hard class only (EASY stays on Flash)
            if r.get("ex_match") is False and cls in {"NON-NESTED", "NESTED"}:
                out.append(i)
        else:
            raise ValueError(
                f"Unknown policy={policy!r}; use hardness|nested|exec|fails_hard"
            )
    return out


def _infer_provider(model: str, provider: Optional[str] = None) -> str:
    if provider:
        p = provider.lower().strip()
        if p in {"openai", "gemini", "vertex", "google"}:
            return "gemini" if p in {"vertex", "google"} else p
        raise ValueError(f"Unknown --provider={provider!r}")
    name = (model or "").lower()
    if name.startswith("gpt-") or "gpt-4" in name or name.startswith("o1"):
        return "openai"
    if "gemini" in name or name.startswith("gemma"):
        return "gemini"
    # Default: respect .env CHAT_PROVIDER, else openai for legacy gpt-4 default
    base = (get_settings().chat_provider or "openai").lower().strip()
    return "gemini" if base in {"gemini", "vertex", "google"} else "openai"


def _cascade_settings(*, model: str, provider: Optional[str] = None):
    """Build settings for cascade chat model (OpenAI or Gemini/Vertex)."""
    base = get_settings()
    model = (model or "").strip()
    prov = _infer_provider(model, provider)
    if prov == "openai":
        if not base.openai_api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env for OpenAI cascade")
        model = model or base.openai_model or "gpt-4o"
        update: dict[str, Any] = {
            "chat_provider": "openai",
            "openai_model": model,
        }
        # Classic GPT-4 list prices for telemetry
        if model.startswith("gpt-4") and "gpt-4o" not in model and "turbo" not in model:
            update["openai_input_per_m"] = 30.0
            update["openai_output_per_m"] = 60.0
        return base.model_copy(update=update)

    # Gemini / Vertex (same credentials as Flash runs)
    model = model or base.gemini_model or "gemini-2.5-pro"
    if not base.gemini_api_key and not base.google_cloud_project:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT (+ credentials) for Gemini cascade"
        )
    update = {
        "chat_provider": "gemini",
        "gemini_model": model,
    }
    if "pro" in model.lower():
        update["gemini_pro_input_per_m"] = base.gemini_pro_input_per_m
        update["gemini_pro_output_per_m"] = base.gemini_pro_output_per_m
    return base.model_copy(update=update)


def merge_predictions(
    flash: dict[int, dict[str, Any]],
    cascade: dict[int, dict[str, Any]],
    cascaded_ids: set[int],
    *,
    cascade_label: str = "gpt4",
) -> list[dict[str, Any]]:
    """For cascaded indices prefer cascade row; else Flash. Tag provenance."""
    merged: list[dict[str, Any]] = []
    for i in sorted(set(flash) | set(cascade)):
        if i in cascaded_ids and i in cascade:
            row = dict(cascade[i])
            row["cascade_source"] = cascade_label
            row["cascade_policy_applied"] = True
            if i in flash:
                row["flash_predicted_sql"] = flash[i].get("predicted_sql")
                row["flash_ex_match"] = flash[i].get("ex_match")
                row["flash_total_cost_usd"] = flash[i].get("total_cost_usd")
        elif i in flash:
            row = dict(flash[i])
            row["cascade_source"] = "flash"
            row["cascade_policy_applied"] = False
        else:
            row = dict(cascade[i])
            row["cascade_source"] = cascade_label
            row["cascade_policy_applied"] = True
        merged.append(row)
    return merged


def write_merged_summary(
    out_dir: Path,
    *,
    split: str,
    policy: str,
    model: str,
    flash_path: Path,
    merged: list[dict[str, Any]],
    cascaded_ids: set[int],
    cascade_cost: float,
) -> Path:
    n = len(merged)
    correct = sum(1 for r in merged if r.get("ex_match"))
    flash_keep = [r for r in merged if not r.get("cascade_policy_applied")]
    casc = [r for r in merged if r.get("cascade_policy_applied")]
    flash_correct_on_keep = sum(1 for r in flash_keep if r.get("ex_match"))
    casc_correct = sum(1 for r in casc if r.get("ex_match"))
    flash_cost = sum(
        float(r.get("flash_total_cost_usd") or r.get("total_cost_usd") or 0)
        for r in flash_keep
    )
    flash_cost += sum(float(r.get("flash_total_cost_usd") or 0) for r in casc)
    uses_gold = policy == "fails_hard"
    if uses_gold:
        claim = (
            f"ORACLE ceiling: Flash kept except gold-EX-fail hard queries "
            f"re-run with {model}. Not a fair deployable system claim."
        )
        system = f"RAID-SQL Flash + {model} oracle fails_hard cascade"
    else:
        claim = (
            f"Report as RAID-SQL v2 (Gemini 2.5 Flash) with {model} cascade on "
            f"blind '{policy}' routes — not as Flash-only SOTA."
        )
        system = f"RAID-SQL v2 Flash + {model} cascade (fair blind routing)"

    flips_gain = sum(
        1 for r in casc if r.get("ex_match") and r.get("flash_ex_match") is False
    )
    flips_loss = sum(
        1
        for r in casc
        if (not r.get("ex_match")) and r.get("flash_ex_match") is True
    )

    summary = {
        "split": split,
        "system": system,
        "cascade_policy": policy,
        "cascade_model": model,
        "cascade_uses_gold_ex": uses_gold,
        "flash_dir": str(flash_path),
        "n_examples": n,
        "n_cascaded": len(cascaded_ids),
        "n_flash_only": n - len(cascaded_ids),
        "execution_accuracy": (correct / n) if n else 0.0,
        "ex_correct": correct,
        "ex_total_scored": n,
        "beats_din_85_3": bool(n and (correct / n) > 0.853),
        "cascade_slot_accuracy": (casc_correct / len(casc)) if casc else None,
        "flash_slot_accuracy": (
            (flash_correct_on_keep / len(flash_keep)) if flash_keep else None
        ),
        "cascade_flips_gain_vs_flash": flips_gain,
        "cascade_flips_loss_vs_flash": flips_loss,
        "sum_cascade_cost_usd": cascade_cost,
        "sum_flash_cost_usd_approx": flash_cost,
        "sum_system_cost_usd_approx": flash_cost + cascade_cost,
        "claim_wording": claim,
    }
    path = out_dir / "cascade_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Flash→Pro/GPT cascade (fair policies or oracle fails_hard)"
    )
    parser.add_argument(
        "--flash-dir",
        type=Path,
        default=ROOT / "outputs" / "test_raid_v2_values",
        help="Flash run directory with predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "test_raid_v2_values_pro_fails_hard",
        help="Cascade + merged output directory",
    )
    parser.add_argument(
        "--policy",
        choices=["hardness", "nested", "exec", "fails_hard"],
        default="fails_hard",
        help="fails_hard=oracle gold-EX-fail hard only; others are blind",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Cascade chat model (e.g. gemini-2.5-pro, gpt-4)",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["openai", "gemini", "vertex", "google"],
        help="Force chat provider (default: infer from --model)",
    )
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Max cascade queries")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing cascade preds with Flash (no LLM calls)",
    )
    args = parser.parse_args(argv)

    flash_pred = args.flash_dir / "predictions.jsonl"
    if not flash_pred.exists():
        print(f"Missing Flash predictions: {flash_pred}", file=sys.stderr)
        return 1

    flash = _load_preds(flash_pred)
    to_cascade = select_cascade_indices(flash, policy=args.policy)
    if args.limit is not None:
        to_cascade = to_cascade[: max(0, args.limit)]

    out_dir = args.output
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cascade_pred_path = out_dir / "cascade_predictions.jsonl"

    if args.merge_only or args.resume:
        cascaded_done = _load_preds(cascade_pred_path)
    else:
        if cascade_pred_path.exists():
            print(
                f"{cascade_pred_path} exists — pass --resume to continue, "
                f"or delete it for a fresh cascade",
                file=sys.stderr,
            )
            return 1
        cascaded_done = {}

    oracle = args.policy == "fails_hard"
    print(
        f"Cascade policy={args.policy} model={args.model} flash_n={len(flash)} "
        f"route={len(to_cascade)} "
        f"{'(ORACLE uses gold EX)' if oracle else '(blind / fair)'}",
        flush=True,
    )

    settings = _cascade_settings(model=args.model, provider=args.provider)
    cascade_label = args.model.replace("-", "").replace(".", "")
    cascade_cost = sum(
        float(r.get("total_cost_usd") or 0) for r in cascaded_done.values()
    )

    if not args.merge_only:
        examples = load_spider_split(settings.spider_dir, args.split)
        schema = load_schema_store(settings.spider_dir, args.split)
        ledger = RunLedger(out_dir, model=settings.chat_model)
        ledger.predictions_path = cascade_pred_path
        ledger.ledger_path = out_dir / "cascade_cost_ledger.jsonl"
        if cascaded_done:
            ledger.run_cost_usd = cascade_cost
        pipe = RAIDPipeline(
            settings, schema_store=schema, ledger=ledger, split=args.split
        )

        pending = [i for i in to_cascade if i not in cascaded_done]
        print(
            f"{args.model} pending={len(pending)} already={len(cascaded_done)}",
            flush=True,
        )

        for n_done, i in enumerate(pending, start=1):
            ex = examples[i]
            question = ex["question"]
            db_id = ex["db_id"]
            gold = ex.get("query") or ex.get("SQL") or ""
            print(
                f"[cascade {n_done}/{len(pending)}] idx={i} db={db_id} "
                f"flash_class={flash[i].get('predicted_class')}",
                flush=True,
            )
            result = pipe.run_one(
                index=i, question=question, db_id=db_id, gold_sql=gold
            )
            try:
                db_path = resolve_db_path(settings.spider_dir, db_id, split=args.split)
                result.ex_match = execution_match(
                    db_path, result.predicted_sql, gold
                )
            except Exception as exc:  # noqa: BLE001
                result.ex_match = False
                print(f"  EX error: {exc}", flush=True)
            ledger.append_example(result)
            cascade_cost += float(result.total_cost_usd or 0)
            print(
                f"  {args.model} ex={result.ex_match} "
                f"cost=${result.total_cost_usd:.4f} "
                f"flash_ex={flash[i].get('ex_match')}",
                flush=True,
            )

        cascaded_done = _load_preds(cascade_pred_path)

    cascaded_ids = set(to_cascade)
    available_cascade = {
        i: cascaded_done[i] for i in cascaded_done if i in cascaded_ids
    }
    missing = sorted(cascaded_ids - set(available_cascade))
    if missing:
        print(
            f"Warning: {len(missing)} routed indices lack cascade preds; "
            f"keeping Flash SQL (run with --resume to finish)",
            flush=True,
        )
        cascaded_ids_effective = set(available_cascade)
    else:
        cascaded_ids_effective = cascaded_ids

    merged = merge_predictions(
        flash,
        available_cascade,
        cascaded_ids_effective,
        cascade_label=cascade_label,
    )
    merged_path = out_dir / "predictions.jsonl"
    merged_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged),
        encoding="utf-8",
    )
    write_sql_file(
        out_dir / "predicted_sql.txt",
        [(r.get("predicted_sql") or "").replace("\n", " ").strip() for r in merged],
    )
    summary_path = write_merged_summary(
        out_dir,
        split=args.split,
        policy=args.policy,
        model=args.model,
        flash_path=flash_pred,
        merged=merged,
        cascaded_ids=cascaded_ids_effective,
        cascade_cost=cascade_cost,
    )
    data = json.loads(summary_path.read_text())
    print(
        f"Merged EX={data['execution_accuracy']:.4%} "
        f"({data['ex_correct']}/{data['ex_total_scored']}) "
        f"cascaded={data['n_cascaded']} spend≈${cascade_cost:.2f} "
        f"flips=+{data.get('cascade_flips_gain_vs_flash')}/"
        f"-{data.get('cascade_flips_loss_vs_flash')} "
        f"beats_din={data.get('beats_din_85_3')}",
        flush=True,
    )
    print(f"Wrote {merged_path} and {summary_path}", flush=True)
    print(data["claim_wording"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
