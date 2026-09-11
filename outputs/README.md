# RAID-SQL outputs

## Final fair claim (use this)

**Locked evaluation package:** `test_raid_v2_values/` — RAID-SQL v2 Flash + DB values (full API run).

| Metric | Value | Recorded in |
| --- | --- | --- |
| Official EX | **87.38%** (1876/2147) | `metrics_claim.json` / `summary.json` |
| vs DIN-SQL | exceeds published **85.3%** | claim metrics |
| Mean tokens/query | ~11,300 in + ~477 out | `summary.json` |
| Mean LLM calls/query | ~8.85 | `summary.json` |
| Run cost (telemetry) | ~$9.81 (~$0.0046/query) | `summary.json` / cost stages |
| Classifier diagonal | 75.8% | `confusion_matrices.json` / `confusion_matrix.png` |

Regenerate derived artefacts from the locked predictions (offline):

```bash
python scripts/build_metrics_claim.py outputs/test_raid_v2_values
python scripts/analyze_ex_fails.py outputs/test_raid_v2_values --scoring strict --no-baseline
python scripts/compute_confusion_matrices.py
python scripts/plot_confusion_matrix.py
```

Do not overwrite this folder for new experiments; write to a new directory under `outputs/`.
