# RAID-SQL method flow

## RAID-SQL v2 (default)

1. **Slim schema link** — short format anchors + **target DB schema/values only** (no RAG demo wall). Values are **question-matched** when possible (exact DB spellings for WHERE literals).
2. **Classify** — EASY / NON-NESTED / NESTED (+ sub-questions for nested).
3. **Retrieve** — embed question with **Gemini text-embedding-004** (or OpenAI/local); pull top-k Spider-train demos with **hardness + skeleton boost**.
4. **Generate** — **RAG-first**: class-shaped retrieved demos + 2–3 static format anchors (not full DIN banks). Optional flag restores legacy DIN static banks for ablation.
5. **Text debug** — SELECT-order / JOIN / INTERSECT / DISTINCT / threshold rules; soft stop sequences.
6. **Post-normalize** (deterministic, applied in finalisation step):
   - Question-gated aggregate-first, keys-first, DISTINCT, name order, and related rules.
   - Generation may emit a select-order plan then apply the order-plan helper when parseable.
7. **EG repair** — sqlite error/empty + semantic gates (incomplete SQL, INTERSECT, missing JOIN).
8. **Self-consistency** — non-EASY, default k=5; vote by **order-insensitive result bag**, then apply SELECT-order finalize among winners.

## Locked claim outputs

- Fair system: **locked evaluation package**
- Official EX: **aggregate results record** (**87.38%**)
- Claim sheet: **official claim metrics**

## Execution accuracy (EX)

`execution_match()` in the spider data module uses **Spider official EX**: result rows are compared as multisets, and **SELECT column order within a row is ignored**. Use `execution_match_strict()` only for legacy debugging. Rescore an existing run with the **prediction rescorer** component after changing EX logic.

**Exact set match (EM)** is secondary. Run `python scripts/evaluate_em.py outputs/test_raid_v2_values` to score the locked `predicted_sql.txt` / `gold.sql` pair with the vendored Spider evaluator (`third_party/spider_eval/`). EM ignores literal values (Spider default) and writes `em_metrics.json`.

Run the **fail analyser** on the **locked evaluation package** (Spider test split).

Report semantic fail kinds only in write-ups (271 residual fails).

## Token / latency recording

Per LLM/SQLite call: measured cost, latency, and token counts. Summary includes token summary in the **aggregate results record**.

## Model

Flash claim: **gemini-2.5-flash** + **text-embedding-004**.
