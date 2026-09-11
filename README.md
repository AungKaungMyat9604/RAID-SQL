# RAID-SQL

**R**etrieval-**A**ugmented **I**n-context **D**ecomposition for Text-to-SQL.

RAID-SQL extends DIN-style decomposed prompting with structure-aware retrieval of Spider-train demonstrations, database-value grounding, execution-guided (EG) repair, and self-consistency by execution voting. Generation uses **Gemini 2.5 Flash**; retrieval uses **text-embedding-004**.

## Locked claim (do not overwrite casually)

| Metric | Value |
| --- | --- |
| Official Spider-test EX | **87.38%** (1876 / 2147) |
| Spider exact set match (EM) | **62.55%** (1343 / 2147) |
| DIN-SQL published baseline | **85.3%** EX / **60%** EM (GPT-4) |
| Mean tokens / query | ~11,300 input + ~477 output |
| Mean LLM calls / query | ~8.85 |
| Approximate API cost | ~$9.81 total (~$0.0046 / query) |
| Classifier diagonal accuracy | **75.8%** |

Locked package: [`outputs/test_raid_v2_values/`](outputs/test_raid_v2_values/)  
Method notes: [`METHOD_FLOW.md`](METHOD_FLOW.md)

## Pipeline (v2)

```text
Question + Schema
  → Slim schema link (+ DB values) → Classify → Skeleton RAG
  → Generate + text debug → post-normalize
  → EG repair → Self-consistency
  → Final SQL
```

## Repository layout

```text
raid_sql/          # pipeline, prompts, retrieval, metrics, Spider I/O
scripts/           # index build, test/dev runners, EX/EM scoring, fail analysis
third_party/
  spider_eval/     # vendored Spider EM scripts (Yu et al., 2018)
outputs/
  test_raid_v2_values/   # locked fair evaluation package
config.py
requirements.txt
.env.example
```

## Setup

1. **Python 3.9+** and a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Spider data** (not bundled). Place Spider files so `tables.json`, train/dev/test JSON, `database/`, and `test_database/` are available. Either:

- keep the sibling path `../Few-shot-NL2SQL-with-prompting/data` (default in `config.py`), or  
- set `SPIDER_DATA_DIR` in `.env` to your Spider data root.

Spider is a third-party benchmark; obtain it under its own licence terms ([Spider / Yale](https://yale-lily.github.io/spider)).

3. **Credentials** — copy the template and fill secrets locally:

```bash
cp .env.example .env
```

Never commit `.env`, service-account JSON, or API keys.

## Reproduce / inspect the locked package

Offline inspection (no API calls):

```bash
python scripts/report_run.py outputs/test_raid_v2_values/summary.json
python scripts/build_metrics_claim.py outputs/test_raid_v2_values
python scripts/evaluate_em.py outputs/test_raid_v2_values --update-claim
python scripts/analyze_ex_fails.py outputs/test_raid_v2_values --scoring strict --no-baseline
```

`evaluate_em.py` writes `em_metrics.json` (Spider exact set match) and can merge EM into `metrics_claim.json`.

Rebuild the embedding index (requires embedding credentials):

```bash
python scripts/build_index.py
```

Full Spider-test re-run (expensive; not required to verify the locked claim):

```bash
python scripts/run_test.py
```

**Do not overwrite** `outputs/test_raid_v2_values/` if you need the published 87.38% claim package intact. Write new experiments to a different output directory.

## Citation

```bibtex
@software{myat2026raidsql,
  author = {Myat, Aung Kaung},
  title  = {{RAID-SQL}: Retrieval-Augmented Decomposed Text-to-{SQL}},
  year   = {2026},
  url    = {https://github.com/AungKaungMyat9604/RAID-SQL},
  note   = {Locked Spider-test official EX 87.38\% (1876/2147)}
}
```

Also see `CITATION.cff`.

## Licence

Code is released under the [MIT License](LICENSE). Spider data and third-party models remain under their respective licences and terms of use.

## Security

- `.env`, credential JSON, and local `.chroma/` indexes are gitignored.
- Before the first `git push`, run: `git status` and confirm `.env` is **not** staged.
