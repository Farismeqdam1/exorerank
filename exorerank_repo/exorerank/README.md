# ExoRerank — LLM Re-Ranking for Exomiser Variant Prioritization

A pipeline that uses general-purpose large language models (LLMs) as a **second-stage
re-ranker** on top of [Exomiser](https://github.com/exomiser/Exomiser) variant
prioritization for rare (Mendelian) disease diagnosis, benchmarked across eight models
and a 2×2 design: phenotype availability (**with / without HPO**) × retrieval strategy
(**RAG / no-RAG**).

> Rename the tool/repo name (`ExoRerank`) to whatever you decide before submission.

---

## Pipeline overview

```
Stage 1  Data setup          1000 Genomes background + ClinVar pathogenic spike-in  →  synthetic VCFs
Stage 2  Exomiser baseline   Exomiser v14.1, run WITH HPO and WITHOUT HPO           →  ranked HTML
Stage 3  AI re-ranking        LLM re-orders the candidate list, in two modes:
                                 • RAG    : + GeneBe (ACMG), InterVar, ClinGen, PubMed
                                 • no-RAG : parametric knowledge only
Stage 4  Benchmarking         rank of the known causative gene  →  Rank@1 / Top-5 / detection
```

Four re-ranking conditions result from the 2×2 design, one script each (see `pipeline/`).

---

## Repository layout

```
exorerank/
├── vcf_generation/
│   └── generate_synthetic_vcfs.py     # PLACEHOLDER — add your Stage-1 VCF builder here
├── pipeline/
│   ├── batch_rerank_rag_hpo.py        # RAG,   with HPO
│   ├── batch_rerank_norag_hpo.py      # no-RAG, with HPO
│   ├── batch_rerank_rag_nohpo.py      # RAG,   no HPO (phenotype-blind)
│   └── batch_rerank_norag_nohpo.py    # no-RAG, no HPO (phenotype-blind)
├── analysis/
│   ├── rank_summary_hpo.py            # build Excel summary (with-HPO runs)
│   ├── rank_summary_nohpo.py          # build Excel summary (no-HPO runs)
│   ├── check_progress.py             # audit per-model completion
│   ├── inspect_errors.py             # group failure reasons
│   ├── cleanup_failed.py             # delete failed JSONs so they retry
│   └── diagnose_keys.py              # verify API keys are discoverable
├── figures/
│   └── make_figures.py               # publication figures (300 dpi)
├── data/                             # (optional) place summary spreadsheets here
├── requirements.txt
├── secrets.toml.template             # copy → secrets.toml, add your keys
├── .gitignore
└── LICENSE                           # Apache 2.0
```

---

## Setup

```bash
# 1. clone your repo, then:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. add API keys
cp secrets.toml.template secrets.toml
#   edit secrets.toml — Groq is required; OpenAI/OpenRouter only for GPT-4.1 / kimi-k2

# 3. verify keys are found
python analysis/diagnose_keys.py
```

Each pipeline script has a **CONFIGURATION** block at the top — set `HTML_DIR`
(your Exomiser HTML output) and `OUTPUT_ROOT` (where JSON results are written).

---

## Running the benchmark

Each script loops over all eight models and writes one folder of per-case JSON
per model. Progress is resumable — re-running skips completed cases.

```bash
# with-HPO
python pipeline/batch_rerank_rag_hpo.py
python pipeline/batch_rerank_norag_hpo.py

# no-HPO (phenotype-blind)
python pipeline/batch_rerank_rag_nohpo.py
python pipeline/batch_rerank_norag_nohpo.py

# run a single model:
python pipeline/batch_rerank_rag_nohpo.py --model "GPT 4.1"
```

Controls while running: create an empty file named `PAUSE` in the output root to
pause; delete it to resume. `Ctrl-C` stops safely (progress is saved).

---

## Analysis & figures

```bash
# per-model completion + failures
python analysis/check_progress.py
python analysis/inspect_errors.py

# retry failures (delete failed JSONs, then re-run the pipeline)
python analysis/cleanup_failed.py --delete

# build Excel summaries (edit the CONDITION switch inside for no-HPO)
python analysis/rank_summary_hpo.py
python analysis/rank_summary_nohpo.py

# publication figures → figures/figures_out/
python figures/make_figures.py
```

---

## Models benchmarked

| Model | Provider | Architecture | Open-weight |
|---|---|---|---|
| llama-3.1-8b-instant | Groq | Dense 8B | Yes |
| Llama-3.3-70B | Groq | Dense 70B | Yes |
| qwen3-32b | Groq | Dense 32B | Yes |
| openai/gpt-oss-120b | Groq | MoE 117B (5.1B active) | Yes |
| openai/gpt-oss-20b | Groq | MoE 21B (3.6B active) | Yes |
| llama-4-scout-17b-16e | Groq | MoE 109B (17B active) | Yes |
| GPT-4.1 | OpenAI | Dense | No |
| kimi-k2-instruct-0905 | OpenRouter | MoE 1T (32B active) | Yes |

---

## External resources used

- **Exomiser** v14.1 — variant prioritization
- **GeneBe** — ACMG classification + gnomAD frequency (RAG)
- **InterVar** — clinical pathogenicity (RAG)
- **ClinGen** Gene–Disease Summary (RAG)
- **NCBI PubMed E-utilities** — gene–phenotype literature (RAG)
- **Human Phenotype Ontology (HPO)** — phenotype terms
- **1000 Genomes Project** + **ClinVar** — synthetic cohort construction

## License

Apache License 2.0 — see [LICENSE](LICENSE). Free for non-commercial and commercial use.
