# ExoRerank — LLM Re-Ranking for Exomiser Variant Prioritization

A pipeline that uses general-purpose large language models (LLMs) as a **second-stage re-ranker** on top of [Exomiser](https://github.com/exomiser/Exomiser) variant prioritization for rare (Mendelian) disease diagnosis, benchmarked across eight models and a 2×2 design: phenotype availability (**with / without HPO**) × retrieval strategy (**RAG / no-RAG**).

## Pipeline overview

```
Stage 1  Data setup          synthetic VCFs (ClinVar-background mode implemented)
Stage 2  Exomiser baseline   Exomiser v14.1, run WITH HPO and WITHOUT HPO → ranked HTML
Stage 3  AI re-ranking       LLM re-orders the candidate list, with/without RAG
Stage 4  Benchmarking        rank of known causative gene → Rank@1 / Top-5 / detection
```

## Stage 1: synthetic benchmark VCFs

`vcf_generation/generate_synthetic_vcfs.py` is an Exomiser-compatible, deterministic Stage 1 generator based on the repository's earlier VCF generator.

The implemented mode is **ClinVar background**: one ClinVar pathogenic variant is spiked into ClinVar benign/likely-benign background variants (VUS are opt-in). This is a methodology distinction from the original project description, which proposes **1000 Genomes** background. No 1000 Genomes generator exists in the connected repositories, so `--background-source 1000g` fails explicitly; it must not be presented as a 1000 Genomes benchmark until an adapter is implemented and validated.

Each VCF has exactly one selected pathogenic causal variant. Chromosome names have their `chr` prefix removed, and records use `GT:GQ:DP:AD`, preserving the Exomiser assumptions of the earlier generator. The output filename always begins with the causal gene symbol followed by an underscore, e.g. `CFTR_case0001_allele11.vcf`; downstream analysis extracts the target gene from this first token.

```bash
python vcf_generation/generate_synthetic_vcfs.py \
  --build GRCh38 --num-cases 100 --num-variants 50000 \
  --seed 20261005 --output-dir synthetic_vcfs
```

A local ClinVar fixture can be supplied with `--clinvar-vcf` for offline/reproducible testing. Every VCF has a `.metadata.json` sidecar, and `run_metadata.json` records build, source URL/path, catalogue sizes, seed, per-case causal variant, VCF checksum, and the explicit background-source statement. Pin/archive the ClinVar VCF used for a final analysis, because the default URL is ClinVar's weekly release.

Run the smoke tests without downloading ClinVar:

```bash
python -m unittest discover -s vcf_generation/tests -v
```

## Repository layout

```
exorerank/
├── vcf_generation/
│   ├── generate_synthetic_vcfs.py
│   └── tests/                         # fixture-based Stage 1 smoke tests
├── pipeline/
├── analysis/
├── figures/
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running the benchmark

```bash
python pipeline/batch_rerank_rag_hpo.py
python pipeline/batch_rerank_norag_hpo.py
python pipeline/batch_rerank_rag_nohpo.py
python pipeline/batch_rerank_norag_nohpo.py
```

## External resources used

- **Exomiser** v14.1
- **GeneBe**, **InterVar**, **ClinGen**, **PubMed** (RAG)
- **Human Phenotype Ontology**
- **ClinVar** — implemented Stage 1 pathogenic and background source
- **1000 Genomes Project** — proposed background source; not implemented by this repository's Stage 1 generator

## License

Apache License 2.0 — see [LICENSE](LICENSE).
