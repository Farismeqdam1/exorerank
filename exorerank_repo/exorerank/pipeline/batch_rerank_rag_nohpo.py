#!/usr/bin/env python3
"""
Batch AI Re-ranking — RAG, NO-HPO (phenotype-blind) — All 8 Models
===================================================================
Runs the full RAG enrichment pipeline (GeneBe ACMG, InterVar pathogenicity,
ClinGen gene-disease) on Exomiser results that were generated WITHOUT HPO terms,
and gives the LLM NO phenotype information.

Because this is the pure no-HPO condition:
  • HPO terms are NOT extracted and NOT placed in the prompt.
  • PubMed gene–phenotype paper counts are SKIPPED (there are no phenotypes).
  • GeneBe (ACMG + gnomAD freq), InterVar, and ClinGen evidence ARE still used —
    these are variant/gene-level and do not depend on phenotype.

Runs all 8 models sequentially, each writing to its own folder under OUTPUT_ROOT.

API routing:
  • Groq        – llama-3.1-8b-instant, Llama-3.3-70B, qwen3-32b,
                  llama-4-scout-17b-16e, openai/gpt-oss-120b, openai/gpt-oss-20b
  • OpenAI      – GPT 4.1
  • OpenRouter  – kimi-k2-instruct-0905

Controls:
    PAUSE:  touch <OUTPUT_ROOT>/PAUSE      RESUME: rm <OUTPUT_ROOT>/PAUSE
    STOP:   Ctrl+C  (resumes from last saved file next run)
    Single model:  python batch_rerank_rag_nohpo.py --model "GPT 4.1"
"""

import os
import re
import sys
import json
import time
import logging
import argparse
import urllib.parse
from pathlib import Path
from datetime import datetime
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

try:
    import obonet            # noqa: F401  (kept for parity; unused in no-HPO mode)
    import requests
    import pandas as pd
    import genebe as gnb
    from groq import Groq
    import openai
except ImportError as e:
    logger.error(f"Missing package: {e}")
    logger.error("pip install groq openai obonet genebe requests pandas")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION  — adjust the two paths to your no-HPO Exomiser results
# ==============================================================================
HTML_DIR    = "/home/faris/exomiser-cli-14.1.0-distribution_faster/exomiser-cli-14.1.0/nohporesult"
OUTPUT_ROOT = "/home/faris/reranking_rag_nohpo"

MAX_VARIANTS = 50

# Pure no-HPO: LLM receives no phenotype terms; PubMed gene-phenotype skipped.
FORCE_NO_HPO = True

# (display_name, provider, api_model_string, delay_sec, max_variants)
MODELS = [
    ("llama-3.1-8b-instant",  "groq",       "llama-3.1-8b-instant",                       8,  20),
    ("Llama-3.3-70B",         "groq",       "llama-3.3-70b-versatile",                    30, 10),
    ("qwen3-32b",             "groq",       "qwen/qwen3-32b",                             15, 15),
    ("llama-4-scout-17b-16e", "groq",       "meta-llama/llama-4-scout-17b-16e-instruct",  8,  20),
    ("openai/gpt-oss-120b",   "groq",       "openai/gpt-oss-120b",                        8,  20),
    ("openai/gpt-oss-20b",    "groq",       "openai/gpt-oss-20b",                         8,  20),
    ("GPT 4.1",               "openai",     "gpt-4.1",                                    2,  50),
    ("kimi-k2-instruct-0905", "openrouter", "moonshotai/kimi-k2",                         2,  50),
]

# ==============================================================================
# API KEY LOADING  (env → secrets.toml / .env in several locations)
# ==============================================================================
def _load_key_files():
    secrets = {}
    candidates = [
        Path.cwd() / "secrets.toml", Path.cwd() / ".env",
        Path(__file__).parent / "secrets.toml", Path(__file__).parent / ".env",
        Path("/home/faris/.streamlit/secrets.toml"),
        Path("/home/faris/Downloads/DreamJournal/.env"),
        Path("/home/faris/groq-prompt-optimizer/.env"),
        Path.home() / ".streamlit" / "secrets.toml",
        Path.home() / ".env", Path.home() / "secrets.toml",
    ]
    for p in candidates:
        if not p.exists():
            continue
        logger.info(f"  Reading keys from: {p}")
        try:
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    secrets[k.strip().upper()] = v
        except Exception as e:
            logger.warning(f"  Could not read {p}: {e}")
    return secrets


def load_api_keys():
    logger.info("Searching for API keys...")
    fk = _load_key_files()

    def g(name):
        return os.getenv(name) or os.getenv(name.lower()) or fk.get(name.upper()) or ""

    keys = {
        "groq":       g("GROQ_API_KEY"),
        "openai":     g("OPENAI_API_KEY"),
        "openrouter": g("OPENROUTER_API_KEY"),
        "eutils":     g("EUTILS_API_KEY"),   # not needed in no-HPO mode, but loaded if present
    }
    for prov in ("groq", "openai", "openrouter"):
        s = f"✅ found ({keys[prov][:8]}…)" if keys[prov] else "❌ not found — that provider's models will be skipped"
        logger.info(f"  {prov:<12} {s}")
    return keys


API_KEYS = load_api_keys()
if not any(API_KEYS[p] for p in ("groq", "openai", "openrouter")):
    logger.error("No LLM API keys found. Set at least GROQ_API_KEY.")
    sys.exit(1)

session = requests.Session()

# ==============================================================================
# API CLIENTS
# ==============================================================================
_clients = {}

def get_client(provider):
    if provider in _clients:
        return _clients[provider]
    key = API_KEYS.get(provider, "")
    if not key:
        raise RuntimeError(f"No API key for provider '{provider}'.")
    if provider == "groq":
        _clients[provider] = Groq(api_key=key)
    elif provider == "openai":
        _clients[provider] = openai.OpenAI(api_key=key)
    elif provider == "openrouter":
        _clients[provider] = openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    else:
        raise ValueError(provider)
    return _clients[provider]


def call_llm(provider, model_str, messages, max_tokens=3072, retries=6):
    client = get_client(provider)
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model_str, messages=messages,
                temperature=0.3, max_tokens=max_tokens)
            return r.choices[0].message.content
        except Exception as e:
            es = str(e)
            if "429" in es or "rate_limit" in es.lower() or "too many requests" in es.lower():
                m = re.search(r'(\d+(?:\.\d+)?)\s*second', es, re.IGNORECASE)
                wait = float(m.group(1)) + 2 if m else 60.0
                logger.warning(f"  ⚠️ 429 (attempt {attempt+1}/{retries}) — sleeping {wait:.0f}s")
                time.sleep(wait)
            else:
                wait = min(2 ** attempt, 60)
                logger.warning(f"  ⚠️ API error (attempt {attempt+1}/{retries}): {e} — retry in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"All {retries} attempts failed for {provider}/{model_str}")


# ==============================================================================
# HTML PARSING
# ==============================================================================
def extract_variants_with_regex(html, max_variants=None):
    variants = []
    for section in re.split(r'(?i)<b>\s*Variants contributing to score:\s*</b>', html)[1:]:
        stop = re.search(r'(?i)<b>\s*Other passed variants:\s*</b>', section)
        if stop:
            section = section[:stop.start()]
        variants.extend(re.findall(r'\b([XY\d]+-\d+-[ACGT]+-[ACGT\w]+)\b', section))
        if max_variants and len(variants) >= max_variants:
            break
    return variants if max_variants is None else variants[:max_variants]


# ==============================================================================
# ClinGen gene-disease
# ==============================================================================
_gene_disease_df = None

def load_gene_disease_summary():
    global _gene_disease_df
    if _gene_disease_df is None:
        url = 'https://raw.githubusercontent.com/wah644/streamlit_app.py/main/Clingen-Gene-Disease-Summary-2025-01-03.csv'
        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
            import io
            _gene_disease_df = pd.read_csv(io.StringIO(resp.text), dtype=str).fillna("")
            logger.info("ClinGen gene-disease data loaded")
        except Exception as e:
            logger.warning(f"ClinGen load failed: {e}")
            _gene_disease_df = pd.DataFrame()
    return _gene_disease_df


def find_gene_match(gene_symbol, hgnc_id):
    df = load_gene_disease_summary()
    if df.empty or 'GENE SYMBOL' not in df.columns:
        return "No existing gene-disease match found"
    rows = df[(df['GENE SYMBOL'] == gene_symbol) & (df['GENE ID (HGNC)'] == hgnc_id)]
    if rows.empty:
        return "No disease found"
    return dict(zip(rows['DISEASE LABEL'], rows['CLASSIFICATION']))


# ==============================================================================
# Per-variant enrichment (GeneBe + InterVar)
# ==============================================================================
def get_variant_info(variant_str):
    parts = variant_str.split('-')
    if len(parts) == 4:
        chrom, pos, ref, alt = parts
        return True, [chrom, pos, ref, alt, "hg38"]
    return False, []


def safe_get(url, *, headers=None, params=None, retries=3, backoff=2, timeout=10):
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    return None


def process_variant(variant_response, variant_index):
    is_valid, parts = get_variant_info(variant_response)
    vd = {
        "GeneBe_results": ['-', '-', '-', '-', '-', '-', '-', '-'],
        "InterVar_results": ['-', '', '-', ''],
    }
    if is_valid:
        # GeneBe ACMG
        r = safe_get("https://api.genebe.net/cloud/api-public/v1/variant",
                     headers={"Accept": "application/json"},
                     params={"chr": parts[0], "pos": parts[1], "ref": parts[2],
                             "alt": parts[3], "genome": parts[4]})
        if r and r.status_code == 200:
            try:
                v = r.json()["variants"][0]
                g = vd["GeneBe_results"]
                g[0] = v.get("acmg_classification", "Not Available")
                g[1] = v.get("effect", "Not Available")
                g[2] = v.get("gene_symbol", "Not Available")
                g[3] = v.get("gene_hgnc_id", "Not Available")
                g[4] = v.get("dbsnp", "Not Available")
                g[5] = v.get("frequency_reference_population", "Not Available")
                g[6] = v.get("acmg_score", "Not Available")
                g[7] = v.get("acmg_criteria", "Not Available")
            except (JSONDecodeError, KeyError, IndexError):
                pass
        # InterVar
        r = safe_get("http://wintervar.wglab.org/api_new.php",
                     params={"queryType": "position", "chr": parts[0], "pos": parts[1],
                             "ref": parts[2], "alt": parts[3], "build": parts[4]})
        if r and r.status_code == 200:
            try:
                res = r.json()
                vd["InterVar_results"][0] = res.get("Intervar", "Not Available")
                vd["InterVar_results"][2] = res.get("Gene", "Not Available")
            except JSONDecodeError:
                pass
    return vd


# ==============================================================================
# PROMPT  — RAG evidence, NO phenotypes
# ==============================================================================
SYSTEM = [{
    "role": "system",
    "content": ("You are a clinician assistant chatbot specializing in genomic research "
                "and variant analysis. Interpret the provided genetic variant evidence and "
                "rank variants by likelihood of being the causative pathogenic variant in a "
                "rare Mendelian disease. No patient phenotype information is available."),
}]


def build_ranking_prompt_nohpo(variants_data, exomiser_ranks, case_name=""):
    s = ("I need a comprehensive analysis of the following variants to identify the most "
         "likely causative pathogenic variant in a rare Mendelian disease. "
         "No patient phenotype (HPO) information is available; rank using ACMG "
         "classification, predicted effect, population frequency, and curated "
         "gene-disease validity.\n\n")

    for i, vinfo in enumerate(variants_data):
        vid = vinfo.get("variant_id", f"Variant_{i+1}")
        s += f"Variant {i+1}: {vid}\n"
        s += f"Exomiser Rank: {exomiser_ranks.get(vid, i + 1)}\n"
        s += f"GeneBe Results: {vinfo['GeneBe_results']}\n"
        s += f"InterVar Results: {vinfo['InterVar_results']}\n"
        gene = vinfo["GeneBe_results"][2]
        hgnc = 'HGNC:' + str(vinfo["GeneBe_results"][3])
        s += f"ClinGen Gene-Disease relationships: {find_gene_match(gene, hgnc)}\n"
        s += "\n---\n\n"

    s += f"Case Description: {case_name}\n"
    s += "\nBased on all the data above for each variant, please:\n"
    s += "1. Rank the variants from most to least likely to be the causative pathogenic variant.\n"
    s += "2. In your ranking, consider: ACMG classification, predicted effect, population frequency, AND ClinGen gene-disease relationships.\n"
    s += "3. Explain your reasoning for each variant, considering all available evidence.\n"
    s += "4. Provide an overall conclusion about which variant(s) are most likely causative.\n"
    s += ("Additional Rules: Start your response immediately with the numbered ranking lines "
          "(e.g., '1. Variant X (GENE): ...') with no introduction or header. After listing all "
          "variants, add a single 'Conclusion:' line. Prioritize ACMG pathogenic classification "
          "and strong gene-disease validity.\n")
    return s


# ==============================================================================
# PER-FILE PROCESSING
# ==============================================================================
def process_file(html_path, provider, model_str, max_variants=MAX_VARIANTS):
    result = {
        "file_name": html_path.name, "processed_at": datetime.now().isoformat(),
        "status": "success", "error": None,
        "variants_found": 0, "phenotypes_found": 0,
        "variants": [], "phenotypes": [], "exomiser_ranks": {},
        "ai_ranking": None, "variant_details": [],
        "model": model_str, "provider": provider, "rag": True, "no_hpo": True,
    }
    try:
        html = html_path.read_text(encoding="utf-8")
        variants = extract_variants_with_regex(html, max_variants=max_variants)
        result["variants"] = variants
        result["variants_found"] = len(variants)
        exo = {v: i + 1 for i, v in enumerate(variants)}
        result["exomiser_ranks"] = exo

        if not variants:
            result["status"] = "no_variants"
            result["error"] = "No variants found in HTML"
            return result

        logger.info(f"    enriching {len(variants)} variants (GeneBe/InterVar/ClinGen)…")
        variants_data = []
        for i, v in enumerate(variants):
            vinfo = process_variant(v, i)
            vinfo["variant_id"] = v
            variants_data.append(vinfo)
            result["variant_details"].append({
                "variant": v, "exomiser_rank": exo.get(v, i + 1),
                "acmg_classification": vinfo["GeneBe_results"][0],
                "gene_symbol": vinfo["GeneBe_results"][2],
                "effect": vinfo["GeneBe_results"][1],
                "intervar": vinfo["InterVar_results"][0],
                "acmg_criteria": vinfo["GeneBe_results"][7],
            })

        case_name = html_path.stem
        prompt = build_ranking_prompt_nohpo(variants_data, exo, case_name)
        messages = SYSTEM + [{"role": "user", "content": prompt}]

        logger.info(f"    → calling {provider}/{model_str}")
        result["ai_ranking"] = call_llm(provider, model_str, messages)
        logger.info(f"    ✅ done: {case_name}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"    ❌ {html_path.name}: {e}")
    return result


# ==============================================================================
# SKIP / PAUSE
# ==============================================================================
def already_done(html_file, out_dir):
    out = Path(out_dir) / (html_file.stem + "_rag_nohpo.json")
    if not out.exists():
        return False
    try:
        d = json.loads(out.read_text())
        return d.get("status") == "success" and bool(d.get("ai_ranking"))
    except Exception:
        return False


def check_pause():
    pause = Path(OUTPUT_ROOT) / "PAUSE"
    if pause.exists():
        logger.info("⏸  PAUSE — waiting (delete PAUSE to resume)")
        while pause.exists():
            time.sleep(5)
        logger.info("▶  Resuming")


# ==============================================================================
# RUN ONE MODEL
# ==============================================================================
def run_model(display_name, provider, model_str, html_files, delay, max_variants):
    if not API_KEYS.get(provider, ""):
        logger.warning(f"\n⚠️  Skipping '{display_name}' — no {provider.upper()}_API_KEY.")
        return

    safe = re.sub(r'[^A-Za-z0-9_\-\.]', '_', display_name)
    out_dir = os.path.join(OUTPUT_ROOT, safe)
    os.makedirs(out_dir, exist_ok=True)

    done = sum(1 for f in html_files if already_done(f, out_dir))
    todo = len(html_files) - done
    logger.info(f"\n{'='*65}")
    logger.info(f"MODEL: {display_name}  [{provider}]  (RAG, no-HPO)")
    logger.info(f"  Output : {out_dir}")
    logger.info(f"  Delay  : {delay:.1f}s   Max variants: {max_variants}")
    logger.info(f"  {done} done | {todo} to process")
    logger.info(f"{'='*65}")
    if todo == 0:
        logger.info("  ✅ already complete — skipping")
        return

    ok = fail = skip = 0
    for idx, html_path in enumerate(html_files, 1):
        check_pause()
        if already_done(html_path, out_dir):
            skip += 1
            continue
        logger.info(f"  [{idx}/{len(html_files)}] {html_path.name}")
        result = process_file(html_path, provider, model_str, max_variants)
        (Path(out_dir) / (html_path.stem + "_rag_nohpo.json")).write_text(
            json.dumps(result, indent=2, ensure_ascii=False))
        ok += result["status"] == "success"
        fail += result["status"] != "success"
        time.sleep(delay)

    summary = {"model": display_name, "provider": provider, "api_model": model_str,
               "rag": True, "no_hpo": True, "run_at": datetime.now().isoformat(),
               "total_files": len(html_files), "successful": ok, "failed": fail, "skipped": skip}
    (Path(out_dir) / "_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"\n  Model complete → OK:{ok}  FAIL:{fail}  SKIP:{skip}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    ap = argparse.ArgumentParser(description="RAG no-HPO batch re-ranking")
    ap.add_argument("--model", default=None)
    ap.add_argument("--html-dir", default=None)
    args = ap.parse_args()
    hdir = args.html_dir or HTML_DIR

    print("\n" + "="*65)
    print("  RAG + NO-HPO Batch Re-ranking — all 8 models")
    print("="*65)
    print(f"  HTML dir   : {hdir}")
    print(f"  Output root: {OUTPUT_ROOT}")
    print(f"  Phenotype-blind: {FORCE_NO_HPO}")
    print("="*65 + "\n")

    if not Path(hdir).is_dir():
        logger.error(f"HTML directory not found: {hdir}")
        sys.exit(1)
    html_files = sorted(Path(hdir).glob("*.html"))
    if not html_files:
        logger.error(f"No HTML files in: {hdir}")
        sys.exit(1)
    logger.info(f"Found {len(html_files)} HTML files")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # warm the ClinGen cache once
    load_gene_disease_summary()

    models = MODELS
    if args.model:
        models = [m for m in MODELS if m[0] == args.model]
        if not models:
            logger.error("Unknown model. Valid: " + ", ".join(m[0] for m in MODELS))
            sys.exit(1)

    for display_name, provider, model_str, delay, max_vars in models:
        try:
            run_model(display_name, provider, model_str, html_files, delay, max_vars)
        except KeyboardInterrupt:
            logger.info("\n⛔ Interrupted — progress saved, re-run to continue")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Model '{display_name}' failed entirely: {e} — continuing")

    print("\n" + "="*65)
    print(f"  ALL MODELS COMPLETE → {OUTPUT_ROOT}/")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
