#!/usr/bin/env python3
"""
Batch AI Re-ranking Script – NO RAG VERSION
============================================
Processes all Exomiser HTML files with ZERO external database calls.
The LLM receives only:
  • The Exomiser-ranked variant list (chr-pos-ref-alt, Exomiser rank)
  • The patient HPO phenotype names
  • Nothing from GeneBe / InterVar / ClinGen / PubMed

Runs all 8 models from AIranking.xlsx sequentially.
Each model writes results to its own sub-directory under OUTPUT_ROOT.

API routing:
  • Groq          – llama-3.1-8b-instant, Llama-3.3-70B,
                    qwen3-32b, llama-4-scout-17b-16e-instruct
  • OpenAI        – GPT 4.1
  • OpenRouter    – openai/gpt-oss-120b, openai/gpt-oss-20b,
                    kimi-k2-instruct-0905

Usage:
    python batch_rerank_norag.py [--model MODEL_NAME] [--html-dir PATH]

    --model     Run only one specific model (optional; default = all 8)
    --html-dir  Override the HTML directory

Controls (while running):
    PAUSE:  touch <OUTPUT_ROOT>/PAUSE   →  resume: rm <OUTPUT_ROOT>/PAUSE
    STOP:   Ctrl+C  (resumes from last saved file next run)
"""

import os
import re
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── package check ──────────────────────────────────────────────────────────────
try:
    import obonet
    import requests
    from groq import Groq
    import openai        # used for both OpenAI and OpenRouter (same SDK)
except ImportError as e:
    logger.error(f"Missing package: {e}")
    logger.error("pip install groq openai obonet requests")
    sys.exit(1)

# ==============================================================================
# CONFIGURATION — adjust paths to match your system
# ==============================================================================

HTML_DIR    = "/home/faris/exomiser-cli-14.1.0-distribution_faster/exomiser-cli-14.1.0/nohporesult"
OUTPUT_ROOT = "/home/faris/reranking_norag_nohpo"

MAX_VARIANTS  = 50   # top-N Exomiser variants sent to the LLM
MAX_PHENOTYPES = 20  # max HPO phenotype names in the prompt

# NO-HPO CONDITION: when True, the LLM is fully phenotype-blind —
# HPO terms are NOT extracted and NOT included in the prompt.
FORCE_NO_HPO = True

# ------------------------------------------------------------------------------
# Model registry
# Each entry: (display_name, api_provider, api_model_string, delay_sec, max_variants)
#
#   delay_sec    — seconds to sleep BETWEEN requests (avoids 429s)
#   max_variants — how many Exomiser variants to include in the prompt
#
# Groq free tier limits are tokens-per-minute, not just requests-per-minute.
# A prompt with 50 variants + 20 HPO = ~2500 tokens → hits TPM fast.
# Keeping max_variants=20 and delay=8s keeps Groq comfortably under limits.
# OpenAI / OpenRouter are more permissive → can use more variants.
# ------------------------------------------------------------------------------
MODELS = [
    # display_name               provider      api_model_string                                  delay  max_vars
    # ── Groq ─────────────────────────────────────────────────────────────────────────────────────────────────
    # Llama-3.3-70B uses more tokens per output → needs a bigger gap
    ("llama-3.1-8b-instant",    "groq",       "llama-3.1-8b-instant",                           8,    20),
    ("Llama-3.3-70B",           "groq",       "llama-3.3-70b-versatile",                        30,   10),
    ("qwen3-32b",               "groq",       "qwen/qwen3-32b",                                 15,   15),
    ("llama-4-scout-17b-16e",   "groq",       "meta-llama/llama-4-scout-17b-16e-instruct",      8,    20),
    # gpt-oss models are on Groq with their exact model strings
    ("openai/gpt-oss-120b",     "groq",       "openai/gpt-oss-120b",                            8,    20),
    ("openai/gpt-oss-20b",      "groq",       "openai/gpt-oss-20b",                             8,    20),
    # ── OpenAI ───────────────────────────────────────────────────────────────────────────────────────────────
    ("GPT 4.1",                 "openai",     "gpt-4.1",                                         2,    50),
    # ── OpenRouter ───────────────────────────────────────────────────────────────────────────────────────────
    ("kimi-k2-instruct-0905",   "openrouter", "moonshotai/kimi-k2",                              2,    50),
]

# ==============================================================================
# API KEY LOADING
# ==============================================================================

def _read_secrets_toml():
    """Read a secrets.toml / .env file and return a key→value dict."""
    secrets = {}
    candidates = [
        # ── .toml files ───────────────────────────────────────────────────────
        Path.cwd() / "secrets.toml",
        Path(__file__).parent / "secrets.toml",
        Path("/home/faris/.streamlit/secrets.toml"),          # ← confirmed location
        Path.home() / ".streamlit" / "secrets.toml",
        Path.home() / "secrets.toml",
        # ── .env files ────────────────────────────────────────────────────────
        Path.cwd() / ".env",
        Path(__file__).parent / ".env",
        Path("/home/faris/Downloads/DreamJournal/.env"),      # ← confirmed location
        Path("/home/faris/groq-prompt-optimizer/.env"),       # ← confirmed location
        Path.home() / ".env",
    ]
    for path in candidates:
        if not path.exists():
            continue
        logger.info(f"  Reading keys from: {path}")
        try:
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    # strip surrounding quotes if present
                    v = v.strip().strip('"').strip("'")
                    if v:
                        secrets[k.strip().upper()] = v
        except Exception as e:
            logger.warning(f"  Could not read {path}: {e}")
    return secrets


def _read_dotenv():
    return {}   # merged into _read_secrets_toml above


def load_api_keys():
    """
    Load API keys from (in priority order):
      1. Environment variables  (export KEY=... in your shell)
      2. Any secrets.toml / .env found in the candidate list above
    """
    logger.info("Searching for API keys...")
    file_keys = _read_secrets_toml()

    def _get(name):
        return (os.getenv(name)
                or os.getenv(name.lower())
                or file_keys.get(name.upper())
                or file_keys.get(name.lower())
                or "")

    keys = {
        "groq":       _get("GROQ_API_KEY"),
        "openai":     _get("OPENAI_API_KEY"),
        "openrouter": _get("OPENROUTER_API_KEY"),
    }

    for provider, key in keys.items():
        status = f"✅ found ({key[:8]}…)" if key else "❌ not found — models using this provider will be skipped"
        logger.info(f"  {provider:<12} {status}")

    return keys


API_KEYS = load_api_keys()

# Validate that at least one key is present
if not any(API_KEYS.values()):
    logger.error(
        "No API keys found. Set GROQ_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY "
        "as environment variables or in ~/.streamlit/secrets.toml"
    )
    sys.exit(1)

# ==============================================================================
# API CLIENTS  (initialised lazily so missing keys for unused providers don't fail)
# ==============================================================================

_clients = {}

def get_client(provider: str):
    """Return (and cache) the SDK client for the given provider."""
    if provider in _clients:
        return _clients[provider]

    key = API_KEYS.get(provider, "")
    if not key:
        raise RuntimeError(
            f"No API key for provider '{provider}'. "
            f"Set {provider.upper()}_API_KEY."
        )

    if provider == "groq":
        _clients[provider] = Groq(api_key=key)

    elif provider == "openai":
        _clients[provider] = openai.OpenAI(api_key=key)

    elif provider == "openrouter":
        _clients[provider] = openai.OpenAI(
            api_key=key,
            base_url="https://openrouter.ai/api/v1",
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return _clients[provider]


# ==============================================================================
# ==============================================================================
# LLM CALL  (provider-agnostic, 429-aware)
# ==============================================================================

def call_llm(provider: str, model_str: str, messages: list,
             max_tokens: int = 3072, retries: int = 6) -> str:
    """
    Send a chat-completion request. Returns the reply text.

    Rate-limit handling:
      • On HTTP 429 the Groq/OpenAI SDK raises an exception whose string
        contains the 'retry-after' seconds (e.g. '42.000000 seconds').
        We parse that value and sleep exactly that long before retrying.
      • For all other transient errors we use exponential back-off (2^n s).
      • Max retries: 6  (handles the occasional very long Groq back-off).
    """
    client = get_client(provider)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model_str,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except Exception as e:
            err_str = str(e)

            # ── 429 Rate Limit ────────────────────────────────────────────────
            if "429" in err_str or "rate_limit" in err_str.lower() or "too many requests" in err_str.lower():
                # Try to read the exact retry-after value from the error message
                m = re.search(r'(\d+(?:\.\d+)?)\s*second', err_str, re.IGNORECASE)
                wait = float(m.group(1)) + 2 if m else 60.0   # +2 s safety margin
                logger.warning(
                    f"  ⚠️  429 Rate limit (attempt {attempt+1}/{retries}) — "
                    f"sleeping {wait:.0f}s then retrying…"
                )
                time.sleep(wait)

            # ── Other transient errors ────────────────────────────────────────
            else:
                wait = min(2 ** attempt, 60)
                logger.warning(
                    f"  ⚠️  API error (attempt {attempt+1}/{retries}): {e} "
                    f"— retrying in {wait}s"
                )
                time.sleep(wait)

    raise RuntimeError(f"All {retries} attempts failed for {provider}/{model_str}")


# ==============================================================================
# HTML PARSING  (identical to RAG version)
# ==============================================================================

def extract_variants(html: str, max_variants: int = MAX_VARIANTS) -> list:
    """Return list of variant strings (chr-pos-ref-alt) from Exomiser HTML."""
    variants = []
    sections = re.split(
        r'(?i)<b>\s*Variants contributing to score:\s*</b>', html
    )
    for section in sections[1:]:
        stop = re.search(r'(?i)<b>\s*Other passed variants:\s*</b>', section)
        chunk = section[: stop.start()] if stop else section
        matches = re.findall(r'\b([XY\d]+-\d+-[ACGT]+-[ACGT\w]+)\b', chunk)
        variants.extend(matches)
        if len(variants) >= max_variants:
            break
    return variants[:max_variants]


def extract_hpo_ids(html: str, max_hpos: int = MAX_PHENOTYPES) -> list:
    """Return HPO IDs from the <pre> block in Exomiser HTML."""
    pre = re.search(r'<pre>(.*?)</pre>', html, re.DOTALL)
    if not pre:
        return []
    return re.findall(r'-\s*&quot;(HP:\d{7})&quot;', pre.group(1))[:max_hpos]


# HPO ontology (loaded once)
_hpo_graph = None

def get_hpo_graph():
    global _hpo_graph
    if _hpo_graph is None:
        logger.info("Loading HPO ontology (one-time)…")
        for url in [
            "https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo",
            "http://purl.obolibrary.org/obo/hp.obo",
        ]:
            try:
                _hpo_graph = obonet.read_obo(url)
                logger.info("✅ HPO ontology loaded")
                return _hpo_graph
            except Exception:
                continue
        logger.error("❌ Could not load HPO ontology — phenotype names will be blank")
        _hpo_graph = {}
    return _hpo_graph


def hpo_name(hpo_id: str) -> str:
    graph = get_hpo_graph()
    if not graph:
        return hpo_id
    node = graph.nodes.get(hpo_id)
    return node.get("name", hpo_id) if node else hpo_id


# ==============================================================================
# PROMPT BUILDER  — No-RAG version
# ==============================================================================

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a clinical genomics expert specialising in rare Mendelian disease diagnosis. "
        "Your task is to re-rank a list of candidate pathogenic variants produced by Exomiser, "
        "using only your own medical and genomics knowledge — no external databases are queried. "
        "Base your reasoning on known gene-disease relationships, variant effect classes, "
        "inheritance patterns, and the clinical relevance of each candidate gene to the "
        "patient's phenotype."
    ),
}


def build_norag_prompt(variants: list, phenotypes: list,
                       case_name: str = "") -> str:
    """
    Build the No-RAG prompt.
    Provides ONLY:
      - Patient HPO phenotypes (names)
      - Exomiser-ranked variant list (variant string + rank)
    No ACMG scores, no ClinGen, no InterVar, no PubMed.
    """
    lines = []

    lines.append(
        "You are given an Exomiser-ranked list of candidate variants for a patient "
        "with a suspected rare Mendelian disease. No patient phenotype information is "
        "available. Re-rank these variants from most to least likely to be causal, "
        "using only your knowledge of gene-disease relationships and variant biology."
    )
    lines.append("")

    # ── Patient phenotypes — OMITTED in the pure no-HPO condition ─────────────
    if not FORCE_NO_HPO and phenotypes:
        lines.append("### Patient HPO Phenotypes:")
        for ph in phenotypes:
            lines.append(f"  - {ph}")
        lines.append("")

    # ── Exomiser ranked list ──────────────────────────────────────────────────
    lines.append("### Exomiser-Ranked Variant List (rank | variant):")
    for rank, var in enumerate(variants, start=1):
        lines.append(f"  {rank:>3}. {var}")
    lines.append("")

    if case_name:
        lines.append(f"Case ID: {case_name}")
        lines.append("")

    # ── Task instructions ─────────────────────────────────────────────────────
    lines.append("### Task:")
    lines.append(
        "1. Re-rank the variants above from most to least likely to be the causative "
        "pathogenic variant in a rare Mendelian disease."
    )
    lines.append(
        "2. For each variant, briefly justify your ranking based on the known biology "
        "of the gene and the general clinical significance of variants in that gene."
    )
    lines.append(
        "3. Consider variant effect class (e.g. frameshift, missense, splice-site) "
        "implied by the variant notation."
    )
    lines.append(
        "4. End with a single 'Conclusion:' line naming the most likely causal gene "
        "and variant."
    )
    lines.append("")
    lines.append(
        "IMPORTANT: Start immediately with '1. Variant …' — no preamble or headers. "
        "Do NOT access any database or URL. Use only the information above and your "
        "pre-trained medical knowledge."
    )

    return "\n".join(lines)


# ==============================================================================
# PER-FILE PROCESSOR
# ==============================================================================

def process_file(html_path: Path, provider: str, model_str: str,
                 max_variants: int = MAX_VARIANTS) -> dict:
    """Parse one HTML file, call the LLM, return result dict."""
    result = {
        "file_name":        html_path.name,
        "processed_at":     datetime.now().isoformat(),
        "status":           "success",
        "error":            None,
        "variants_found":   0,
        "phenotypes_found": 0,
        "variants":         [],
        "phenotypes":       [],
        "exomiser_ranks":   {},
        "ai_ranking":       None,
        "model":            model_str,
        "provider":         provider,
        "rag":              False,
    }

    try:
        html = html_path.read_text(encoding="utf-8")

        variants = extract_variants(html, max_variants)
        result["variants"]       = variants
        result["variants_found"] = len(variants)
        result["exomiser_ranks"] = {v: i+1 for i, v in enumerate(variants)}

        # In the pure no-HPO condition we do NOT read or use phenotypes at all
        if FORCE_NO_HPO:
            phenotypes = []
        else:
            hpo_ids    = extract_hpo_ids(html)
            phenotypes = [hpo_name(h) for h in hpo_ids]
        result["phenotypes"]       = phenotypes
        result["phenotypes_found"] = len(phenotypes)

        if not variants:
            result["status"] = "no_variants"
            result["error"]  = "No variants found in HTML"
            return result

        case_name = html_path.stem
        prompt    = build_norag_prompt(variants, phenotypes, case_name)
        messages  = [SYSTEM_MSG, {"role": "user", "content": prompt}]

        logger.info(f"    → calling {provider}/{model_str}  ({len(variants)} variants, {len(phenotypes)} HPO)")
        result["ai_ranking"] = call_llm(provider, model_str, messages)
        logger.info(f"    ✅ done: {case_name}")

    except Exception as e:
        result["status"] = "error"
        result["error"]  = str(e)
        logger.error(f"    ❌ {html_path.name}: {e}")

    return result


# ==============================================================================
# SKIP CHECK
# ==============================================================================

def already_done(html_file: Path, out_dir: str) -> bool:
    """Return True if a successful JSON already exists for this file."""
    out = Path(out_dir) / (html_file.stem + "_norag_nohpo.json")
    if not out.exists():
        return False
    try:
        data = json.loads(out.read_text())
        return data.get("status") == "success" and bool(data.get("ai_ranking"))
    except Exception:
        return False


# ==============================================================================
# PAUSE SUPPORT
# ==============================================================================

def check_pause():
    pause = Path(OUTPUT_ROOT) / "PAUSE"
    if pause.exists():
        logger.info("⏸  PAUSE detected — waiting…  (delete the PAUSE file to resume)")
        while pause.exists():
            time.sleep(5)
        logger.info("▶  Resuming")


# ==============================================================================
# RUN ONE MODEL OVER ALL HTML FILES
# ==============================================================================

def run_model(display_name: str, provider: str, model_str: str,
              html_files: list, delay: float, max_variants: int):
    """Process every HTML file for one model, writing to its own output folder."""

    # ── check API key is available before doing anything ─────────────────────
    if not API_KEYS.get(provider, ""):
        logger.warning(
            f"\n⚠️  Skipping '{display_name}' — no API key for provider '{provider}'.\n"
            f"   Add {provider.upper()}_API_KEY to your environment or secrets.toml."
        )
        return

    # ── create a dedicated output folder for this model ───────────────────────
    # Replace characters that are invalid in directory names
    safe_name = re.sub(r'[^A-Za-z0-9_\-\.]', '_', display_name)
    out_dir   = os.path.join(OUTPUT_ROOT, safe_name)
    os.makedirs(out_dir, exist_ok=True)   # creates the folder if it doesn't exist

    done    = sum(1 for f in html_files if already_done(f, out_dir))
    todo    = len(html_files) - done

    logger.info(f"\n{'='*65}")
    logger.info(f"MODEL: {display_name}  [{provider}]")
    logger.info(f"  Output     : {out_dir}")
    logger.info(f"  Delay      : {delay:.1f}s between requests")
    logger.info(f"  Max variants: {max_variants} per case")
    logger.info(f"  {done} already done  |  {todo} to process")
    logger.info(f"{'='*65}")

    if todo == 0:
        logger.info("  ✅ All files already processed — skipping model")
        return

    ok = fail = skip = 0
    summary_rows = []

    for idx, html_path in enumerate(html_files, 1):
        check_pause()

        if already_done(html_path, out_dir):
            logger.info(f"  [{idx}/{len(html_files)}] ⏭  skip (done): {html_path.name}")
            skip += 1
            continue

        logger.info(f"  [{idx}/{len(html_files)}] {html_path.name}")
        result = process_file(html_path, provider, model_str, max_variants)

        # Save JSON
        out_path = Path(out_dir) / (html_path.stem + "_norag_nohpo.json")
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        summary_rows.append({
            "file":             html_path.name,
            "status":           result["status"],
            "variants_found":   result["variants_found"],
            "phenotypes_found": result["phenotypes_found"],
        })

        if result["status"] == "success":
            ok += 1
        else:
            fail += 1

        # Respect per-provider rate limits between successful calls
        # Respect per-model rate limit delay between calls
        time.sleep(delay)

    # Save per-model summary
    summary = {
        "model":        display_name,
        "provider":     provider,
        "api_model":    model_str,
        "rag":          False,
        "run_at":       datetime.now().isoformat(),
        "total_files":  len(html_files),
        "successful":   ok,
        "failed":       fail,
        "skipped":      skip,
        "files":        summary_rows,
    }
    summary_path = Path(out_dir) / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    logger.info(f"\n  Model complete → OK:{ok}  FAIL:{fail}  SKIP:{skip}")
    logger.info(f"  Summary: {summary_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args():
    ap = argparse.ArgumentParser(description="No-RAG batch Exomiser re-ranking")
    ap.add_argument(
        "--model", default=None,
        help="Run only this model (use the display name from the script, e.g. 'GPT 4.1')"
    )
    ap.add_argument(
        "--html-dir", default=None,
        help="Override HTML_DIR in the script"
    )
    return ap.parse_args()


def main():
    args  = parse_args()
    hdir  = args.html_dir or HTML_DIR

    print("\n" + "="*65)
    print("  No-RAG Batch Re-ranking  —  all 8 models")
    print("="*65)
    print(f"  HTML dir   : {hdir}")
    print(f"  Output root: {OUTPUT_ROOT}")
    print(f"  PAUSE file : {OUTPUT_ROOT}/PAUSE")
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

    # Pre-load HPO ontology once for all models (skip in pure no-HPO mode)
    if not FORCE_NO_HPO:
        get_hpo_graph()

    # Filter models if --model flag is given
    models_to_run = MODELS
    if args.model:
        models_to_run = [m for m in MODELS if m[0] == args.model]
        if not models_to_run:
            valid = ", ".join(m[0] for m in MODELS)
            logger.error(f"Unknown model '{args.model}'. Valid names:\n  {valid}")
            sys.exit(1)

    # ── main loop ──────────────────────────────────────────────────────────────
    for display_name, provider, model_str, delay, max_vars in models_to_run:
        try:
            run_model(display_name, provider, model_str, html_files, delay, max_vars)
        except KeyboardInterrupt:
            logger.info("\n⛔ Interrupted by user — progress saved, re-run to continue")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Model '{display_name}' failed entirely: {e}")
            logger.error("  Continuing with next model…")

    print("\n" + "="*65)
    print("  ALL MODELS COMPLETE")
    print(f"  Results in: {OUTPUT_ROOT}/")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()
