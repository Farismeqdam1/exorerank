#!/usr/bin/env python3
"""
Batch AI Reranking Script for Exomiser HTML Results
====================================================
This script processes all HTML files in the Exomiser results directory,
extracts variants and phenotypes, performs AI reranking using the same
prompts and logic as App.py, and saves results as JSON files.

Usage:
    python batch_rerank.py

Requirements:
    - GROQ_API_KEY environment variable (or in ~/.streamlit/secrets.toml)
    - EUTILS_API_KEY environment variable (or in ~/.streamlit/secrets.toml)
    
Output:
    JSON files in ./ai_reranking_results/ directory
"""

import os
import sys
import re
import json
import time
import urllib.parse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from json.decoder import JSONDecodeError
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required packages
try:
    import requests
    from groq import Groq
    import obonet
    import genebe as gnb
    import pandas as pd
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    logger.error("Please install: pip install requests groq obonet genebe pandas")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing Exomiser HTML results
HTML_DIR = "/home/faris/exomiser-cli-14.1.0-distribution_faster/exomiser-cli-14.1.0/hporesult"

# Output directory for AI reranking results
OUTPUT_DIR = "/home/faris/reranking_qwen"

# Model configuration
DEFAULT_MODEL = "qwen/qwen3-32b"

# Processing settings
MAX_WORKERS = 4
MAX_VARIANTS = 50  # Maximum variants to process per file

# =============================================================================
# API KEY LOADING
# =============================================================================

def load_api_keys():
    """Load API keys from environment variables or Streamlit secrets."""
    eutils_api_key = os.getenv("EUTILS_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    # Try to load from Streamlit secrets file
    secrets_path = Path.home() / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            with open(secrets_path, 'r') as f:
                content = f.read()
                
            # Simple TOML parsing for our keys
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('eutils_api_key') and not eutils_api_key:
                    match = re.search(r'["\'](.+?)["\']', line)
                    if match:
                        eutils_api_key = match.group(1)
                elif line.startswith('groq_api_key') and not groq_api_key:
                    match = re.search(r'["\'](.+?)["\']', line)
                    if match:
                        groq_api_key = match.group(1)
                elif line.startswith('EUTILS_API_KEY') and not eutils_api_key:
                    match = re.search(r'["\'](.+?)["\']', line)
                    if match:
                        eutils_api_key = match.group(1)
                elif line.startswith('GROQ_API_KEY') and not groq_api_key:
                    match = re.search(r'["\'](.+?)["\']', line)
                    if match:
                        groq_api_key = match.group(1)
        except Exception as e:
            logger.warning(f"Could not read secrets file: {e}")
    
    return eutils_api_key, groq_api_key

# Load API keys
EUTILS_API_KEY, GROQ_API_KEY = load_api_keys()

# Validate API keys
if not EUTILS_API_KEY:
    logger.error("❌ Missing NCBI E-utilities key. Set EUTILS_API_KEY environment variable or add to ~/.streamlit/secrets.toml")
    sys.exit(1)

if not GROQ_API_KEY:
    logger.error("❌ Missing Groq API key. Set GROQ_API_KEY environment variable or add to ~/.streamlit/secrets.toml")
    sys.exit(1)

logger.info("✅ API keys loaded successfully")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Requests session for connection pooling
session = requests.Session()

# =============================================================================
# HELPER FUNCTIONS (from App.py)
# =============================================================================

def safe_get(url, *, headers=None, params=None, retries=3, backoff=2, timeout=10):
    """Wrapper around session.get with simple retry logic."""
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            logger.debug(f"Request error for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
    return None


def call_ai(messages, max_tokens: int = 1024) -> str:
    """Send a chat completion request using the Groq API."""
    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=1,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content


# =============================================================================
# HTML PARSING FUNCTIONS (from App.py)
# =============================================================================

def extract_variants_with_regex(html_content, max_variants=None):
    """Extract variants from Exomiser HTML output."""
    logger.debug("Extracting variants from HTML")
    variants = []
    
    # Split on "Variants contributing to score:" (with HTML <b> tags)
    split_sections = re.split(r'(?i)<b>\s*Variants contributing to score:\s*</b>', html_content)
    
    for section in split_sections[1:]:  # Skip the first section (before the split pattern)
        # Stop processing when we reach "Other passed variants:" section
        other_variants_match = re.search(r'(?i)<b>\s*Other passed variants:\s*</b>', section)
        if other_variants_match:
            section = section[:other_variants_match.start()]
        
        # Extract variant patterns from the section
        matches = re.findall(r'\b([XY\d]+-\d+-[ACGT]+-[ACGT\w]+)\b', section)
        variants.extend(matches)
        
        if max_variants and len(variants) >= max_variants:
            break
    
    return variants if max_variants is None else variants[:max_variants]


def extract_hpo_ids(html_content, max_hpos=20):
    """Extract HPO IDs from Exomiser HTML output."""
    logger.debug("Extracting HPO IDs from HTML")
    pre_block = re.search(r'<pre>(.*?)</pre>', html_content, re.DOTALL)
    if not pre_block:
        return []
    hpo_matches = re.findall(r'-\s*&quot;(HP:\d{7})&quot;', pre_block.group(1))
    return hpo_matches[:max_hpos]


# HPO ontology cache
_hpo_graph = None

def get_hpo_graph():
    """Load HPO ontology (cached). Tries multiple sources."""
    global _hpo_graph
    if _hpo_graph is None:
        logger.info("Loading HPO ontology (one-time load)...")
        
        # Try multiple URLs - the original one in App.py may be outdated
        hpo_urls = [
            'https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo',
            'http://purl.obolibrary.org/obo/hp.obo',
            'https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo',
        ]
        
        for url in hpo_urls:
            try:
                logger.info(f"  Trying: {url}")
                _hpo_graph = obonet.read_obo(url)
                logger.info("✅ HPO ontology loaded successfully!")
                return _hpo_graph
            except Exception as e:
                logger.debug(f"  Failed: {e}")
                continue
        
        # If all URLs fail, log error and return empty dict
        logger.error("❌ Could not load HPO ontology from any source")
        _hpo_graph = {}
    
    return _hpo_graph


def get_hpo_name(hpo_id):
    """Get the human-readable name for an HPO ID."""
    graph = get_hpo_graph()
    if not graph:
        return None
    node = graph.nodes.get(hpo_id)
    return node.get('name') if node else None


# =============================================================================
# GENE-DISEASE DATA (from App.py)
# =============================================================================

# Load ClinGen gene-disease summary
_gene_disease_df = None

def load_gene_disease_summary():
    """Load ClinGen gene-disease summary dataset."""
    global _gene_disease_df
    if _gene_disease_df is None:
        url = 'https://raw.githubusercontent.com/wah644/streamlit_app.py/main/Clingen-Gene-Disease-Summary-2025-01-03.csv'
        response = safe_get(url, timeout=20)
        if response:
            try:
                import io
                _gene_disease_df = pd.read_csv(io.StringIO(response.text), dtype=str).fillna("")
                logger.info("ClinGen gene-disease data loaded")
            except Exception as e:
                logger.warning(f"Failed to parse ClinGen data: {e}")
                _gene_disease_df = pd.DataFrame()
        else:
            _gene_disease_df = pd.DataFrame()
    return _gene_disease_df


def find_gene_match(gene_symbol, hgnc_id):
    """Find matching gene-disease relationships from ClinGen."""
    df = load_gene_disease_summary()
    if df.empty:
        return "No existing gene-disease match found"
    
    if 'GENE SYMBOL' in df.columns and 'GENE ID (HGNC)' in df.columns:
        matching_rows = df[(df['GENE SYMBOL'] == gene_symbol) & (df['GENE ID (HGNC)'] == hgnc_id)]
        if not matching_rows.empty:
            return dict(zip(matching_rows['DISEASE LABEL'], matching_rows['CLASSIFICATION']))
        else:
            return "No disease found"
    else:
        return "No existing gene-disease match found"


# =============================================================================
# VARIANT PROCESSING (from App.py)
# =============================================================================

def get_variant_info(variant_str):
    """Parse variant string into components."""
    # Try format: chr-pos-ref-alt
    parts = variant_str.split('-')
    if len(parts) == 4:
        chrom, pos, ref, alt = parts
        return True, [chrom, pos, ref, alt, "hg38"]
    return False, []


def convert_variant_format(variant_str):
    """Convert variant format for processing."""
    return variant_str


def snp_to_vcf(snp_id):
    """Convert rsID to VCF format using genebe."""
    logger.debug(f"Converting {snp_id} to VCF format")
    formatted_alleles = []
    
    try:
        if not snp_id:
            return []
        
        if snp_id.startswith('rs'):
            try:
                parsed = gnb.parse_variants([snp_id], genome="hg38")
                
                if not parsed:
                    return []
                
                if isinstance(parsed[0], dict):
                    variant_str = parsed[0]['variant']
                elif isinstance(parsed[0], str):
                    variant_str = parsed[0]
                else:
                    return []
                
                try:
                    chrom, pos, ref, alt = variant_str.split('-')
                    if pos is None or pos == "":
                        return []
                    pos = int(pos)
                except Exception:
                    return []
                
                formatted_variant = f"{chrom}-{pos}-{ref.upper()}-{alt.upper()}"
                formatted_alleles.append(formatted_variant)
                return formatted_alleles
                
            except Exception as e:
                logger.debug(f"Error processing rsID {snp_id}: {e}")
                return []
        else:
            return [snp_id] if '-' in snp_id else []
            
    except Exception as e:
        logger.debug(f"Error processing {snp_id}: {e}")
        return []


def process_variant(variant_response, variant_index):
    """Process a single variant and get ACMG classification."""
    logger.debug(f"Processing variant {variant_index}: {variant_response}")
    
    is_valid, parts = get_variant_info(variant_response)
    
    variant_data = {
        "GeneBe_results": ['-', '-', '-', '-', '-', '-', '-', '-'],
        "InterVar_results": ['-', '', '-', ''],
        "disease_classification_dict": {"No diseases found"},
        "hgvs_val": "",
        "paper_count": 0,
        "papers": []
    }
    
    if is_valid:
        # GeneBe API for ACMG classification
        url = "https://api.genebe.net/cloud/api-public/v1/variant"
        params = {
            "chr": parts[0],
            "pos": parts[1],
            "ref": parts[2],
            "alt": parts[3],
            "genome": parts[4]
        }
        headers = {"Accept": "application/json"}
        
        response = safe_get(url, headers=headers, params=params)
        
        if response and response.status_code == 200:
            try:
                data = response.json()
                variant = data["variants"][0]
                variant_data["GeneBe_results"][0] = variant.get("acmg_classification", "Not Available")
                variant_data["GeneBe_results"][1] = variant.get("effect", "Not Available")
                variant_data["GeneBe_results"][2] = variant.get("gene_symbol", "Not Available")
                variant_data["GeneBe_results"][3] = variant.get("gene_hgnc_id", "Not Available")
                variant_data["GeneBe_results"][4] = variant.get("dbsnp", "Not Available")
                variant_data["GeneBe_results"][5] = variant.get("frequency_reference_population", "Not Available")
                variant_data["GeneBe_results"][6] = variant.get("acmg_score", "Not Available")
                variant_data["GeneBe_results"][7] = variant.get("acmg_criteria", "Not Available")
            except (JSONDecodeError, KeyError, IndexError):
                pass
        
        # InterVar API
        url = "http://wintervar.wglab.org/api_new.php"
        params = {
            "queryType": "position",
            "chr": parts[0],
            "pos": parts[1],
            "ref": parts[2],
            "alt": parts[3],
            "build": parts[4]
        }
        
        response = safe_get(url, params=params)
        
        if response and response.status_code == 200:
            try:
                results = response.json()
                variant_data["InterVar_results"][0] = results.get("Intervar", "Not Available")
                variant_data["InterVar_results"][2] = results.get("Gene", "Not Available")
            except JSONDecodeError:
                pass
    
    return variant_data


def get_gene_phenotype_paper_counts(gene_symbol, phenotypes):
    """Search for papers linking a gene with specific phenotypes."""
    logger.debug(f"Getting paper counts for {gene_symbol}")
    
    if not gene_symbol or not phenotypes:
        return {}
    
    gene_symbol = str(gene_symbol).strip()
    if not gene_symbol:
        return {}
    
    counts = {}
    
    for phenotype in phenotypes:
        if phenotype is None or not str(phenotype).strip():
            continue
        
        phenotype_str = str(phenotype).strip()
        
        if len(phenotype_str) > 100:
            continue
        
        try:
            encoded_gene = urllib.parse.quote_plus(gene_symbol)
            encoded_phenotype = urllib.parse.quote_plus(phenotype_str)
            
            url = (
                f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
                f"?db=pubmed&term={encoded_gene}+AND+{encoded_phenotype}"
                f"&retmode=json&api_key={EUTILS_API_KEY}"
            )
            
            response = session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                count = int(data.get('esearchresult', {}).get('count', 0))
                counts[phenotype_str] = count
            else:
                counts[phenotype_str] = 0
                
        except Exception as e:
            logger.debug(f"Error querying {gene_symbol} and {phenotype_str}: {e}")
            counts[phenotype_str] = 0
    
    return counts


# =============================================================================
# AI RANKING (from App.py - EXACT SAME PROMPTS)
# =============================================================================

# System message for variant ranking (from App.py)
SYSTEM_1 = [
    {
        "role": "system",
        "content": (
            "You are a clinician assistant chatbot specializing in genomic research and variant analysis. "
            "Your task is to interpret user-provided genetic variant data, and identify possible Mendelian diseases linked to genes if provided with research paper articles."
        ),
    }
]


def build_ranking_prompt(variants_data, phenotypes, exomiser_ranks, case_name=""):
    """
    Build the AI ranking prompt - EXACTLY as in App.py lines 2102-2178.
    This preserves the exact same prompt structure and logic.
    """
    all_variants_summary = f"I need a comprehensive analysis of the following variants in relation to these phenotypes: {', '.join(phenotypes)}\n\n"
    
    # Collect gene-phenotype paper counts for all genes across variants
    gene_phenotype_data = {}
    
    for i, variant_info in enumerate(variants_data):
        gene_symbol = variant_info["GeneBe_results"][2]
        if gene_symbol not in gene_phenotype_data and gene_symbol != "Not Available" and gene_symbol != "-":
            gene_phenotype_data[gene_symbol] = get_gene_phenotype_paper_counts(gene_symbol, phenotypes)
    
    # First, add a summary of gene-phenotype literature counts
    all_variants_summary += "### Gene-Phenotype Literature Summary:\n"
    for gene, phenotype_counts in gene_phenotype_data.items():
        all_variants_summary += f"Gene: {gene}\n"
        for phenotype, count in phenotype_counts.items():
            all_variants_summary += f"  - {phenotype}: {count} papers\n"
    all_variants_summary += "\n---\n\n"
    
    # Then add detailed variant information
    for i, variant_info in enumerate(variants_data):
        variant_id = variant_info.get("variant_id", f"Variant_{i+1}")
        all_variants_summary += f"Variant {i+1}: {variant_id}\n"
        
        ex_rank = exomiser_ranks.get(variant_id, i + 1)
        all_variants_summary += f"Exomiser Rank: {ex_rank}\n"
        all_variants_summary += f"GeneBe Results: {variant_info['GeneBe_results']}\n"
        all_variants_summary += f"InterVar Results: {variant_info['InterVar_results']}\n"
        
        # Add gene-disease relationship from ClinGen
        gene_symbol = variant_info["GeneBe_results"][2]
        hgnc_id = 'HGNC:' + str(variant_info["GeneBe_results"][3])
        disease_dict = find_gene_match(gene_symbol, hgnc_id)
        all_variants_summary += f"ClinGen Gene-Disease relationships: {disease_dict}\n"
        
        # Add paper counts for this gene if available
        if gene_symbol in gene_phenotype_data:
            all_variants_summary += f"Gene-Phenotype Paper Counts: {gene_phenotype_data[gene_symbol]}\n"
        
        all_variants_summary += "\n---\n\n"
    
    # Add case description if provided
    all_variants_summary += f"Case Description: {case_name}\n"
    
    # Add ranking instructions (EXACT copy from App.py lines 2167-2178)
    all_variants_summary += f"\nBased on all the data above for each variant, please:\n"
    all_variants_summary += f"1. Rank the variants from most to least likely to cause the phenotypes '{', '.join(phenotypes)}'.\n"
    all_variants_summary += f"2. In your ranking, consider: ACMG classification, ClinGen gene-disease relationships, AND the gene-phenotype literature counts shown at the beginning.\n"
    all_variants_summary += f"3. Explain your reasoning for each variant, considering all available evidence.\n"
    all_variants_summary += f"4. Provide an overall conclusion about which variant(s) most likely explain the phenotypes.\n"
    all_variants_summary += f"5. Specifically mention the gene-phenotype literature evidence in your analysis when relevant.\n"
    all_variants_summary += (
        "Additional Rules: Start your response immediately with the numbered ranking lines "
        "(e.g., '1. Variant X (GENE): ...') and do not include any introduction or header. "
        "After listing all variants, add a single 'Conclusion:' line summarizing your overall reasoning. "
        "Try to consider the overall phenotype terms rather than one only and prioritize ACMG pathogenic classification.\n"
    )
    
    return all_variants_summary


def get_ai_ranking(prompt):
    """Get AI ranking response using the same method as App.py."""
    full_message = SYSTEM_1 + [{"role": "user", "content": prompt}]
    return call_ai(full_message, max_tokens=3072)


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_html_file(html_path):
    """
    Process a single Exomiser HTML file and generate AI reranking.
    Returns a dictionary with all results.
    """
    logger.info(f"Processing: {html_path}")
    
    result = {
        "file_name": os.path.basename(html_path),
        "file_path": str(html_path),
        "processed_at": datetime.now().isoformat(),
        "status": "success",
        "error": None,
        "variants_found": 0,
        "phenotypes_found": 0,
        "variants": [],
        "phenotypes": [],
        "exomiser_ranks": {},
        "ai_ranking": None,
        "variant_details": []
    }
    
    try:
        # Read HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract variants
        file_variants = extract_variants_with_regex(html_content, max_variants=200)
        result["variants_found"] = len(file_variants)
        result["variants"] = file_variants
        
        # Store Exomiser ranking (based on order in HTML)
        exomiser_ranks = {variant: idx + 1 for idx, variant in enumerate(file_variants)}
        result["exomiser_ranks"] = exomiser_ranks
        
        # Extract HPO IDs and convert to names
        hpo_ids = extract_hpo_ids(html_content)
        phenotypes = [name for name in (get_hpo_name(hpo_id) for hpo_id in hpo_ids) if name]
        phenotypes = phenotypes[:20]  # Limit to 20 phenotypes
        result["phenotypes_found"] = len(phenotypes)
        result["phenotypes"] = phenotypes
        
        if not file_variants:
            result["status"] = "no_variants"
            result["error"] = "No variants found in HTML file"
            return result
        
        if not phenotypes:
            logger.warning(f"  ⚠️ No phenotypes found - ranking will be limited")
            # Continue anyway - we can still rank based on ACMG
        
        # Limit to MAX_VARIANTS for processing
        variants_to_process = file_variants[:MAX_VARIANTS]
        
        # Process each variant
        logger.info(f"  Processing {len(variants_to_process)} variants with {len(phenotypes)} phenotypes...")
        variants_data = []
        
        for i, variant in enumerate(variants_to_process):
            variant_info = process_variant(variant, i)
            variant_info["variant_id"] = variant
            variants_data.append(variant_info)
            result["variant_details"].append({
                "variant": variant,
                "exomiser_rank": exomiser_ranks.get(variant, i + 1),
                "acmg_classification": variant_info["GeneBe_results"][0],
                "gene_symbol": variant_info["GeneBe_results"][2],
                "effect": variant_info["GeneBe_results"][1],
                "intervar": variant_info["InterVar_results"][0],
                "acmg_criteria": variant_info["GeneBe_results"][7]
            })
        
        # Build AI ranking prompt
        case_name = os.path.splitext(os.path.basename(html_path))[0]
        prompt = build_ranking_prompt(variants_data, phenotypes, exomiser_ranks, case_name)
        
        # Get AI ranking
        logger.info(f"  Getting AI ranking...")
        ai_response = get_ai_ranking(prompt)
        result["ai_ranking"] = ai_response
        
        logger.info(f"  ✅ Completed: {case_name}")
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"  ❌ Error processing {html_path}: {e}")
    
    return result


def check_pause():
    """
    Check if pause file exists. To pause: touch /home/faris/ai_reranking_results/PAUSE
    To resume: rm /home/faris/ai_reranking_results/PAUSE
    """
    pause_file = os.path.join(OUTPUT_DIR, "PAUSE")
    if os.path.exists(pause_file):
        logger.info("⏸️  PAUSE file detected. Waiting...")
        logger.info(f"   To resume, delete: {pause_file}")
        while os.path.exists(pause_file):
            time.sleep(5)
        logger.info("▶️  Resuming...")


def is_already_processed(html_file):
    """Check if this HTML file has already been processed (JSON exists and has ai_ranking)."""
    output_filename = html_file.stem + "_ai_reranking.json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                data = json.load(f)
            # Check if it was successfully processed (has ai_ranking content)
            if data.get("ai_ranking") and data.get("status") == "success":
                return True
        except (json.JSONDecodeError, IOError):
            pass
    return False


def main():
    """Main function to process all HTML files."""
    print("=" * 70)
    print("Batch AI Reranking for Exomiser Results")
    print("=" * 70)
    print("")
    print("CONTROLS:")
    print(f"  PAUSE:  touch {OUTPUT_DIR}/PAUSE")
    print(f"  RESUME: rm {OUTPUT_DIR}/PAUSE")
    print("  STOP:   Ctrl+C (will resume from where it left off next run)")
    print("=" * 70)
    
    # Check if HTML directory exists
    if not os.path.isdir(HTML_DIR):
        logger.error(f"❌ HTML directory not found: {HTML_DIR}")
        logger.error("Please update HTML_DIR in the script to point to your Exomiser results.")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Find all HTML files
    html_files = list(Path(HTML_DIR).glob("*.html"))
    
    if not html_files:
        logger.error(f"❌ No HTML files found in: {HTML_DIR}")
        sys.exit(1)
    
    # Check how many are already processed
    already_done = sum(1 for f in html_files if is_already_processed(f))
    remaining = len(html_files) - already_done
    
    logger.info(f"Found {len(html_files)} HTML files total")
    logger.info(f"  Already processed: {already_done}")
    logger.info(f"  Remaining: {remaining}")
    print("-" * 70)
    
    if remaining == 0:
        logger.info("✅ All files already processed! Nothing to do.")
        return
    
    # Pre-load resources
    logger.info("Loading resources...")
    get_hpo_graph()  # Pre-load HPO ontology
    load_gene_disease_summary()  # Pre-load ClinGen data
    print("-" * 70)
    
    # Process each file
    results_summary = []
    successful = 0
    failed = 0
    skipped = 0
    
    for i, html_file in enumerate(html_files, 1):
        # Check for pause
        check_pause()
        
        # Skip if already processed
        if is_already_processed(html_file):
            print(f"[{i}/{len(html_files)}] ⏭️  Skipping (already done): {html_file.name}")
            skipped += 1
            continue
        
        print(f"\n[{i}/{len(html_files)}] Processing: {html_file.name}")
        
        # Process the file
        result = process_html_file(html_file)
        
        # Save individual JSON result
        output_filename = html_file.stem + "_ai_reranking.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  Saved: {output_filename}")
        
        # Track summary
        results_summary.append({
            "file": html_file.name,
            "status": result["status"],
            "variants_found": result["variants_found"],
            "phenotypes_found": result["phenotypes_found"]
        })
        
        if result["status"] == "success":
            successful += 1
        else:
            failed += 1
        
        # Rate limiting - be nice to the APIs
        time.sleep(1)
    
    # Save summary file
    summary_path = os.path.join(OUTPUT_DIR, "_processing_summary.json")
    summary_data = {
        "processed_at": datetime.now().isoformat(),
        "total_files": len(html_files),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "results": results_summary
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files: {len(html_files)}")
    print(f"Skipped (already done): {skipped}")
    print(f"Processed this run - Successful: {successful}")
    print(f"Processed this run - Failed: {failed}")
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
