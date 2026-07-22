#!/usr/bin/env python3
"""
Analyze No-RAG AI Reranking Results — All 8 Models
====================================================
Reads all _norag.json files from each model's output folder,
extracts where the target gene was ranked by the LLM, and
produces a single Excel file with:
  • Summary sheet  — Rank@1, 2-5, 6-10, 11-50, >50, not found (all models side-by-side)
  • One detail sheet per model
"""

import os
import re
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose which no-HPO condition to analyse:
CONDITION   = "norag"   # "norag"  → reranking_norag_nohpo  (suffix _norag_nohpo.json)
                        # "rag"    → reranking_rag_nohpo    (suffix _rag_nohpo.json)
if CONDITION == "norag":
    OUTPUT_ROOT = "/home/faris/reranking_norag_nohpo"
    JSON_SUFFIX = "_norag_nohpo.json"
    OUTPUT_FILE = "/home/faris/nohpo_norag_ranking.xlsx"
else:
    OUTPUT_ROOT = "/home/faris/reranking_rag_nohpo"
    JSON_SUFFIX = "_rag_nohpo.json"
    OUTPUT_FILE = "/home/faris/nohpo_rag_ranking.xlsx"
HTML_DIR    = "/home/faris/exomiser-cli-14.1.0-distribution_faster/exomiser-cli-14.1.0/hporesult"

# Maps (display_name → folder_name) — must match what batch_rerank_norag.py created
MODELS = [
    ("llama-3.1-8b-instant",   "llama-3.1-8b-instant"),
    ("Llama-3.3-70B",          "Llama-3.3-70B"),
    ("qwen3-32b",              "qwen3-32b"),
    ("llama-4-scout-17b-16e",  "llama-4-scout-17b-16e"),
    ("openai/gpt-oss-120b",    "openai_gpt-oss-120b"),
    ("openai/gpt-oss-20b",     "openai_gpt-oss-20b"),
    ("GPT 4.1",                "GPT_4.1"),
    ("kimi-k2-instruct-0905",  "kimi-k2-instruct-0905"),
]

# Rank categories (in display order)
CATEGORIES = ["Rank 1", "Ranks 2-5", "Ranks 6-10", "Ranks 11-50", "Beyond rank 50"]

# =============================================================================
# HELPERS
# =============================================================================

def extract_target_gene(filename: str) -> str | None:
    """Extract gene name from filename (first token before underscore)."""
    base = filename.replace(JSON_SUFFIX, "").replace("_exomiser", "")
    parts = base.split("_")
    return parts[0] if parts else None


def find_gene_rank(target_gene: str, ai_text: str) -> int | None:
    """
    Search the LLM response for lines like:
      1. GENE …
      1. Variant X (GENE) …
      1. Variant X: GENE …
    Returns the integer rank or None if not found.
    """
    if not ai_text or not target_gene:
        return None

    gene = re.escape(target_gene)
    patterns = [
        # "1. GENE:" or "1. GENE,"  or "1. **GENE**"
        rf'(\d+)\.\s+\*{{0,2}}{gene}\*{{0,2}}[\s,:\(]',
        # "1. Variant N (GENE)"
        rf'(\d+)\.\s+Variant\s+\d+\s*\({gene}\)',
        # "1. Variant N: GENE"
        rf'(\d+)\.\s+Variant\s+\d+\s*:\s*{gene}',
        # "1. ... GENE ..." anywhere on the numbered line
        rf'(\d+)\.[^\n]*\b{gene}\b',
    ]
    for pat in patterns:
        m = re.search(pat, ai_text, re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


def categorize(rank: int | None) -> str:
    if rank is None:        return "Not Found"
    if rank == 1:           return "Rank 1"
    if rank <= 5:           return "Ranks 2-5"
    if rank <= 10:          return "Ranks 6-10"
    if rank <= 50:          return "Ranks 11-50"
    return "Beyond rank 50"


def process_folder(folder_path: Path) -> list[dict]:
    """Load all _norag.json files from a folder and return list of result dicts."""
    records = []
    json_files = sorted(folder_path.glob("*" + JSON_SUFFIX))

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue

        # Only count successful runs
        if data.get("status") != "success" or not data.get("ai_ranking"):
            continue

        gene = extract_target_gene(jf.name)
        rank = find_gene_rank(gene, data.get("ai_ranking", ""))

        records.append({
            "file_name":          jf.name,
            "target_gene":        gene,
            "ai_rank":            rank,
            "ai_category":        categorize(rank),
            "status":             data.get("status", "?"),
            "variants_processed": data.get("variants_found", 0),
            "phenotypes_found":   data.get("phenotypes_found", 0),
        })

    return records


def build_summary_row(display_name: str, records: list[dict], total_html: int) -> dict:
    """Build one column worth of summary stats for the summary sheet."""
    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["ai_rank", "ai_category"])

    processed  = len(records)
    gene_found = int(df["ai_rank"].notna().sum()) if not df.empty else 0

    row = {
        "Metric":    None,          # filled separately
        display_name: None,         # placeholder
    }

    stats = {
        "Total cases processed": total_html,
        "Cases with AI result":  processed,
        "Gene found":            gene_found,
        "Gene NOT found":        processed - gene_found,
    }
    for cat in CATEGORIES:
        stats[cat] = int((df["ai_category"] == cat).sum()) if not df.empty else 0

    return stats


# =============================================================================
# EXCEL STYLING
# =============================================================================

HEADER_FILL   = PatternFill("solid", fgColor="1F3864")   # dark navy
SUBHDR_FILL   = PatternFill("solid", fgColor="2E75B6")   # blue
RANK1_FILL    = PatternFill("solid", fgColor="C6EFCE")   # green
NOTFOUND_FILL = PatternFill("solid", fgColor="FFCCCC")   # red
ALT_FILL      = PatternFill("solid", fgColor="EBF3FB")   # light blue

WHITE_BOLD   = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
DARK_BOLD    = Font(bold=True, name="Calibri", size=11)
NORMAL       = Font(name="Calibri", size=10)
CENTER       = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT         = Alignment(horizontal="left",   vertical="center")

def thin_border():
    s = Side(style="thin", color="AAAAAA")
    return Border(left=s, right=s, top=s, bottom=s)

def style_header(cell, fill=HEADER_FILL):
    cell.font      = WHITE_BOLD
    cell.fill      = fill
    cell.alignment = CENTER
    cell.border    = thin_border()

def style_cell(cell, fill=None, bold=False, center=False):
    cell.font      = Font(bold=bold, name="Calibri", size=10)
    cell.alignment = CENTER if center else LEFT
    cell.border    = thin_border()
    if fill:
        cell.fill = fill


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  No-RAG Ranking Analysis — All Models")
    print("=" * 65)

    # Count total HTML files as baseline
    html_total = len(list(Path(HTML_DIR).glob("*.html")))
    print(f"  Total HTML cases: {html_total}")

    wb = Workbook()
    ws_summary = wb.active
    ws_summary.title = "Summary"

    # ── collect results per model ─────────────────────────────────────────────
    all_model_records = {}   # display_name → list[dict]
    all_stats         = {}   # display_name → stats dict

    for display_name, folder_name in MODELS:
        folder = Path(OUTPUT_ROOT) / folder_name
        if not folder.exists():
            print(f"  ⬜ {display_name:<35} folder not found — skipping")
            all_model_records[display_name] = []
            all_stats[display_name] = build_summary_row(display_name, [], html_total)
            continue

        records = process_folder(folder)
        all_model_records[display_name] = records
        all_stats[display_name]         = build_summary_row(display_name, records, html_total)
        pct = records and (
            sum(1 for r in records if r["ai_rank"] == 1) / html_total * 100
        )
        print(f"  ✅ {display_name:<35} {len(records):>3} results  "
              f"Rank@1: {pct:.1f}%" if records else
              f"  ⬜ {display_name:<35} 0 results")

    # ── SUMMARY SHEET ─────────────────────────────────────────────────────────
    print("\n  Building summary sheet…")

    METRICS = [
        "Total cases processed",
        "Cases with AI result",
        "Gene found",
        "Gene NOT found",
        "Rank 1",
        "Ranks 2-5",
        "Ranks 6-10",
        "Ranks 11-50",
        "Beyond rank 50",
    ]
    model_names = [d for d, _ in MODELS]

    # Title
    ws_summary.merge_cells(f"A1:{get_column_letter(1 + len(MODELS))}1")
    title_cell = ws_summary["A1"]
    title_cell.value     = f"No-HPO {CONDITION.upper()} LLM Re-ranking — Performance Summary"
    title_cell.font      = Font(bold=True, size=14, color="1F3864", name="Calibri")
    title_cell.alignment = CENTER
    ws_summary.row_dimensions[1].height = 30

    # Header row
    ws_summary.cell(row=2, column=1, value="Metric")
    style_header(ws_summary.cell(row=2, column=1))
    for col, name in enumerate(model_names, start=2):
        c = ws_summary.cell(row=2, column=col, value=name)
        style_header(c, fill=SUBHDR_FILL)
    ws_summary.row_dimensions[2].height = 40

    # Data rows
    HIGHLIGHT = {
        "Rank 1":        RANK1_FILL,
        "Gene NOT found": NOTFOUND_FILL,
    }
    for row_i, metric in enumerate(METRICS, start=3):
        # alternating row shading
        row_fill = ALT_FILL if row_i % 2 == 0 else None

        mc = ws_summary.cell(row=row_i, column=1, value=metric)
        style_cell(mc, fill=row_fill, bold=True)

        for col_i, name in enumerate(model_names, start=2):
            val  = all_stats[name].get(metric, "—")
            cell = ws_summary.cell(row=row_i, column=col_i, value=val)
            fill = HIGHLIGHT.get(metric, row_fill)
            style_cell(cell, fill=fill, center=True)

    # Column widths
    ws_summary.column_dimensions["A"].width = 28
    for col in range(2, 2 + len(MODELS)):
        ws_summary.column_dimensions[get_column_letter(col)].width = 22

    # ── PER-MODEL DETAIL SHEETS ───────────────────────────────────────────────
    DETAIL_COLS = [
        ("file_name",          "File",              50),
        ("target_gene",        "Target Gene",       16),
        ("ai_rank",            "AI Rank",           10),
        ("ai_category",        "Category",          16),
        ("variants_processed", "Variants Sent",     14),
        ("phenotypes_found",   "HPO Terms",         12),
    ]

    CAT_FILL = {
        "Rank 1":        PatternFill("solid", fgColor="C6EFCE"),
        "Ranks 2-5":     PatternFill("solid", fgColor="FFEB9C"),
        "Ranks 6-10":    PatternFill("solid", fgColor="FFCC99"),
        "Ranks 11-50":   PatternFill("solid", fgColor="FFB3B3"),
        "Beyond rank 50":PatternFill("solid", fgColor="FF8080"),
        "Not Found":     PatternFill("solid", fgColor="FFCCCC"),
    }

    for display_name, _ in MODELS:
        records = all_model_records[display_name]
        safe    = re.sub(r'[\\/*?:\[\]]', '_', display_name)[:31]
        ws      = wb.create_sheet(title=safe)

        # Sheet title
        ws.merge_cells(f"A1:{get_column_letter(len(DETAIL_COLS))}1")
        t = ws["A1"]
        t.value     = f"No-RAG Results — {display_name}"
        t.font      = Font(bold=True, size=12, color="1F3864", name="Calibri")
        t.alignment = CENTER
        ws.row_dimensions[1].height = 25

        # Header
        for col_i, (_, header, width) in enumerate(DETAIL_COLS, start=1):
            c = ws.cell(row=2, column=col_i, value=header)
            style_header(c)
            ws.column_dimensions[get_column_letter(col_i)].width = width
        ws.row_dimensions[2].height = 20

        if not records:
            ws.cell(row=3, column=1, value="No successful results found for this model.")
            continue

        df = pd.DataFrame(records).sort_values("file_name")
        for row_i, (_, row) in enumerate(df.iterrows(), start=3):
            cat  = row["ai_category"]
            fill = CAT_FILL.get(cat)
            for col_i, (key, _, _) in enumerate(DETAIL_COLS, start=1):
                val  = row.get(key, "")
                cell = ws.cell(row=row_i, column=col_i, value=val)
                cell.border    = thin_border()
                cell.alignment = CENTER if col_i > 1 else LEFT
                cell.font      = NORMAL
                if fill and col_i in (3, 4):   # colour only rank + category cols
                    cell.fill = fill

        # Freeze header
        ws.freeze_panes = "A3"

    # ── save ─────────────────────────────────────────────────────────────────
    ws_summary.freeze_panes = "A3"
    wb.save(OUTPUT_FILE)
    print(f"\n  ✅ Saved: {OUTPUT_FILE}")
    print("=" * 65)


if __name__ == "__main__":
    main()
