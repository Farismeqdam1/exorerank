#!/usr/bin/env python3
"""
check_progress.py  —  Audit No-RAG reranking results
======================================================
Shows for each model:
  • How many files succeeded / failed / still missing
  • Which specific files failed or are missing
  • A summary table at the end

Usage:
    python check_progress.py
    python check_progress.py --model "qwen3-32b"   # one model only
    python check_progress.py --failed-only          # only show problems
"""

import os
import json
import argparse
from pathlib import Path

OUTPUT_ROOT = "/home/faris/reranking_norag"
HTML_DIR    = "/home/faris/exomiser-cli-14.1.0-distribution_faster/exomiser-cli-14.1.0/hporesult"

# Must match MODELS list in batch_rerank_norag.py
MODEL_FOLDERS = [
    "llama-3.1-8b-instant",
    "Llama-3.3-70B",
    "qwen3-32b",
    "llama-4-scout-17b-16e",
    "openai_gpt-oss-120b",
    "openai_gpt-oss-20b",
    "GPT_4.1",
    "kimi-k2-instruct-0905",
]

def check_model(folder_name: str, html_files: list, failed_only: bool) -> dict:
    out_dir = Path(OUTPUT_ROOT) / folder_name

    done = failed = missing = 0
    failed_files = []
    missing_files = []

    for html in html_files:
        json_path = out_dir / (html.stem + "_norag.json")

        if not json_path.exists():
            missing += 1
            missing_files.append(html.name)
            continue

        try:
            data = json.loads(json_path.read_text())
            if data.get("status") == "success" and data.get("ai_ranking"):
                done += 1
            else:
                failed += 1
                failed_files.append((html.name, data.get("error", data.get("status", "?"))))
        except Exception as e:
            failed += 1
            failed_files.append((html.name, f"JSON parse error: {e}"))

    total = len(html_files)
    pct   = (done / total * 100) if total else 0

    bar_len = 30
    filled  = int(bar_len * done / total) if total else 0
    bar     = "█" * filled + "░" * (bar_len - filled)

    # ── print ─────────────────────────────────────────────────────────────────
    if not failed_only or (failed + missing) > 0:
        print(f"\n{'─'*65}")
        print(f"  📁  {folder_name}")
        print(f"  [{bar}]  {done}/{total}  ({pct:.1f}% done)")
        print(f"  ✅ Success: {done}   ❌ Failed: {failed}   ⬜ Missing: {missing}")

        if failed_files:
            print(f"\n  ── Failed files ({failed}) ──")
            for fname, reason in failed_files:
                print(f"    ✗  {fname}")
                print(f"       reason: {reason}")

        if missing_files:
            print(f"\n  ── Not yet processed ({missing}) ──")
            for fname in missing_files[:20]:         # cap at 20 lines
                print(f"    ○  {fname}")
            if missing > 20:
                print(f"    … and {missing - 20} more")

    return {"model": folder_name, "total": total, "done": done,
            "failed": failed, "missing": missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",       default=None, help="Check one model folder only")
    ap.add_argument("--failed-only", action="store_true",
                    help="Only print models that have failures or missing files")
    args = ap.parse_args()

    html_files = sorted(Path(HTML_DIR).glob("*.html"))
    if not html_files:
        print(f"❌  No HTML files found in: {HTML_DIR}")
        return

    folders = [args.model] if args.model else MODEL_FOLDERS
    rows = []

    for folder in folders:
        rows.append(check_model(folder, html_files, args.failed_only))

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  SUMMARY   ({len(html_files)} total HTML files)")
    print(f"{'='*65}")
    print(f"  {'Model':<35} {'Done':>5} {'Fail':>5} {'Miss':>5}  {'%':>6}")
    print(f"  {'─'*35} {'─'*5} {'─'*5} {'─'*5}  {'─'*6}")
    for r in rows:
        pct = r['done'] / r['total'] * 100 if r['total'] else 0
        flag = "✅" if r['failed'] == 0 and r['missing'] == 0 else "⚠️ "
        print(f"  {flag} {r['model']:<33} {r['done']:>5} {r['failed']:>5} "
              f"{r['missing']:>5}  {pct:>5.1f}%")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
