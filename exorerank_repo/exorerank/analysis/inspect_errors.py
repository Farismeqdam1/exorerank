#!/usr/bin/env python3
"""Peek at the actual error messages in failed JSON files."""
import json
from pathlib import Path
from collections import Counter

OUTPUT_ROOT = "/home/faris/reranking_norag"

FOLDERS = [
    "Llama-3.3-70B",
    "openai_gpt-oss-120b",
    "openai_gpt-oss-20b",
    "GPT_4.1",
    "kimi-k2-instruct-0905",
]

for folder in FOLDERS:
    out_dir = Path(OUTPUT_ROOT) / folder
    if not out_dir.exists():
        print(f"\n❌ Folder not found: {folder}")
        continue

    failed = []
    for jf in sorted(out_dir.glob("*_norag.json")):
        try:
            data = json.loads(jf.read_text())
            if not (data.get("status") == "success" and data.get("ai_ranking")):
                failed.append((jf.name, data.get("error", data.get("status", "?"))))
        except Exception as e:
            failed.append((jf.name, f"JSON parse error: {e}"))

    if not failed:
        continue

    # Count unique error patterns
    errors = Counter()
    for _, err in failed:
        # Truncate to first 120 chars to group similar errors
        errors[str(err)[:120]] += 1

    print(f"\n{'='*65}")
    print(f"  {folder}  —  {len(failed)} failed files")
    print(f"{'='*65}")
    print("  Error breakdown:")
    for msg, count in errors.most_common():
        print(f"    [{count}x]  {msg}")
    print()
    # Show first failed file details
    print(f"  First failed file: {failed[0][0]}")
    print(f"  Full error: {failed[0][1]}")
