#!/usr/bin/env python3
"""
cleanup_failed.py  —  Delete failed No-RAG JSON files so they get retried
==========================================================================
Removes any JSON file where status != "success" or ai_ranking is empty.
Safe to run anytime — it never touches successful files.

Usage:
    python cleanup_failed.py           # preview what would be deleted (dry run)
    python cleanup_failed.py --delete  # actually delete them
"""

import json
import argparse
from pathlib import Path

OUTPUT_ROOT = "/home/faris/reranking_norag"

# Only clean up models that have a fixable problem (rate limits, wrong IDs).
# Models with no API key at all are excluded — deleting them won't help until
# you add the key.
FOLDERS_TO_CLEAN = [
    "Llama-3.3-70B",          # rate limit failures
    "openai_gpt-oss-120b",    # old JSONs written with wrong groq routing — now fixed to openrouter
    "openai_gpt-oss-20b",     # same
    "GPT_4.1",                # key loading failures — clean so they retry once keys are confirmed
    "kimi-k2-instruct-0905",  # same
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true",
                    help="Actually delete the files (default is dry-run preview)")
    args = ap.parse_args()

    mode = "🗑️  DELETE" if args.delete else "🔍 DRY RUN (use --delete to actually remove)"
    print(f"\n{mode}\n{'='*60}")

    total_removed = 0

    for folder in FOLDERS_TO_CLEAN:
        out_dir = Path(OUTPUT_ROOT) / folder
        if not out_dir.exists():
            print(f"\n  ⬜ {folder}  — folder doesn't exist, skipping")
            continue

        to_delete = []
        kept = 0

        for jf in sorted(out_dir.glob("*_norag.json")):
            try:
                data = json.loads(jf.read_text())
                if data.get("status") == "success" and data.get("ai_ranking"):
                    kept += 1          # good file — leave it alone
                else:
                    to_delete.append(jf)
            except Exception:
                to_delete.append(jf)  # unreadable → delete

        print(f"\n  📁 {folder}")
        print(f"     ✅ Keeping  : {kept}")
        print(f"     🗑️  To delete: {len(to_delete)}")

        if args.delete:
            for jf in to_delete:
                jf.unlink()
            print(f"     ✔  Deleted {len(to_delete)} files")

        total_removed += len(to_delete)

    print(f"\n{'='*60}")
    if args.delete:
        print(f"  Done — {total_removed} failed files deleted.")
        print(f"  Re-run batch_rerank_norag.py to process them again.")
    else:
        print(f"  Would delete {total_removed} failed files.")
        print(f"  Run with --delete to apply:  python cleanup_failed.py --delete")
    print()


if __name__ == "__main__":
    main()
