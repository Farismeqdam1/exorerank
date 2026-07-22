#!/usr/bin/env python3
"""
diagnose_keys.py  —  Show exactly what API keys were found and where
"""
import os, re
from pathlib import Path

SEARCH_PATHS = [
    Path.cwd() / "secrets.toml",
    Path.cwd() / ".env",
    Path(__file__).parent / "secrets.toml",
    Path(__file__).parent / ".env",
    Path("/home/faris/.streamlit/secrets.toml"),
    Path("/home/faris/Downloads/DreamJournal/.env"),
    Path("/home/faris/groq-prompt-optimizer/.env"),
    Path.home() / ".streamlit" / "secrets.toml",
    Path.home() / ".env",
    Path.home() / "secrets.toml",
]

TARGET_KEYS = ["GROQ_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]

print("\n=== File scan ===")
found_keys = {}

for path in SEARCH_PATHS:
    if not path.exists():
        print(f"  ✗ {path}")
        continue
    print(f"  ✓ {path}  —  reading...")
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            if "=" not in line: continue
            k, _, v = line.partition("=")
            k = k.strip().upper()
            v = v.strip().strip('"').strip("'")
            if k in TARGET_KEYS and v and k not in found_keys:
                found_keys[k] = (v, str(path))
                print(f"      → {k} = {v[:10]}…  ✅")
    except Exception as e:
        print(f"      ERROR reading: {e}")

print("\n=== Environment variables ===")
for key in TARGET_KEYS:
    val = os.getenv(key, "")
    if val:
        print(f"  {key} = {val[:10]}…  ✅")
        if key not in found_keys:
            found_keys[key] = (val, "environment")
    else:
        print(f"  {key}  ✗ not set")

print("\n=== Final result ===")
for key in TARGET_KEYS:
    if key in found_keys:
        val, src = found_keys[key]
        print(f"  ✅ {key:<25} found in: {src}")
    else:
        print(f"  ❌ {key:<25} NOT FOUND anywhere")
print()
