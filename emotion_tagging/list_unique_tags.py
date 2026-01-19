#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

def list_unique_tags(jsonl_path):
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        print(f"Error: File {jsonl_path} does not exist.")
        return

    unique_tags = set()
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    tag = record.get("tag")
                    if tag:
                        unique_tags.add(tag)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not unique_tags:
        print("No tags found in the JSONL file.")
    else:
        print(f"Found {len(unique_tags)} unique tags:")
        for tag in sorted(unique_tags):
            print(f"- {tag}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List unique tags from a JSONL file.")
    parser.add_argument("input", help="Path to the transcript.jsonl file")
    args = parser.parse_args()
    list_unique_tags(args.input)
