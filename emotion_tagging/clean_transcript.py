#!/usr/bin/env python3
"""
Clean transcript_shifted.jsonl by removing entries for deleted audio files.
"""

import json
from pathlib import Path


def main():
    # Read deleted files list
    deleted_file = Path(__file__).parent.parent / "deleted_audio_bases.txt"
    if not deleted_file.exists():
        print(f"Error: {deleted_file} not found")
        return

    with open(deleted_file, 'r') as f:
        deleted_bases = set(line.strip() for line in f if line.strip())

    print(f"Found {len(deleted_bases)} deleted audio files")

    # Read transcript
    transcript_file = Path(__file__).parent.parent / "transcript_shifted.jsonl"
    if not transcript_file.exists():
        print(f"Error: {transcript_file} not found")
        return

    entries = []
    removed_count = 0

    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                audio_file = entry.get('audio_file', '')

                # Extract base name (e.g., "718_edited.wav" -> "718")
                base = audio_file.replace('_edited.wav', '').replace('.wav', '')

                if base in deleted_bases:
                    removed_count += 1
                    continue

                entries.append(line.strip())

    print(f"Original entries: {len(entries) + removed_count}")
    print(f"Removed entries: {removed_count}")
    print(f"Remaining entries: {len(entries)}")

    # Write back cleaned transcript
    with open(transcript_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(entry + '\n')

    print(f"\nCleaned transcript written to: {transcript_file}")


if __name__ == "__main__":
    main()
