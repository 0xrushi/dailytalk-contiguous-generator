#!/usr/bin/env python3
"""
TTS Audio Generator and Conversation Reconstructor

Reads transcript.jsonl, generates TTS audio for Left channel segments.
Detects Right channel segments using VAD.
Reconstructs the conversation timeline sequentially to accommodate TTS duration changes,
preventing overlaps and ensuring "one starts after another" where appropriate.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
import requests

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import Resample

# Add parent directory to path to import local modules
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from tag_emotions import load_silero_vad, segment_audio_vad, prepare_audio
except ImportError:
    from .tag_emotions import load_silero_vad, segment_audio_vad, prepare_audio

# Fish Audio TTS API endpoint
FISH_TTS_URL = "https://api.fish.audio/v1/tts"

# Tag remapping for legacy/unsupported tags
TAG_MAPPING = {
    "in a hurry tone": "excited",
    "disappointed": "sad",
    "uncertain": "worried",
    "determined": "confident",
    "relaxed": "calm"
}

def generate_fish_tts(text, tag, token, temperature=0.7, top_p=0.7):
    """
    Generate TTS audio using Fish Audio API.
    """
    try:
        # Remap tag if needed
        tag = TAG_MAPPING.get(tag, tag)
        
        # Prepare payload for Fish TTS API
        # Request 32kHz as 16kHz might not be supported directly for mp3 output by the API
        payload = {
            "text": text,
            "language": "en",
            "sample_rate": 32000,
            "reference_id": os.getenv("FISH_AUDIO_REFERENCE_ID"),
            "temperature": temperature,
            "top_p": top_p
        }
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "model": "s1",
        }
        
        response = requests.post(FISH_TTS_URL, json=payload, headers=headers, timeout=60)
        
        if response.status_code == 200:
            audio_content = response.content
            print(f"    TTS Content Len: {len(audio_content)}")
            if not audio_content:
                print("  TTS API returned empty content.")
                samples = int(len(text.split()) * 0.2 * 16000)
                return np.zeros(samples, dtype=np.float32)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_content)
                tmp_path = tmp.name
            
            # Load as float32
            try:
                audio_data, sr = sf.read(tmp_path, dtype='float32')
            except Exception as e:
                print(f"  Error reading wav: {e}")
                os.unlink(tmp_path)
                samples = int(len(text.split()) * 0.2 * 16000)
                return np.zeros(samples, dtype=np.float32)

            os.unlink(tmp_path)
            
            if len(audio_data) == 0:
                 print("  TTS API returned valid wav but no samples.")
                 samples = int(len(text.split()) * 0.2 * 16000)
                 return np.zeros(samples, dtype=np.float32)

            if sr != 16000:
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                resampler = Resample(sr, 16000, dtype=torch.float32)
                audio_tensor = resampler(audio_tensor)
                audio_data = audio_tensor.squeeze().numpy()
            
            return audio_data.astype(np.float32)
        else:
            print(f"  TTS API error: {response.status_code} - {response.text}")
            samples = int(len(text.split()) * 0.2 * 16000)
            return np.zeros(samples, dtype=np.float32)
    
    except Exception as e:
        print(f"  TTS API exception: {e}")
        samples = int(len(text.split()) * 0.2 * 16000)
        return np.zeros(samples, dtype=np.float32)


def process_and_reconstruct(transcript_jsonl, output_dir, token, output_transcript, temperature=0.7, top_p=0.7, resume=False):
    """
    Reconstruct conversation with TTS and dynamic timing.
    
    NOTE: This function now REPLACES entries in output_transcript instead of appending.
    When an audio file is re-processed, its corresponding entries are updated in-place.
    """
    transcript_path = Path(transcript_jsonl)
    output_path = Path(output_dir)
    
    # Load existing transcript entries for deduplication
    existing_transcript_entries = {}
    if Path(output_transcript).exists():
        with open(output_transcript, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    key = f"{entry.get('audio_file')}_{entry.get('tagged_text', '')}_{entry.get('start', 0)}"
                    existing_transcript_entries[key] = entry

    # Load VAD
    print("Loading VAD model...")
    vad_model, get_speech_timestamps = load_silero_vad()

    # Read transcript
    entries = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    print(f"Loaded {len(entries)} transcript entries")

    entries_by_file = {}
    for entry in entries:
        af = entry["audio_file"]
        if af not in entries_by_file:
            entries_by_file[af] = []
        entries_by_file[af].append(entry)

    # Resume logic: skip already processed files
    if resume:
        # Get all unique files in order
        unique_files = list(entries_by_file.keys())

        # Find which files already have _edited.wav versions
        files_to_process = []
        skipped_count = 0
        for audio_file in unique_files:
            original_wav_path = output_path / audio_file
            edited_wav_path = output_path / f"{original_wav_path.stem}_edited{original_wav_path.suffix}"
            if edited_wav_path.exists():
                skipped_count += 1
                print(f"  Skipping completed file: {audio_file}")
            else:
                files_to_process.append(audio_file)

        print(f"  Resume mode: skipping {skipped_count} already processed files")
        print(f"  Will process {len(files_to_process)} remaining files")
    else:
        files_to_process = list(entries_by_file.keys())

        # Clear output transcript only if not resuming
        if output_transcript:
            with open(output_transcript, "w", encoding="utf-8") as f:
                pass
    
    for audio_file in files_to_process:
        left_entries = entries_by_file[audio_file]

        print(f"\nProcessing: {audio_file}")

        original_wav_path = output_path / audio_file
        
        if not original_wav_path.exists():
            print(f"  Missing: {original_wav_path}")
            continue
            
        # Load original audio
        try:
            original_audio, sr = torchaudio.load(str(original_wav_path))
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

        if original_audio.shape[0] < 2:
            print("  Warning: Not stereo. Treating as single channel replacement.")
            # If mono, we can't really "interleave" with right channel. 
            # We'll just process left entries.
            right_audio = torch.zeros(1, original_audio.shape[1])
        else:
            right_audio = original_audio[1:2, :]
        
        # 1. Detect Right Channel Segments
        print("  Detecting right channel segments...")
        # Prepare right audio for VAD (mono, 16k)
        right_audio_np = right_audio.squeeze().numpy()
        if sr != 16000:
            # Quick resample for VAD
             resampler = Resample(sr, 16000)
             right_audio_16k = resampler(right_audio).squeeze().numpy()
        else:
             right_audio_16k = right_audio_np
             
        # Normalize
        if np.max(np.abs(right_audio_16k)) > 1.0:
            right_audio_16k = right_audio_16k / np.max(np.abs(right_audio_16k))
            
        right_segments_vad = segment_audio_vad(right_audio_16k, 16000, vad_model, get_speech_timestamps)
        
        # Convert VAD 16k samples back to original SR samples
        right_segments = []
        scale = sr / 16000
        for s, e in right_segments_vad:
            right_segments.append({
                "start": int(s * scale),
                "end": int(e * scale),
                "type": "right"
            })
            
        # 2. Merge Events
        events = []
        for entry in left_entries:
            events.append({
                "start": int(entry["start"] * sr),
                "end": int(entry["end"] * sr),
                "type": "left",
                "data": entry
            })
        
        for seg in right_segments:
            events.append({
                "start": seg["start"],
                "end": seg["end"],
                "type": "right",
                "data": None # No transcript data for right
            })
            
        # Sort by original start time
        events.sort(key=lambda x: x["start"])
        
        # 3. Reconstruct
        out_left = []
        out_right = []
        
        current_sample = 0
        last_original_end = 0
        file_new_transcript_entries = []
        
        for event in events:
            # Calculate gap from previous event
            original_gap = event["start"] - last_original_end
            
            # Enforce sequentiality: if gap is negative (overlap), make it 0
            if original_gap < 0:
                original_gap = 0
            
            # Add silence for the gap
            if original_gap > 0:
                silence = np.zeros(original_gap, dtype=np.float32)
                out_left.append(silence)
                out_right.append(silence)
                current_sample += original_gap
            
            if event["type"] == "left":
                # Generate TTS
                entry = event["data"]
                
                # Remap tag for text construction
                raw_tag = entry["tag"]
                mapped_tag = TAG_MAPPING.get(raw_tag, raw_tag)
                
                # Use tagged_text with space: (tag) text
                text_to_gen = f"({mapped_tag}) {entry['text'].strip()}"
                print(f"    Gen Text: {text_to_gen}")
                tts_audio_16k = generate_fish_tts(text_to_gen, mapped_tag, token, temperature, top_p)
                
                # Resample to sr
                if sr != 16000:
                    t_tensor = torch.from_numpy(tts_audio_16k).float()
                    if t_tensor.ndim == 1: t_tensor = t_tensor.unsqueeze(0)
                    resampler_to_sr = Resample(16000, sr, dtype=torch.float32)
                    tts_audio = resampler_to_sr(t_tensor).squeeze().numpy()
                else:
                    tts_audio = tts_audio_16k
                
                length = len(tts_audio)
                
                out_left.append(tts_audio)
                out_right.append(np.zeros(length, dtype=np.float32))
                
                # Record new timing for transcript
                new_entry = entry.copy()
                new_entry["start"] = round(current_sample / sr, 3)
                new_entry["end"] = round((current_sample + length) / sr, 3)
                new_entry["duration"] = round(length / sr, 3)
                new_entry["tag"] = mapped_tag # Save the mapped tag
                new_entry["tagged_text"] = text_to_gen
                file_new_transcript_entries.append(new_entry)
                
                current_sample += length
                
            else: # right
                # Extract original audio
                start = event["start"]
                end = event["end"]
                # Ensure bounds
                start = max(0, min(start, right_audio.shape[1]))
                end = max(0, min(end, right_audio.shape[1]))
                
                if end > start:
                    audio_chunk = right_audio[0, start:end].numpy()
                    length = len(audio_chunk)
                    
                    out_right.append(audio_chunk)
                    out_left.append(np.zeros(length, dtype=np.float32))
                    
                    current_sample += length
            
            last_original_end = event["end"]
            
        # Concatenate
        if out_left:
            full_left = np.concatenate(out_left)
            full_right = np.concatenate(out_right)
            
            # Ensure equal length (float precision issues?)
            max_len = max(len(full_left), len(full_right))
            if len(full_left) < max_len:
                full_left = np.pad(full_left, (0, max_len - len(full_left)))
            if len(full_right) < max_len:
                full_right = np.pad(full_right, (0, max_len - len(full_right)))
                
            final_stereo = np.stack([full_left, full_right])
            final_tensor = torch.from_numpy(final_stereo)
            
            out_filename = original_wav_path.stem + "_edited" + original_wav_path.suffix
            out_path = output_path / out_filename
            print(f"  Saving reconstructed: {out_filename}")
            torchaudio.save(str(out_path), final_tensor, sr)
            
            # Update transcript entries in memory
            if output_transcript and file_new_transcript_entries:
                for entry in file_new_transcript_entries:
                    key = f"{entry.get('audio_file')}_{entry.get('tagged_text', '')}_{entry.get('start', 0)}"
                    existing_transcript_entries[key] = entry
    
    # Write all updated transcript entries to file
    if output_transcript and existing_transcript_entries:
        print(f"\nWriting {len(existing_transcript_entries)} transcript entries to {output_transcript}")
        with open(output_transcript, "w", encoding="utf-8") as f:
            for entry in existing_transcript_entries.values():
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct conversation with Fish TTS")
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--output-transcript", default="transcript_shifted.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for TTS (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.7, help="Top P for TTS (default: 0.7)")
    parser.add_argument("--resume", action="store_true", help="Resume from last entry in output-transcript")

    args = parser.parse_args()

    process_and_reconstruct(
        args.transcript,
        args.output_dir,
        args.token,
        args.output_transcript,
        args.temperature,
        args.top_p,
        args.resume
    )

if __name__ == "__main__":
    main()
