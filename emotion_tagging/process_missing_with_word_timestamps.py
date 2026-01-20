#!/usr/bin/env python3
"""
Process Missing Audio Files with TTS and Word-Level Timestamps

This script:
1. Reads missing entries from transcript.jsonl
2. Processes each missing audio file with TTS reconstruction
3. Generates word-level timestamps using Whisper
4. Updates transcript_shifted.jsonl with aligned timestamps
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import Resample
from faster_whisper import WhisperModel

# Add parent directory to path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from tag_emotions import load_silero_vad, segment_audio_vad, prepare_audio
except ImportError:
    from .tag_emotions import load_silero_vad, segment_audio_vad, prepare_audio

FISH_TTS_URL = "https://api.fish.audio/v1/tts"

TAG_MAPPING = {
    "in a hurry tone": "excited",
    "disappointed": "sad",
    "uncertain": "worried",
    "determined": "confident",
    "relaxed": "calm"
}


def generate_fish_tts(text, tag, token, temperature=0.7, top_p=0.7):
    """Generate TTS audio using Fish Audio API."""
    try:
        tag = TAG_MAPPING.get(tag, tag)
        
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
            if not audio_content:
                return None, "Empty audio content"
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_content)
                tmp_path = tmp.name
            
            audio_data, sr = sf.read(tmp_path, dtype='float32')
            os.unlink(tmp_path)
            
            if len(audio_data) == 0:
                return None, "No audio samples"
            
            if sr != 16000:
                audio_tensor = torch.from_numpy(audio_data).float()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                resampler = Resample(sr, 16000, dtype=torch.float32)
                audio_tensor = resampler(audio_tensor)
                audio_data = audio_tensor.squeeze().numpy()
            
            return audio_data.astype(np.float32), None
        else:
            return None, f"API error: {response.status_code}"
    
    except Exception as e:
        return None, f"Exception: {e}"


def get_word_timestamps(audio, sr, whisper_model):
    """
    Get word-level timestamps using Whisper with word_timestamps=True.
    """
    try:
        segments, info = whisper_model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            beam_size=5
        )
        
        word_timestamps = []
        for segment in segments:
            if hasattr(segment, 'words'):
                for word in segment.words:
                    word_timestamps.append({
                        'word': word.word,
                        'start': word.start,
                        'end': word.end,
                        'probability': word.probability
                    })
        
        return word_timestamps, None
    except Exception as e:
        return [], f"Whisper error: {e}"


def process_missing_files(transcript_jsonl, shifted_jsonl, audio_dir, missing_files_txt, token,
                      output_jsonl, temperature=0.7, top_p=0.7):
    """
    Process missing audio files with TTS and word timestamps
    """
    transcript_path = Path(transcript_jsonl)
    shifted_path = Path(shifted_jsonl)
    audio_path = Path(audio_dir)
    missing_files_path = Path(missing_files_txt)
    
    # Load missing audio files list
    print(f"Reading {missing_files_path}...")
    missing_audio_files = set()
    with open(missing_files_path, 'r') as f:
        for line in f:
            if line.strip():
                missing_audio_files.add(line.strip())
    
    print(f"  Missing audio files to process: {len(missing_audio_files)}")
    
    # Load existing entries
    print(f"Reading {shifted_path}...")
    existing_entries = []
    existing_dict = {}
    with open(shifted_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line.strip())
                existing_entries.append(entry)
                # Create multiple keys for matching (by audio_file)
                existing_dict[entry['audio_file']] = entry
    
    print(f"  Existing entries: {len(existing_entries)}")
    
    # Load all transcript entries
    print(f"Reading {transcript_path}...")
    all_entries = []
    with open(transcript_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_entries.append(json.loads(line.strip()))
    
    print(f"  Total entries: {len(all_entries)}")
    
    # Find entries to reprocess (matching missing audio files)
    entries_to_reprocess = [e for e in all_entries if e['audio_file'] in missing_audio_files]
    
    if len(entries_to_reprocess) == 0:
        print("No entries found to reprocess.")
        return True
    
    print(f"\nEntries to reprocess: {len(entries_to_reprocess)}")
    
    print(f"\nSample entries to reprocess:")
    for i, e in enumerate(entries_to_reprocess[:5]):
        print(f"  {e['audio_file']}: segment_id={e.get('segment_id', 'N/A')}, text=\"{e['text'][:40]}...\"")
    
    # Group entries to reprocess by audio file
    entries_by_file = {}
    for entry in entries_to_reprocess:
        af = entry['audio_file']
        if af not in entries_by_file:
            entries_by_file[af] = []
        entries_by_file[af].append(entry)
    
    # Build a dict of audio_file -> existing entries for replacement
    existing_by_audio_file = {}
    for entry in existing_entries:
        af = entry['audio_file']
        if af not in existing_by_audio_file:
            existing_by_audio_file[af] = []
        existing_by_audio_file[af].append(entry)
    
    # Load VAD for right channel detection
    print("\nLoading VAD model...")
    vad_model, get_speech_timestamps = load_silero_vad()
    
    # Load Whisper model for word timestamps
    print("Loading Whisper model (small) for word timestamps...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = WhisperModel("small", device=device, compute_type="float32")
    
    # Process each audio file
    new_entries = []
    processed_files = []
    
    for audio_file in sorted(entries_by_file.keys()):
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file}")
        print(f"{'='*60}")
        
        left_entries = entries_by_file[audio_file]
        wav_path = audio_path / audio_file
        
        if not wav_path.exists():
            print(f"  Skipping: {wav_path} not found")
            continue
        
        # Load original audio
        try:
            original_audio, sr = torchaudio.load(str(wav_path))
        except Exception as e:
            print(f"  Error loading: {e}")
            continue
        
        if original_audio.shape[0] < 2:
            print("  Warning: Not stereo")
            right_audio = torch.zeros(1, original_audio.shape[1])
        else:
            right_audio = original_audio[1:2, :]
        
        # Detect right channel segments
        right_audio_np = right_audio.squeeze().numpy()
        if sr != 16000:
            resampler = Resample(sr, 16000)
            right_audio_16k = resampler(right_audio).squeeze().numpy()
        else:
            right_audio_16k = right_audio_np
        
        if np.max(np.abs(right_audio_16k)) > 1.0:
            right_audio_16k = right_audio_16k / np.max(np.abs(right_audio_16k))
        
        right_segments_vad = segment_audio_vad(right_audio_16k, 16000, vad_model, get_speech_timestamps)
        
        scale = sr / 16000
        right_segments = []
        for s, e in right_segments_vad:
            right_segments.append({
                "start": int(s * scale),
                "end": int(e * scale),
                "type": "right"
            })
        
        # Build events list
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
                "data": None
            })
        
        events.sort(key=lambda x: x["start"])
        
        # Reconstruct with TTS
        out_left = []
        out_right = []
        
        current_sample = 0
        last_original_end = 0
        file_new_entries = []
        
        for event in events:
            original_gap = event["start"] - last_original_end
            if original_gap < 0:
                original_gap = 0
            
            if original_gap > 0:
                silence = np.zeros(original_gap, dtype=np.float32)
                out_left.append(silence)
                out_right.append(silence)
                current_sample += original_gap
            
            if event["type"] == "left":
                entry = event["data"]
                
                # Remap tag
                raw_tag = entry["tag"]
                mapped_tag = TAG_MAPPING.get(raw_tag, raw_tag)
                text_to_gen = f"({mapped_tag}) {entry['text'].strip()}"
                
                print(f"  Segment {entry.get('segment_id', 'N/A')}: {text_to_gen[:60]}...")
                
                # Generate TTS
                tts_audio, error = generate_fish_tts(text_to_gen, mapped_tag, token, temperature, top_p)
                
                if tts_audio is None:
                    print(f"  Error generating TTS: {error}")
                    # Use original audio as fallback
                    start = max(0, int(entry["start"] * sr))
                    end = min(int(entry["end"] * sr), original_audio.shape[1])
                    if end > start:
                        tts_audio = original_audio[0, start:end].numpy()
                    else:
                        tts_audio = np.zeros(100, dtype=np.float32)
                else:
                    print(f"  TTS generated: {len(tts_audio)} samples ({len(tts_audio)/sr:.2f}s)")
                    
                    if sr != 16000:
                        t_tensor = torch.from_numpy(tts_audio).float()
                        if t_tensor.ndim == 1:
                            t_tensor = t_tensor.unsqueeze(0)
                        resampler_to_sr = Resample(16000, sr, dtype=torch.float32)
                        tts_audio = resampler_to_sr(t_tensor).squeeze().numpy()
                
                length = len(tts_audio)
                
                out_left.append(tts_audio)
                out_right.append(np.zeros(length, dtype=np.float32))
                
                # Get word-level timestamps from TTS audio
                word_timestamps, error = get_word_timestamps(tts_audio, sr, whisper_model)
                if error:
                    print(f"  Word timestamp error: {error}")
                    word_timestamps = []
                else:
                    print(f"  Word timestamps: {len(word_timestamps)} words")
                
                # Record new timing
                new_entry = entry.copy()
                new_entry["start"] = round(current_sample / sr, 3)
                new_entry["end"] = round((current_sample + length) / sr, 3)
                new_entry["duration"] = round(length / sr, 3)
                new_entry["tag"] = mapped_tag
                new_entry["tagged_text"] = text_to_gen
                new_entry["word_timestamps"] = word_timestamps
                file_new_entries.append(new_entry)
                
                current_sample += length
                
            else:  # right channel
                start = event["start"]
                end = event["end"]
                start = max(0, min(start, right_audio.shape[1]))
                end = max(0, min(end, right_audio.shape[1]))
                
                if end > start:
                    audio_chunk = right_audio[0, start:end].numpy()
                    length = len(audio_chunk)
                    
                    out_right.append(audio_chunk)
                    out_left.append(np.zeros(length, dtype=np.float32))
                    
                    current_sample += length
            
            last_original_end = event["end"]
        
        # Save reconstructed audio
        if out_left:
            full_left = np.concatenate(out_left)
            full_right = np.concatenate(out_right)
            
            max_len = max(len(full_left), len(full_right))
            if len(full_left) < max_len:
                full_left = np.pad(full_left, (0, max_len - len(full_left)))
            if len(full_right) < max_len:
                full_right = np.pad(full_right, (0, max_len - len(full_right)))
            
            final_stereo = np.stack([full_left, full_right])
            final_tensor = torch.from_numpy(final_stereo)
            
            out_filename = wav_path.stem + "_edited" + wav_path.suffix
            out_path = audio_path / out_filename
            print(f"\n  Saving reconstructed: {out_filename}")
            print(f"  Duration: {len(full_left)/sr:.2f}s")
            torchaudio.save(str(out_path), final_tensor, sr)
            
            new_entries.extend(file_new_entries)
            processed_files.append(audio_file)
    
    # Replace existing entries with new ones (by audio_file)
    # Keep all other existing entries unchanged
    audio_files_processed = set(processed_files)
    kept_entries = []
    replaced_count = 0
    
    for entry in existing_entries:
        if entry['audio_file'] in audio_files_processed:
            # Skip this entry, will be replaced with new one
            replaced_count += 1
        else:
            kept_entries.append(entry)
    
    # Combine kept entries + new processed entries
    combined_entries = kept_entries + new_entries
    
    print(f"\nReplaced {replaced_count} entries with TTS-processed versions")
    
    # Write to output
    print(f"\n{'='*60}")
    print(f"Writing {len(combined_entries)} entries to {output_jsonl}")
    print(f"{'='*60}")
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in combined_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*60}")
    print(f"Processed {len(processed_files)} audio files")
    print(f"Replaced {replaced_count} entries with TTS-processed versions")
    print(f"Added {len(new_entries)} new entries with word timestamps")
    print("Done!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Process missing files with TTS and word timestamps")
    parser.add_argument("--transcript", required=True, help="Path to transcript.jsonl")
    parser.add_argument("--shifted", required=True, help="Path to transcript_shifted.jsonl")
    parser.add_argument("--audio-dir", required=True, help="Directory containing WAV files")
    parser.add_argument("--missing-files", required=True, help="Path to missing_audio_files.txt")
    parser.add_argument("--token", required=True, help="Fish Audio API token")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.7)
    
    args = parser.parse_args()
    
    success = process_missing_files(
        args.transcript,
        args.shifted,
        args.audio_dir,
        args.missing_files,
        args.token,
        args.output,
        args.temperature,
        args.top_p
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    main()
