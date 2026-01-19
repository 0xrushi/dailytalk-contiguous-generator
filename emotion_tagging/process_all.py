#!/usr/bin/env python3
"""
Batch Emotion-Tagged Transcript Generator

Processes all WAV files in a directory and generates a single transcript.jsonl file.
Uses audEERING VAD (Valence/Arousal/Dominance) + Prosody features to generate deterministic Fish Audio tags.
"""

import argparse
import json
import sys
import os
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to import local modules
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import helpers
try:
    from tag_emotions import (
        load_silero_vad,
        prepare_audio,
        segment_audio_vad,
        transcribe_segment
    )
    import vad_model_audeering
    import vad_emotion_mapper
except ImportError:
    from .tag_emotions import (
        load_silero_vad,
        prepare_audio,
        segment_audio_vad,
        transcribe_segment
    )
    from . import vad_model_audeering
    from . import vad_emotion_mapper

from faster_whisper import WhisperModel


def process_all_wavs(input_dir, output_jsonl, channel="left", language="en", device="auto"):
    """
    Process all WAV files in input directory and generate single JSONL output.
    """
    input_dir = Path(input_dir)
    wav_files = sorted(input_dir.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} WAV files.")
    
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Load models
    print("Loading models...")
    print("  - Silero VAD")
    vad_model, get_speech_timestamps = load_silero_vad(device)
    
    print("  - Whisper (Medium)")
    whisper_model = WhisperModel("medium", device=device, compute_type="float32")
    
    print("  - audEERING VAD (Emotion)")
    emotion_model = vad_model_audeering.AudeeringVAD()

    # Prepare output file
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing files to {output_path}...")
    segment_id_global = 0
    
    # Open file for writing
    with open(output_path, "w", encoding="utf-8") as f_out:
        for wav_path in tqdm(wav_files, desc="Processing Files"):
            try:
                # Prepare audio
                sr = 16000
                audio = prepare_audio(str(wav_path), channel=channel, target_sr=sr)
                
                # Segment audio
                segments = segment_audio_vad(audio, sr, vad_model, get_speech_timestamps)
                
                # Collect records for this file
                file_records = []
                
                for start, end in segments:
                    # Extract segment audio
                    segment_audio = audio[start:end]
                    
                    # Transcribe
                    transcript = transcribe_segment(segment_audio, sr, whisper_model, language)
                    if not transcript:
                        continue
                        
                    # Normalize raw transcript
                    text_raw = " ".join(transcript.split())
                    
                    # Predict VAD (Valence, Arousal, Dominance)
                    vad_scores = emotion_model.predict(segment_audio, sr)
                    
                    # Compute prosody features
                    start_sec = start / sr
                    end_sec = end / sr
                    duration_sec = end_sec - start_sec
                    
                    words_count = len(text_raw.split())
                    
                    rms = vad_emotion_mapper.compute_rms(segment_audio)
                    zcr = vad_emotion_mapper.compute_zcr(segment_audio)
                    f0_mean, f0_std = vad_emotion_mapper.compute_pitch_stats(segment_audio, sr)
                    speech_rate = vad_emotion_mapper.compute_speech_rate(words_count, duration_sec)

                    features = {
                        "rms": rms,
                        "zcr": zcr,
                        "f0_mean": f0_mean,
                        "f0_std": f0_std,
                        "speech_rate": speech_rate
                    }
                    
                    # Store record
                    record = {
                        "segment_id": segment_id_global,
                        "audio_file": wav_path.name,
                        "channel": channel,
                        "start": round(start_sec, 3),
                        "end": round(end_sec, 3),
                        "duration": round(duration_sec, 3),
                        "vad": vad_scores,
                        "features": features,
                        "text_raw": text_raw,
                        "text": text_raw, # Initial text, will be punctuated later
                        "audio_path": str(wav_path) 
                    }
                    file_records.append(record)
                    segment_id_global += 1
                
                # Assign Fish tags (batch process for percentiles)
                if file_records:
                    tagged_records = vad_emotion_mapper.assign_fish_tags_from_vad(file_records)
                    
                    for rec in tagged_records:
                        # Write to output
                        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        f_out.flush()
            
            except Exception as e:
                print(f"ERROR processing {wav_path.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    print(f"Done! Generated {segment_id_global} segments.")


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate emotion-tagged transcripts using VAD + Prosody"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing WAV files to process"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--channel",
        choices=["left", "right", "mix"],
        default="left",
        help="Audio channel to process (default: left)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language hint (default: en)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: 'cpu', 'cuda', or 'auto' (default)"
    )
    
    args = parser.parse_args()
    
    process_all_wavs(
        input_dir=args.input_dir,
        output_jsonl=args.output,
        channel=args.channel,
        language=args.language,
        device=args.device
    )


if __name__ == "__main__":
    main()