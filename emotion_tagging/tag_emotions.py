#!/usr/bin/env python3
"""
Audio processing helpers for emotion tagging.
"""

import numpy as np
import torch
import torchaudio

def load_silero_vad(device="cpu"):
    """Load Silero VAD model."""
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False,
        onnx=False
    )
    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils
    
    model.to(device)
    model.eval()
    return model, get_speech_timestamps

def prepare_audio(audio_path, channel="mix", target_sr=16000):
    """
    Load audio file and convert to mono at target sample rate.
    
    Args:
        audio_path: Path to audio file
        channel: 'left', 'right', or 'mix' (default)
        target_sr: Target sample rate (default 16000)
    
    Returns:
        numpy array of mono audio at target_sr
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Handle stereo
    if waveform.shape[0] == 2:
        if channel == "left":
            waveform = waveform[0:1, :]
        elif channel == "right":
            waveform = waveform[1:2, :]
        else:  # mix
            waveform = waveform.mean(dim=0, keepdim=True)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to numpy float32 in range [-1, 1]
    audio = waveform.squeeze().numpy()
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if necessary
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    
    return audio

def segment_audio_vad(audio, sr, model, get_speech_timestamps):
    """
    Segment audio using Silero VAD.
    
    Args:
        audio: numpy array of audio samples
        sr: sample rate
        model: Silero VAD model
        get_speech_timestamps: VAD utils function
    
    Returns:
        List of (start_sample, end_sample) tuples
    """
    # Convert to torch tensor and move to model device
    device = next(model.parameters()).device
    audio_tensor = torch.from_numpy(audio).float().to(device)
    
    # Get speech timestamps
    speech_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=0.5,
        sampling_rate=sr,
        min_speech_duration_ms=250,
        min_silence_duration_ms=350,
        window_size_samples=512,
        speech_pad_ms=30,
        return_seconds=False
    )
    
    if not speech_timestamps:
        return []
    
    # Merge close segments
    segments = []
    current_start = speech_timestamps[0]['start']
    current_end = speech_timestamps[0]['end']
    max_gap_samples = int(0.35 * sr)  # 0.35 seconds gap
    
    for ts in speech_timestamps[1:]:
        if ts['start'] - current_end <= max_gap_samples:
            # Merge
            current_end = ts['end']
        else:
            segments.append((current_start, current_end))
            current_start = ts['start']
            current_end = ts['end']
    
    segments.append((current_start, current_end))
    
    # Filter short segments (< 0.8s)
    min_duration_samples = int(0.8 * sr)
    segments = [(s, e) for s, e in segments if e - s >= min_duration_samples]
    
    # Split long segments (> 12s)
    max_duration_samples = int(12.0 * sr)
    final_segments = []
    
    for start, end in segments:
        duration = end - start
        if duration <= max_duration_samples:
            final_segments.append((start, end))
        else:
            # Split into chunks
            num_chunks = int(np.ceil(duration / max_duration_samples))
            chunk_size = duration // num_chunks
            
            for i in range(num_chunks):
                chunk_start = start + i * chunk_size
                chunk_end = chunk_start + chunk_size if i < num_chunks - 1 else end
                final_segments.append((chunk_start, chunk_end))
    
    return final_segments

def transcribe_segment(audio, sr, model, language="en"):
    """
    Transcribe audio segment using faster-whisper.
    
    Args:
        audio: numpy array of audio samples
        sr: sample rate
        model: faster-whisper model
        language: language hint (default 'en')
    
    Returns:
        Transcript text or None if empty
    """
    segments, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=False,  # We already did VAD
        word_timestamps=False
    )
    
    transcript = " ".join([seg.text for seg in segments])
    transcript = transcript.strip()
    
    return transcript if transcript else None