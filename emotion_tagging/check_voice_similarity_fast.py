#!/usr/bin/env python3
"""
Fast Voice Similarity Checker for TTS-generated audio

Compares _edited.wav files against a reference voice file to detect
files that don't have the expected voice (wrong voice or mute).

This version uses simplified features for faster processing.
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import librosa
from scipy.spatial.distance import cosine


def load_audio(file_path, target_sr=16000):
    """Load audio file and convert to target sample rate."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        return None, None


def extract_fast_features(audio, sr=16000):
    """Extract simplified features for fast voice comparison."""
    if audio is None or len(audio) == 0:
        return None
    
    try:
        # MFCC features (first 13 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Spectral centroid and rolloff
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, spectral_rolloff, zcr]
        ])
        
        return features
    except Exception:
        return None


def calculate_similarity(features1, features2):
    """Calculate cosine similarity between feature vectors."""
    if features1 is None or features2 is None:
        return 0.0
    
    # Normalize features
    features1 = features1 / (np.linalg.norm(features1) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2) + 1e-8)
    
    # Calculate cosine similarity
    try:
        similarity = 1 - cosine(features1, features2)
        return max(0.0, min(1.0, similarity))
    except:
        return 0.0


def is_silent_audio(audio, threshold=0.01):
    """Check if audio is silent or very quiet."""
    if audio is None or len(audio) == 0:
        return True
    
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold


def process_file(reference_features, reference_energy, target_file, energy_threshold):
    """Process a single file and return results."""
    file_name = Path(target_file).name
    
    # Load target audio
    target_audio, target_sr = load_audio(target_file)
    
    if target_audio is None:
        return {
            'file': file_name,
            'similarity': 0.0,
            'energy': 0.0,
            'status': 'load_error'
        }
    
    # Check if silent
    rms = np.sqrt(np.mean(target_audio**2))
    if is_silent_audio(target_audio, energy_threshold):
        return {
            'file': file_name,
            'similarity': 0.0,
            'energy': rms,
            'status': 'mute'
        }
    
    # Extract features
    target_features = extract_fast_features(target_audio, target_sr)
    
    if target_features is None:
        return {
            'file': file_name,
            'similarity': 0.0,
            'energy': rms,
            'status': 'feature_error'
        }
    
    # Calculate similarity
    similarity = calculate_similarity(reference_features, target_features)
    
    status = "MATCH" if similarity >= 0.5 else "DIFFERENT"
    
    return {
        'file': file_name,
        'similarity': similarity,
        'energy': rms,
        'status': status
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fast voice similarity check for TTS-generated files"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference voice file (e.g., data/moody1.mp3)"
    )
    parser.add_argument(
        "--directory",
        default=None,
        help="Directory containing _edited.wav files"
    )
    parser.add_argument(
        "--pattern",
        default="*_edited.wav",
        help="File pattern to match (default: *_edited.wav)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Minimum similarity score to consider voice matching (default: 0.5)"
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.01,
        help="Minimum RMS energy to consider non-silent (default: 0.01)"
    )
    parser.add_argument(
        "--output",
        default="voice_similarity_results.txt",
        help="Output file to save results (default: voice_similarity_results.txt)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Determine directory
    if args.directory is None:
        script_dir = Path(__file__).parent.parent
        directory = script_dir / "DailyTalkContiguous" / "data_stereo"
    else:
        directory = Path(args.directory)
    
    if not directory.exists():
        print(f"Error: Directory {directory} does not exist")
        sys.exit(1)
    
    # Find all matching files
    target_files = sorted(directory.glob(args.pattern))
    
    if not target_files:
        print(f"No files found matching pattern '{args.pattern}' in {directory}")
        sys.exit(1)
    
    print(f"Found {len(target_files)} files to check\n")
    
    # Load reference audio
    print(f"Loading reference audio: {args.reference}")
    ref_audio, ref_sr = load_audio(args.reference)
    
    if ref_audio is None:
        print(f"Error: Could not load reference audio {args.reference}")
        sys.exit(1)
    
    ref_energy = np.sqrt(np.mean(ref_audio**2))
    print(f"Reference energy: {ref_energy:.6f}")
    
    # Extract reference features
    print("Extracting reference voice features...")
    ref_features = extract_fast_features(ref_audio, ref_sr)
    
    if ref_features is None:
        print("Error: Could not extract features from reference audio")
        sys.exit(1)
    
    print(f"Reference features shape: {ref_features.shape}\n")
    
    # Process files in parallel
    results = []
    low_similarity_files = []
    
    print(f"Checking {len(target_files)} files using {args.workers} workers...")
    print("-" * 80)
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(
                process_file,
                ref_features,
                ref_energy,
                str(target_file),
                args.energy_threshold
            ): i
            for i, target_file in enumerate(target_files, 1)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            results.append(result)
            completed += 1
            
            if completed % 50 == 0:
                print(f"  Processed {completed}/{len(target_files)} files...", flush=True)
            
            if result['similarity'] < args.similarity_threshold:
                low_similarity_files.append(result['file'])
    
    print("-" * 80)
    
    # Sort results by file name
    results.sort(key=lambda x: x['file'])
    low_similarity_files.sort()
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Total files checked: {len(target_files)}")
    print(f"  Files with matching voice: {len([r for r in results if r['similarity'] >= args.similarity_threshold])}")
    print(f"  Files with different/missing voice: {len(low_similarity_files)}")
    
    # Print low similarity files (first 50)
    if low_similarity_files:
        print(f"\nFiles with low similarity (showing first 50):")
        for i, file_name in enumerate(low_similarity_files[:50], 1):
            result = next(r for r in results if r['file'] == file_name)
            print(f"  {i}. {file_name}")
            print(f"     Similarity: {result['similarity']:.4f}, Energy: {result['energy']:.6f}, Status: {result['status']}")
        
        if len(low_similarity_files) > 50:
            print(f"  ... and {len(low_similarity_files) - 50} more files")
    
    # Save results to file
    with open(args.output, 'w') as f:
        f.write(f"Voice Similarity Check Results\n")
        f.write(f"Reference: {args.reference}\n")
        f.write(f"Similarity Threshold: {args.similarity_threshold}\n")
        f.write(f"Energy Threshold: {args.energy_threshold}\n\n")
        
        f.write(f"Files with low similarity (n={len(low_similarity_files)}):\n")
        for file_name in low_similarity_files:
            result = next(r for r in results if r['file'] == file_name)
            f.write(f"{file_name},{result['similarity']:.4f},{result['energy']:.6f},{result['status']}\n")
        
        f.write(f"\nFull results:\n")
        for result in results:
            f.write(f"{result['file']},{result['similarity']:.4f},{result['energy']:.6f},{result['status']}\n")
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
