import math
import numpy as np
import librosa

# -----------------------------
# Fish tag mapping helpers
# -----------------------------

SER_TO_BASE = {
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "neutral": "calm",
    "excited": "excited",
    # fallback:
    "other": "calm",
}

def normalize_text_basic(t: str) -> str:
    t = (t or "").strip()
    t = " ".join(t.split())
    return t

def compute_rms(y: np.ndarray) -> float:
    # robust RMS energy
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    return float(np.mean(rms))

def compute_zcr(y: np.ndarray) -> float:
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)[0]
    return float(np.mean(zcr))

def compute_pitch_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Pitch extraction with librosa.pyin.
    Returns (f0_mean, f0_std).
    If pitch cannot be extracted, return (nan, nan).
    """
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=2048,
            hop_length=256,
        )
        if f0 is None:
            return float("nan"), float("nan")
        f0 = f0[~np.isnan(f0)]
        if len(f0) < 5:
            return float("nan"), float("nan")
        return float(np.mean(f0)), float(np.std(f0))
    except Exception:
        return float("nan"), float("nan")

def compute_speech_rate(words: int, dur_sec: float) -> float:
    if dur_sec <= 0.05:
        return 0.0
    return float(words / dur_sec)

def bucket_by_percentiles(values: list[float], p20: float, p80: float) -> list[str]:
    """
    Bucket each value into LOW/MID/HIGH based on p20/p80.
    """
    out = []
    for v in values:
        if math.isnan(v):
            out.append("MID")
        elif v <= p20:
            out.append("LOW")
        elif v >= p80:
            out.append("HIGH")
        else:
            out.append("MID")
    return out

def fish_tag_from_ser_and_prosody(
    ser_label: str,
    energy_bucket: str,
    rate_bucket: str,
    zcr_bucket: str,
    pitch_var_bucket: str = "MID",
) -> str:
    """
    Deterministic mapping:
    Inputs must be discrete buckets: LOW/MID/HIGH.
    """
    ser_label = (ser_label or "neutral").strip().lower()
    base = SER_TO_BASE.get(ser_label, "calm")

    # Priority order:
    # 1) whispering
    if energy_bucket == "LOW" and zcr_bucket == "HIGH":
        return "whispering"

    # 2) shouting
    if energy_bucket == "HIGH" and base == "angry":
        return "shouting"
    if energy_bucket == "HIGH" and base in {"happy", "excited"} and rate_bucket == "HIGH":
        return "shouting"

    # 3) soft tone
    if energy_bucket == "LOW" and base in {"sad", "calm"}:
        return "soft tone"

    # 4) in a hurry tone
    if rate_bucket == "HIGH" and energy_bucket != "LOW" and base != "sad":
        return "in a hurry tone"

    # 5) emotion refinement
    if base == "sad":
        if energy_bucket == "LOW" and rate_bucket == "LOW":
            return "depressed"
        return "sad"

    if base == "happy":
        if energy_bucket == "HIGH" or pitch_var_bucket == "HIGH":
            return "delighted"
        return "happy"

    if base == "angry":
        if rate_bucket == "HIGH" and energy_bucket != "LOW":
            return "frustrated"
        return "angry"

    if base == "calm":
        if rate_bucket == "LOW":
            return "relaxed"
        return "calm"

    if base == "excited":
        return "excited"

    return "calm"
