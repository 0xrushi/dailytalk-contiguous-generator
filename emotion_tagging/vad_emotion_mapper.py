import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import librosa

@dataclass
class Prosody:
    rms: float
    zcr: float
    f0_mean: float
    f0_std: float
    speech_rate: float  # words/sec


def compute_rms(y: np.ndarray) -> float:
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    return float(np.mean(rms))


def compute_zcr(y: np.ndarray) -> float:
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256)[0]
    return float(np.mean(zcr))


def compute_pitch_stats(y: np.ndarray, sr: int) -> Tuple[float, float]:
    try:
        f0, _, _ = librosa.pyin(
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


def bucket_by_percentiles(values: List[float], p33: float, p66: float) -> List[str]:
    """
    Bucket into LOW/MID/HIGH using 33rd and 66th percentiles.
    """
    out = []
    for v in values:
        if math.isnan(v):
            out.append("MID")
        elif v <= p33:
            out.append("LOW")
        elif v >= p66:
            out.append("HIGH")
        else:
            out.append("MID")
    return out


def _p33_p66(vals: List[float]) -> Tuple[float, float]:
    a = np.array([v for v in vals if not np.isnan(v)])
    if len(a) < 5:
        # fall back to min/max-ish
        return float(np.nanmin(vals)), float(np.nanmax(vals))
    return float(np.percentile(a, 33)), float(np.percentile(a, 66))


def infer_end_punct(tag: str, text: str) -> str:
    text = " ".join((text or "").strip().split())
    if not text:
        return text
    if text[-1] in ".?!":
        return text
    if tag in {"excited", "delighted", "shouting", "screaming"}:
        return text + "!"
    if tag in {"curious", "uncertain", "doubtful", "confused"}:
        return text + "?"
    return text + "."


def fish_tag_from_vad_and_prosody(
    vad_b: Dict[str, str],  # {"valence":"LOW|MID|HIGH", "arousal":..., "dominance":...}
    pros_b: Dict[str, str], # {"energy":..., "rate":..., "zcr":..., "pitch_var":...}
) -> str:
    """
    Deterministic Fish-tag mapping that yields richer variety than categorical SER.

    Priority overrides:
      1) whispering (quiet + noisy) 
      2) shouting (very loud + high arousal)
      3) soft tone (very quiet)
      4) in a hurry tone (fast + high arousal)
    Then VAD grid mapping for emotion-ish tags.
    """
    V = vad_b["valence"]
    A = vad_b["arousal"]
    D = vad_b["dominance"]

    energy = pros_b["energy"]
    rate = pros_b["rate"]
    zcr = pros_b["zcr"]
    pitch_var = pros_b["pitch_var"]

    # 1) whispering
    if energy == "LOW" and zcr == "HIGH":
        return "whispering"

    # 2) shouting (use energy + arousal)
    if energy == "HIGH" and A == "HIGH":
        # if negative valence & high dominance => angry shout
        if V == "LOW" and D == "HIGH":
            return "shouting"
        # otherwise energetic shout
        return "shouting"

    # 3) soft tone (quiet delivery)
    if energy == "LOW":
        return "soft tone"

    # 4) in a hurry tone (fast delivery)
    if rate == "HIGH" and A != "LOW":
        return "in a hurry tone"

    # ---------- VAD → emotion-ish tags ----------
    # Arousal LOW: calm family
    if A == "LOW":
        if V == "HIGH":
            # positive calm
            return "relaxed" if D != "HIGH" else "satisfied"
        if V == "LOW":
            # negative calm
            return "depressed" if D == "LOW" else "sad"
        # V MID
        return "calm"

    # Arousal MID: conversational / nuanced
    if A == "MID":
        if V == "HIGH":
            # positive
            return "happy" if pitch_var == "HIGH" else "satisfied"
        if V == "LOW":
            # negative
            if D == "LOW":
                return "worried"
            if D == "HIGH":
                return "frustrated"
            return "disappointed"
        # V MID
        if D == "HIGH":
            return "confident"
        if D == "LOW":
            return "uncertain"
        return "calm"

    # Arousal HIGH: activated emotions
    if A == "HIGH":
        if V == "HIGH":
            # positive high arousal
            return "delighted" if D == "HIGH" else "excited"
        if V == "LOW":
            # negative high arousal
            if D == "HIGH":
                return "angry"
            if D == "LOW":
                return "anxious"
            return "frustrated"
        # V MID
        if D == "HIGH":
            return "determined"
        if D == "LOW":
            return "nervous"
        return "excited"

    return "calm"


def assign_fish_tags_from_vad(
    records: List[dict],
) -> List[dict]:
    """
    records: each record must contain:
      - vad: {"arousal":float, "dominance":float, "valence":float} (0..1-ish)
      - features: {"rms","zcr","f0_std","speech_rate"} etc.
    Adds:
      - vad_bucket: per dimension LOW/MID/HIGH (p33/p66 across file)
      - prosody_bucket: energy/rate/zcr/pitch_var LOW/MID/HIGH
      - tag
    """
    # collect continuous values
    ar = [r["vad"]["arousal"] for r in records]
    do = [r["vad"]["dominance"] for r in records]
    va = [r["vad"]["valence"] for r in records]

    rms = [r["features"]["rms"] for r in records]
    rate = [r["features"]["speech_rate"] for r in records]
    zcr = [r["features"]["zcr"] for r in records]
    f0s = [r["features"]["f0_std"] for r in records]

    ar_p33, ar_p66 = _p33_p66(ar)
    do_p33, do_p66 = _p33_p66(do)
    va_p33, va_p66 = _p33_p66(va)

    rms_p33, rms_p66 = _p33_p66(rms)
    rate_p33, rate_p66 = _p33_p66(rate)
    zcr_p33, zcr_p66 = _p33_p66(zcr)
    f0s_p33, f0s_p66 = _p33_p66(f0s)

    ar_b = bucket_by_percentiles(ar, ar_p33, ar_p66)
    do_b = bucket_by_percentiles(do, do_p33, do_p66)
    va_b = bucket_by_percentiles(va, va_p33, va_p66)

    en_b = bucket_by_percentiles(rms, rms_p33, rms_p66)
    ra_b = bucket_by_percentiles(rate, rate_p33, rate_p66)
    zc_b = bucket_by_percentiles(zcr, zcr_p33, zcr_p66)
    pv_b = bucket_by_percentiles(f0s, f0s_p33, f0s_p66)

    for i, r in enumerate(records):
        vad_b = {"arousal": ar_b[i], "dominance": do_b[i], "valence": va_b[i]}
        pros_b = {"energy": en_b[i], "rate": ra_b[i], "zcr": zc_b[i], "pitch_var": pv_b[i]}

        tag = fish_tag_from_vad_and_prosody(vad_b, pros_b)

        r["vad_bucket"] = vad_b
        r["prosody_bucket"] = pros_b
        r["tag"] = tag

        # optional punctuation helper
        if "text" in r and isinstance(r["text"], str):
            r["text"] = infer_end_punct(tag, r["text"])
            r["tagged_text"] = f"({tag}) {r['text']}"

    return records
