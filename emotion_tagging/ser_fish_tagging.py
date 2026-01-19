import numpy as np
import torch
import librosa
from speechbrain.inference.interfaces import foreign_class

from emotion_mapper import (
    compute_rms,
    compute_zcr,
    compute_pitch_stats,
    compute_speech_rate,
    bucket_by_percentiles,
    fish_tag_from_ser_and_prosody,
)

SAMPLE_RATE = 16000

# SpeechBrain SER model
SER_CLASSIFIER = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier",
)

def predict_ser_label(wav_path: str) -> str:
    out_prob, score, index, label = SER_CLASSIFIER.classify_file(wav_path)
    # label is usually list-like, pick first
    return str(label[0]).strip().lower()

def extract_features_for_segment(y: np.ndarray, sr: int, transcript: str, start: float, end: float) -> dict:
    dur = max(0.001, float(end - start))
    words = len((transcript or "").split())

    f0_mean, f0_std = compute_pitch_stats(y, sr)

    return {
        "duration": dur,
        "rms": compute_rms(y),
        "zcr": compute_zcr(y),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "speech_rate": compute_speech_rate(words, dur),
    }

def assign_fish_tags(records: list[dict]) -> list[dict]:
    """
    Input: list of dicts, each dict must include:
      - ser_label
      - features: rms, zcr, f0_std, speech_rate
    Output: same list but with:
      - energy_bucket, rate_bucket, zcr_bucket, pitch_var_bucket
      - tag
    """

    rms_vals = [r["features"]["rms"] for r in records]
    rate_vals = [r["features"]["speech_rate"] for r in records]
    zcr_vals = [r["features"]["zcr"] for r in records]
    f0std_vals = [r["features"]["f0_std"] for r in records]

    def p20_p80(vals):
        a = np.array([v for v in vals if not np.isnan(v)])
        if len(a) < 5:
            return float(np.nanmin(vals)), float(np.nanmax(vals))
        return float(np.percentile(a, 20)), float(np.percentile(a, 80))

    rms_p20, rms_p80 = p20_p80(rms_vals)
    rate_p20, rate_p80 = p20_p80(rate_vals)
    zcr_p20, zcr_p80 = p20_p80(zcr_vals)
    f0std_p20, f0std_p80 = p20_p80(f0std_vals)

    energy_b = bucket_by_percentiles(rms_vals, rms_p20, rms_p80)
    rate_b = bucket_by_percentiles(rate_vals, rate_p20, rate_p80)
    zcr_b = bucket_by_percentiles(zcr_vals, zcr_p20, zcr_p80)
    pitchvar_b = bucket_by_percentiles(f0std_vals, f0std_p20, f0std_p80)

    for i, r in enumerate(records):
        eb = energy_b[i]
        rb = rate_b[i]
        zb = zcr_b[i]
        pb = pitchvar_b[i]

        r["energy_bucket"] = eb
        r["rate_bucket"] = rb
        r["zcr_bucket"] = zb
        r["pitch_var_bucket"] = pb

        r["tag"] = fish_tag_from_ser_and_prosody(
            ser_label=r.get("ser_label", "neutral"),
            energy_bucket=eb,
            rate_bucket=rb,
            zcr_bucket=zb,
            pitch_var_bucket=pb,
        )

    return records
