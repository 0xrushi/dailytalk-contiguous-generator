from typing import Dict

import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# Logits order is: arousal, dominance, valence
IDX = {"arousal": 0, "dominance": 1, "valence": 2}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE is not strictly used in the provided snippet but good to have if needed later
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


class AudeeringVAD:
    def __init__(self):
        self.fe = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(DEVICE)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        y: mono float waveform
        sr: sample rate
        Returns arousal/dominance/valence ~ in [0,1] range per model card
        """
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000

        inputs = self.fe(y, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        out = self.model(**inputs)
        logits = out.logits[0].detach().to("cpu").float().numpy()

        # Model expects outputs ~0..1; clip for safety (some users reported occasional out-of-range)
        arousal = float(np.clip(logits[IDX["arousal"]], 0.0, 1.0))
        dominance = float(np.clip(logits[IDX["dominance"]], 0.0, 1.0))
        valence = float(np.clip(logits[IDX["valence"]], 0.0, 1.0))

        return {"arousal": arousal, "dominance": dominance, "valence": valence}
