import os
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 256


@dataclass
class Result:
    label: str
    score: float
    is_injection: bool


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Cf")
    return " ".join(text.split())


class InjectionClassifier:
    def __init__(self, model_dir, threshold=0.5):
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.threshold = threshold
        self.injection_id = self.model.config.label2id.get("injection", 1)

    @torch.inference_mode()
    def predict_batch(self, texts):
        clean = [normalize(t) for t in texts]
        enc = self.tok(clean, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        probs = torch.softmax(self.model(**enc).logits, dim=-1)[:, self.injection_id].tolist()
        return [
            Result("injection" if p >= self.threshold else "benign", round(p, 4), p >= self.threshold)
            for p in probs
        ]

    def predict(self, text):
        return self.predict_batch([text])[0]


@lru_cache(maxsize=1)
def get_classifier():
    return InjectionClassifier(
        os.getenv("MODEL_DIR", "model"),
        threshold=float(os.getenv("THRESHOLD", "0.5")),
    )
