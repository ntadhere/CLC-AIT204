"""
============================================================================
AIT-204 Deep Learning | Topic 4: Natural Language Processing
ACTIVITY 4 — Part A: Backend Service Layer
============================================================================
"""

import json
from pathlib import Path

import torch

from activity1_preprocessing import (
    Vocabulary, clean_text, tokenize, preprocess_for_model
)
from activity2_model import load_model

# -------------------------------------------------------------------------
# IMPORTANT: Build MODEL_DIR relative to THIS FILE, not the working directory.
# This fixes Streamlit Cloud FileNotFoundError for saved_model/vocab.json.
# -------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../Topic4_NLP
MODEL_DIR = BASE_DIR / "saved_model"               # .../Topic4_NLP/saved_model


class SentimentService:
    """
    Backend service: wraps the trained model and exposes clean prediction
    methods to the frontend. Contains zero UI code.
    """

    def __init__(self, model_dir: str | Path = MODEL_DIR):
        """
        Load all model artifacts from disk.
        Called ONCE at app startup (Streamlit caches this with @st.cache_resource).
        """

        model_dir = Path(model_dir)

        # Optional: clearer error if artifacts are missing on Streamlit Cloud
        vocab_path = model_dir / "vocab.json"
        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"

        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Missing vocab.json at: {vocab_path}\n"
                f"Expected folder contents: {model_dir}\n"
                f"Tip: Ensure Topic4_NLP/saved_model/ is committed & pushed to GitHub."
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing model.pt at: {model_path}\n"
                f"Tip: Ensure Topic4_NLP/saved_model/model.pt is committed & pushed."
            )
        if not config_path.exists():
            raise FileNotFoundError(
                f"Missing config.json at: {config_path}\n"
                f"Tip: Ensure Topic4_NLP/saved_model/config.json is committed & pushed."
            )

        # ── TODO 1 ────────────────────────────────────────────────────────
        # Load the vocabulary saved by Activity 3.
        self.vocab = Vocabulary.load(vocab_path)

        # ── TODO 2 ────────────────────────────────────────────────────────
        # Load the trained model saved by Activity 3.
        self.model = load_model(model_path)
        self.model.eval()

        # Load max_length from config
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.max_length = config["max_length"]

        print(f"[Backend] Model loaded  ({sum(p.numel() for p in self.model.parameters()):,} params)")
        print(f"[Backend] Vocabulary    ({len(self.vocab)} words)")
        print(f"[Backend] Max length    ({self.max_length} tokens)")
        print(f"[Backend] Model dir     ({model_dir})")

    def predict(self, text: str) -> dict:
        """Run the full NLP pipeline on one review and return a dict."""

        if not text or not text.strip():
            return {
                "sentiment": "Unknown",
                "confidence": 0.5,
                "positive_score": 0.5,
                "negative_score": 0.5,
                "cleaned": "",
                "tokens": [],
                "encoded": [],
                "known_count": 0,
            }

        cleaned = clean_text(text=text)
        tokens = tokenize(cleaned)
        encoded = self.vocab.encode(tokens)
        tensor = preprocess_for_model(text, self.vocab, self.max_length)

        with torch.no_grad():
            probability = float(self.model(tensor).item())

        if probability >= 0.5:
            sentiment = "Positive"   # FIX typo
            confidence = probability
        else:
            sentiment = "Negative"
            confidence = 1.0 - probability

        known_count = sum(1 for t in tokens if t in self.vocab.word2idx)

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "positive_score": float(probability),
            "negative_score": float(1.0 - probability),
            "cleaned": cleaned,
            "tokens": tokens,
            "encoded": encoded[: self.max_length],
            "known_count": int(known_count),
        }

    def compare(self, original: str, translated: str) -> dict:
        """Score two texts and return a comparison dict."""

        orig_result = self.predict(original)
        trans_result = self.predict(translated)

        delta = float(trans_result["positive_score"] - orig_result["positive_score"])
        changed = bool(orig_result["sentiment"] != trans_result["sentiment"])

        orig_words = set(t for t in orig_result["tokens"] if t in self.vocab.word2idx)
        trans_words = set(t for t in trans_result["tokens"] if t in self.vocab.word2idx)

        lost_words = sorted(orig_words - trans_words)
        new_words = sorted(trans_words - orig_words)

        return {
            "original": orig_result,
            "translated": trans_result,
            "delta": delta,
            "changed": changed,
            "lost_words": lost_words,
            "new_words": new_words,
        }


if __name__ == "__main__":
    print("=" * 62)
    print("  Backend Service — Self-Test")
    print("=" * 62)

    svc = SentimentService()

    print("\n[Test 1] predict() — positive review:")
    r = svc.predict("This movie was absolutely wonderful and I loved it")
    print(f"  Sentiment   : {r['sentiment']} ({r['confidence']:.1%} confidence)")
    print(f"  Score       : {r['positive_score']:.4f}")
    print(f"  Tokens      : {r['tokens']}")
    print(f"  Vocab hit   : {r['known_count']}/{len(r['tokens'])}")

    print("\n[Test 2] predict() — negative review:")
    r2 = svc.predict("Awful acting and a terrible waste of time from start to finish")
    print(f"  Sentiment   : {r2['sentiment']} ({r2['confidence']:.1%} confidence)")
    print(f"  Score       : {r2['positive_score']:.4f}")

    print("\n[Test 3] compare() — round-trip translation:")
    cmp = svc.compare(
        "This film was absolutely brilliant and moving",
        "This film was completely bright and moving",
    )
    print(f"  Original    : {cmp['original']['sentiment']}  ({cmp['original']['positive_score']:.4f})")
    print(f"  Translated  : {cmp['translated']['sentiment']} ({cmp['translated']['positive_score']:.4f})")
    print(f"  Delta       : {cmp['delta']:+.4f}")
    print(f"  Changed     : {cmp['changed']}")
    print(f"  Lost words  : {cmp['lost_words']}")
    print(f"  New words   : {cmp['new_words']}")

    print("\n" + "=" * 62)
    print("  Backend self-test complete.")
    print("  Next: streamlit run activity4_app.py")
    print("=" * 62)