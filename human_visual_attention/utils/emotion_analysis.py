from __future__ import annotations

from typing import Any
import importlib.util
import sys

import numpy as np


EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def _empty_emotions() -> dict[str, float]:
    return {emotion: 0.0 for emotion in EMOTIONS}


def analyze_emotion(image_bgr: np.ndarray) -> dict[str, Any]:
    if image_bgr is None or image_bgr.size == 0:
        return {
            "success": False,
            "error": "Empty or unreadable image.",
            "dominant_emotion": "unknown",
            "emotions": _empty_emotions(),
        }

    if importlib.util.find_spec("tensorflow") is None:
        return {
            "success": False,
            "error": (
                "TensorFlow is not installed in the active Python environment, so DeepFace emotion "
                f"analysis cannot run. Active Python: {sys.executable}. Install with: "
                "python -m pip install tensorflow==2.15.1"
            ),
            "dominant_emotion": "unknown",
            "emotions": _empty_emotions(),
        }

    try:
        from deepface import DeepFace

        result = DeepFace.analyze(
            img_path=image_bgr,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        emotions = result.get("emotion", {}) or {}
        normalized = {emotion: float(emotions.get(emotion, 0.0)) for emotion in EMOTIONS}
        dominant = str(result.get("dominant_emotion") or max(normalized, key=normalized.get))
        return {
            "success": True,
            "error": None,
            "dominant_emotion": dominant,
            "emotions": normalized,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Emotion model could not complete analysis: {exc}",
            "dominant_emotion": "unknown",
            "emotions": _empty_emotions(),
        }
