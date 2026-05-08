from __future__ import annotations


def build_interpretation(
    dominant_emotion: str,
    emotion_confidence: float,
    attention_score: int,
    engagement_level: str,
    orientation: str,
) -> str:
    emotion = (dominant_emotion or "unknown").lower()
    if emotion == "unknown":
        emotion_clause = "The emotional signal is inconclusive, so the interpretation should prioritize attention geometry."
    elif emotion in {"happy", "surprise"}:
        emotion_clause = f"The expression trends toward {emotion}, suggesting positive or activated affect."
    elif emotion in {"sad", "angry", "fear", "disgust"}:
        emotion_clause = f"The expression trends toward {emotion}, indicating possible cognitive load, discomfort, or stress."
    else:
        emotion_clause = "The expression is mostly neutral, which can indicate a steady or task-focused state."

    if attention_score >= 78:
        attention_clause = "Facial alignment, eye openness, and gaze cues are consistent with strong visual attention."
    elif attention_score >= 55:
        attention_clause = "The subject appears partially engaged, with some geometric cues drifting away from a direct attentive pose."
    elif attention_score >= 35:
        attention_clause = "Attention appears reduced; the face or gaze pattern is less aligned with the visual target."
    else:
        attention_clause = "The system detects weak attention cues, so this moment should be reviewed with context."

    confidence_clause = (
        f"Emotion confidence for the dominant class is {emotion_confidence:.1f}%, "
        f"while the heuristic attention score is {attention_score}/100."
    )

    return (
        f"{emotion_clause} {attention_clause} "
        f"Overall engagement is classified as {engagement_level.lower()} with orientation labeled as {orientation.lower()}. "
        f"{confidence_clause} This is an explainable CPU-only estimate, not a clinical or biometric identity assessment."
    )
