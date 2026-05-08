from __future__ import annotations

from typing import Any

import numpy as np


LEFT_EYE = [33, 133, 159, 145, 468]
RIGHT_EYE = [362, 263, 386, 374, 473]
NOSE_TIP = 1
MOUTH_CENTER = 13
CHIN = 152


def _point(landmarks: list[Any], index: int, width: int, height: int) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def analyze_attention(landmarks: list[Any] | None, frame_shape: tuple[int, ...]) -> dict[str, Any]:
    if not landmarks:
        return {
            "success": False,
            "attention_score": 0,
            "engagement_level": "No face",
            "orientation": "Unavailable",
            "factors": {"reason": "No MediaPipe landmarks were detected."},
        }

    height, width = frame_shape[:2]
    left_outer = _point(landmarks, 33, width, height)
    left_inner = _point(landmarks, 133, width, height)
    right_inner = _point(landmarks, 362, width, height)
    right_outer = _point(landmarks, 263, width, height)
    left_top = _point(landmarks, 159, width, height)
    left_bottom = _point(landmarks, 145, width, height)
    right_top = _point(landmarks, 386, width, height)
    right_bottom = _point(landmarks, 374, width, height)
    nose = _point(landmarks, NOSE_TIP, width, height)
    mouth = _point(landmarks, MOUTH_CENTER, width, height)
    chin = _point(landmarks, CHIN, width, height)

    eye_center = (left_inner + right_inner) / 2.0
    face_center_x = nose[0] / max(width, 1)
    centeredness = 100.0 - abs(face_center_x - 0.5) * 220.0

    eye_width = max(_distance(left_outer, right_outer), 1.0)
    eye_alignment_px = abs(left_inner[1] - right_inner[1])
    eye_alignment = 100.0 - (eye_alignment_px / eye_width) * 420.0

    yaw_offset = (nose[0] - eye_center[0]) / eye_width
    yaw_score = 100.0 - abs(yaw_offset) * 260.0

    upper_face = max(abs(mouth[1] - eye_center[1]), 1.0)
    lower_face = max(abs(chin[1] - mouth[1]), 1.0)
    pitch_ratio = upper_face / lower_face
    pitch_score = 100.0 - abs(pitch_ratio - 1.05) * 85.0

    left_eye_open = _distance(left_top, left_bottom) / max(_distance(left_outer, left_inner), 1.0)
    right_eye_open = _distance(right_top, right_bottom) / max(_distance(right_outer, right_inner), 1.0)
    eye_open = (left_eye_open + right_eye_open) / 2.0
    eye_open_score = _clamp((eye_open - 0.12) * 360.0)

    iris_score = 70.0
    if len(landmarks) > 473:
        left_iris = _point(landmarks, 468, width, height)
        right_iris = _point(landmarks, 473, width, height)
        left_ratio = (left_iris[0] - left_outer[0]) / max(left_inner[0] - left_outer[0], 1.0)
        right_ratio = (right_iris[0] - right_inner[0]) / max(right_outer[0] - right_inner[0], 1.0)
        gaze_offset = abs(left_ratio - 0.5) + abs(right_ratio - 0.5)
        iris_score = 100.0 - gaze_offset * 120.0

    factors = {
        "face_centeredness": round(_clamp(centeredness), 1),
        "eye_alignment": round(_clamp(eye_alignment), 1),
        "head_yaw": round(_clamp(yaw_score), 1),
        "head_pitch": round(_clamp(pitch_score), 1),
        "eye_openness": round(_clamp(eye_open_score), 1),
        "gaze_centering": round(_clamp(iris_score), 1),
    }
    score = (
        factors["face_centeredness"] * 0.18
        + factors["eye_alignment"] * 0.14
        + factors["head_yaw"] * 0.2
        + factors["head_pitch"] * 0.14
        + factors["eye_openness"] * 0.18
        + factors["gaze_centering"] * 0.16
    )
    score = int(round(_clamp(score)))

    if score >= 78:
        engagement = "High"
    elif score >= 55:
        engagement = "Moderate"
    elif score >= 35:
        engagement = "Low"
    else:
        engagement = "Minimal"

    if abs(yaw_offset) < 0.08 and 0.72 <= pitch_ratio <= 1.42:
        orientation = "Forward attentive"
    elif yaw_offset > 0.08:
        orientation = "Head turned right"
    elif yaw_offset < -0.08:
        orientation = "Head turned left"
    elif pitch_ratio > 1.42:
        orientation = "Head angled downward"
    else:
        orientation = "Head angled upward"

    return {
        "success": True,
        "attention_score": score,
        "engagement_level": engagement,
        "orientation": orientation,
        "factors": factors,
    }
