from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .attention_analysis import analyze_attention
from .emotion_analysis import EMOTIONS, analyze_emotion
from .face_detection import detect_primary_face, extract_face_mesh, resize_for_processing


def process_video(
    video_path: str | Path,
    frame_step: int = 5,
    max_frames: int = 80,
    max_width: int = 760,
    progress_callback=None,
) -> dict[str, Any]:
    path = Path(video_path)
    if not path.exists() or path.stat().st_size == 0:
        return {"success": False, "error": "Video upload is empty or missing.", "no_face": True}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"success": False, "error": "Video read failure: OpenCV could not open this file.", "no_face": True}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return {"success": False, "error": "Video read failure: no readable frames were found.", "no_face": True}

    emotions_accumulator = {emotion: [] for emotion in EMOTIONS}
    attention_scores: list[int] = []
    dominant_emotions: list[str] = []
    representative = None
    sampled = 0
    faces_found = 0
    frame_index = 0

    try:
        while sampled < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % max(1, frame_step) != 0:
                frame_index += 1
                continue

            sampled += 1
            frame = resize_for_processing(frame, max_width=max_width)
            detection = detect_primary_face(frame)
            mesh = extract_face_mesh(frame, static_image_mode=False, max_width=max_width)

            if not detection.no_face:
                faces_found += 1
                representative = mesh.annotated_image if not mesh.no_face else detection.annotated_image
                emotion = analyze_emotion(frame)
                if emotion["success"]:
                    dominant_emotions.append(emotion["dominant_emotion"])
                    for key, value in emotion["emotions"].items():
                        emotions_accumulator[key].append(value)

                attention = analyze_attention(mesh.landmarks, frame.shape)
                if attention["success"]:
                    attention_scores.append(attention["attention_score"])

            if progress_callback:
                progress_callback(min(1.0, frame_index / max(total_frames, 1)))
            frame_index += 1
    except Exception as exc:
        cap.release()
        return {"success": False, "error": f"Video processing stopped safely: {exc}", "no_face": True}
    finally:
        cap.release()

    if sampled == 0:
        return {"success": False, "error": "Video read failure: no sampled frames were readable.", "no_face": True}
    if faces_found == 0:
        return {
            "success": True,
            "no_face": True,
            "sampled_frames": sampled,
            "face_detection_rate": 0.0,
            "error": None,
        }

    averaged_emotions = {
        emotion: float(np.mean(values)) if values else 0.0
        for emotion, values in emotions_accumulator.items()
    }
    dominant = max(averaged_emotions, key=averaged_emotions.get) if averaged_emotions else "unknown"
    average_attention = int(round(float(np.mean(attention_scores)))) if attention_scores else 0
    if average_attention >= 78:
        engagement = "High"
    elif average_attention >= 55:
        engagement = "Moderate"
    elif average_attention >= 35:
        engagement = "Low"
    else:
        engagement = "Minimal"

    return {
        "success": True,
        "no_face": False,
        "error": None,
        "sampled_frames": sampled,
        "faces_found": faces_found,
        "face_detection_rate": round((faces_found / sampled) * 100.0, 1),
        "emotions": averaged_emotions,
        "dominant_emotion": dominant,
        "average_attention": average_attention,
        "engagement_level": engagement,
        "representative_frame": representative,
    }
