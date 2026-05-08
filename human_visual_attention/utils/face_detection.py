from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import sys

import cv2
import numpy as np


def _load_mediapipe_solutions() -> tuple[Any, Any, Any, str | None]:
    try:
        _patch_protobuf_symbol_database()
        import mediapipe as mp

        return mp.solutions.face_detection, mp.solutions.face_mesh, mp.solutions.drawing_utils, None
    except Exception as exc:
        return None, None, None, str(exc)


def _patch_protobuf_symbol_database() -> None:
    try:
        from google.protobuf import message_factory, symbol_database

        database_cls = symbol_database.SymbolDatabase
        if hasattr(database_cls, "GetPrototype"):
            return

        def get_prototype(self, descriptor):
            return message_factory.GetMessageClass(descriptor)

        database_cls.GetPrototype = get_prototype
    except Exception:
        return


def _mediapipe_error(detail: str | None = None) -> str:
    detail_text = f" Detail: {detail}" if detail else ""
    return (
        "MediaPipe Face Detection/Face Mesh solutions are unavailable in this environment. "
        "Install the Python 3.10 dependency set from requirements.txt, especially mediapipe==0.10.14. "
        f"Active Python: {sys.executable}.{detail_text}"
    )


@dataclass
class FaceDetectionResult:
    face_box: tuple[int, int, int, int] | None
    confidence: float
    annotated_image: np.ndarray
    no_face: bool
    error: str | None = None


@dataclass
class LandmarkResult:
    landmarks: list[Any] | None
    annotated_image: np.ndarray
    no_face: bool
    error: str | None = None


def resize_for_processing(image: np.ndarray, max_width: int = 960) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def detect_primary_face(image_bgr: np.ndarray, min_confidence: float = 0.55) -> FaceDetectionResult:
    if image_bgr is None or image_bgr.size == 0:
        return FaceDetectionResult(None, 0.0, image_bgr, True, "Empty or unreadable image.")

    mp_face_detection, _, _, mp_error = _load_mediapipe_solutions()
    if mp_face_detection is None:
        return FaceDetectionResult(None, 0.0, image_bgr, True, _mediapipe_error(mp_error))

    frame = resize_for_processing(image_bgr)
    annotated = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_confidence,
        ) as detector:
            results = detector.process(rgb)
    except Exception as exc:
        return FaceDetectionResult(None, 0.0, annotated, True, f"MediaPipe face detection failed: {exc}")

    detections = results.detections or []
    if not detections:
        return FaceDetectionResult(None, 0.0, annotated, True)

    best = max(detections, key=lambda det: det.score[0] if det.score else 0.0)
    confidence = float(best.score[0]) if best.score else 0.0
    box = best.location_data.relative_bounding_box
    h, w = frame.shape[:2]
    x1 = max(0, int(box.xmin * w))
    y1 = max(0, int(box.ymin * h))
    bw = min(w - x1, int(box.width * w))
    bh = min(h - y1, int(box.height * h))
    x2 = min(w - 1, x1 + bw)
    y2 = min(h - 1, y1 + bh)

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.rectangle(annotated, (x1, max(0, y1 - 28)), (min(w - 1, x1 + 150), y1), (0, 210, 255), -1)
    cv2.putText(
        annotated,
        f"FACE {confidence:.2f}",
        (x1 + 8, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (8, 12, 20),
        2,
        cv2.LINE_AA,
    )

    return FaceDetectionResult((x1, y1, x2, y2), confidence, annotated, False)


def extract_face_mesh(
    image_bgr: np.ndarray,
    static_image_mode: bool = True,
    max_width: int = 960,
    min_confidence: float = 0.5,
) -> LandmarkResult:
    if image_bgr is None or image_bgr.size == 0:
        return LandmarkResult(None, image_bgr, True, "Empty or unreadable image.")

    _, mp_face_mesh, mp_drawing, mp_error = _load_mediapipe_solutions()
    if mp_face_mesh is None or mp_drawing is None:
        return LandmarkResult(None, image_bgr, True, _mediapipe_error(mp_error))

    frame = resize_for_processing(image_bgr, max_width=max_width)
    annotated = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=0.5,
        ) as mesh:
            results = mesh.process(rgb)
    except Exception as exc:
        return LandmarkResult(None, annotated, True, f"MediaPipe face mesh failed: {exc}")

    faces = results.multi_face_landmarks or []
    if not faces:
        return LandmarkResult(None, annotated, True)

    face_landmarks = faces[0]
    mesh_style = mp_drawing.DrawingSpec(color=(0, 190, 255), thickness=1, circle_radius=1)
    contour_style = mp_drawing.DrawingSpec(color=(90, 255, 230), thickness=1, circle_radius=1)

    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mesh_style,
    )
    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=contour_style,
    )
    mp_drawing.draw_landmarks(
        image=annotated,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 70, 220), thickness=1),
    )

    return LandmarkResult(face_landmarks.landmark, annotated, False)
