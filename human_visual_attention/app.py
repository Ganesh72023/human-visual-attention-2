from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from utils.attention_analysis import analyze_attention
from utils.cognitive_interpreter import build_interpretation
from utils.emotion_analysis import analyze_emotion
from utils.face_detection import detect_primary_face, extract_face_mesh, resize_for_processing
from utils.ui_components import (
    empty_state,
    load_css,
    metric_card,
    panel_title,
    render_header,
    sidebar_note,
    status_box,
    workflow_steps,
)
from utils.video_processor import process_video
from utils.visualization import attention_gauge, emotion_bar_chart, video_summary_chart


BASE_DIR = Path(__file__).parent
REQUIRED_DIRS = ["uploads", "outputs", "temp", "assets/animations"]
IMAGE_TYPES = {"jpg", "jpeg", "png"}
VIDEO_TYPES = {"mp4", "avi", "mov"}


def ensure_directories() -> None:
    for folder in REQUIRED_DIRS:
        (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def read_image_upload(uploaded_file) -> np.ndarray | None:
    try:
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        uploaded_file.seek(0)
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def save_upload(uploaded_file, destination: Path) -> Path:
    destination.write_bytes(uploaded_file.getbuffer())
    uploaded_file.seek(0)
    return destination


def render_image_analysis(image_bgr: np.ndarray, min_confidence: float, max_width: int) -> None:
    panel_title("Analysis Results", "Image-level face, emotion, and attention outputs.")
    frame = resize_for_processing(image_bgr, max_width=max_width)
    detection = detect_primary_face(frame, min_confidence=min_confidence)
    mesh = extract_face_mesh(frame, static_image_mode=True, max_width=max_width, min_confidence=min_confidence)

    if detection.error:
        status_box("error", detection.error)
        return
    if detection.no_face:
        status_box("warn", "No face detected in uploaded media.")
        return

    emotion = analyze_emotion(frame)
    attention = analyze_attention(mesh.landmarks, frame.shape)

    face_col, mesh_col = st.columns(2)
    with face_col:
        panel_title("Face Detection", "Highest-confidence MediaPipe face.")
        st.image(bgr_to_rgb(detection.annotated_image), use_container_width=True)
        metric_card("Detection Confidence", f"{detection.confidence * 100:.1f}%", "Single primary face analyzed")
    with mesh_col:
        panel_title("Facial Landmark Mesh", "Eyes, face contours, and iris geometry.")
        st.image(bgr_to_rgb(mesh.annotated_image), use_container_width=True)
        metric_card("Landmark Status", "Active" if not mesh.no_face else "Unavailable", "Face Mesh refined landmarks")

    if not emotion["success"]:
        status_box("error", emotion["error"])

    dominant = emotion["dominant_emotion"]
    emotions = emotion["emotions"]
    dominant_confidence = float(emotions.get(dominant, 0.0))

    chart_col, gauge_col = st.columns(2)
    with chart_col:
        panel_title("Emotion Analysis", "DeepFace pretrained emotion probabilities.")
        st.plotly_chart(emotion_bar_chart(emotions), use_container_width=True)
    with gauge_col:
        panel_title("Attention Score", "Explainable face and eye geometry estimate.")
        st.plotly_chart(attention_gauge(attention["attention_score"]), use_container_width=True)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        metric_card("Dominant Emotion", dominant.title(), f"{dominant_confidence:.1f}% confidence")
    with metric_col2:
        metric_card("Engagement", attention["engagement_level"], f"{attention['attention_score']}/100 attention")
    with metric_col3:
        metric_card("Orientation", attention["orientation"], "Head and gaze geometry")

    with st.expander("Attention contributing factors", expanded=True):
        st.dataframe(attention["factors"], use_container_width=True)

    panel_title("Cognitive Interpretation")
    st.markdown(
        f"<div class='status-box ok'>{build_interpretation(dominant, dominant_confidence, attention['attention_score'], attention['engagement_level'], attention['orientation'])}</div>",
        unsafe_allow_html=True,
    )


def render_video_analysis(uploaded_file, frame_step: int, max_frames: int, max_width: int) -> None:
    panel_title("Analysis Results", "Video summary based on sampled frames.")
    temp_path = BASE_DIR / "temp" / Path(uploaded_file.name).name
    save_upload(uploaded_file, temp_path)

    st.markdown("<div class='processing-label'>Processing sampled frames</div>", unsafe_allow_html=True)
    progress = st.progress(0)

    def update_progress(value: float) -> None:
        progress.progress(min(1.0, max(0.0, value)))

    summary = process_video(
        temp_path,
        frame_step=frame_step,
        max_frames=max_frames,
        max_width=max_width,
        progress_callback=update_progress,
    )
    progress.progress(1.0)

    if not summary["success"]:
        status_box("error", summary.get("error", "Video read failure."))
        return
    if summary.get("no_face"):
        status_box("warn", "No face detected in uploaded media.")
        metric_card("Frames Sampled", str(summary.get("sampled_frames", 0)), "No valid face tracks found")
        return

    status_box("ok", "Video sampled successfully. Results summarize detected face frames only.")
    if summary.get("representative_frame") is not None:
        panel_title("Representative Annotated Frame")
        st.image(bgr_to_rgb(summary["representative_frame"]), use_container_width=True)

    top1, top2, top3 = st.columns(3)
    with top1:
        metric_card("Average Attention", f"{summary['average_attention']}/100", summary["engagement_level"])
    with top2:
        metric_card("Face Detection Rate", f"{summary['face_detection_rate']:.1f}%", f"{summary['faces_found']} frames with face")
    with top3:
        metric_card("Dominant Emotion", summary["dominant_emotion"].title(), f"{summary['sampled_frames']} sampled frames")

    chart_col, gauge_col = st.columns(2)
    with chart_col:
        panel_title("Aggregated Emotion", "Average DeepFace confidence across sampled frames.")
        st.plotly_chart(emotion_bar_chart(summary["emotions"]), use_container_width=True)
    with gauge_col:
        panel_title("Video Attention Gauge", "Mean attention estimate across detected frames.")
        st.plotly_chart(attention_gauge(summary["average_attention"]), use_container_width=True)

    panel_title("Video Processing Summary")
    st.plotly_chart(video_summary_chart(summary), use_container_width=True)

    dominant_confidence = float(summary["emotions"].get(summary["dominant_emotion"], 0.0))
    st.markdown(
        f"<div class='status-box ok'>{build_interpretation(summary['dominant_emotion'], dominant_confidence, summary['average_attention'], summary['engagement_level'], 'video-average orientation')}</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    ensure_directories()
    st.set_page_config(
        page_title="Human Visual Attention Intelligence",
        page_icon="HVA",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css(BASE_DIR / "assets" / "styles.css")
    render_header()
    workflow_steps(active=1)

    with st.sidebar:
        st.markdown("<div class='sidebar-brand'>HVA Console</div>", unsafe_allow_html=True)
        sidebar_note(
            "How to use",
            [
                "Upload an image or video.",
                "Adjust speed and confidence only if needed.",
                "Press Analyze to run local CPU processing.",
            ],
        )
        uploaded_file = st.file_uploader(
            "Media file",
            type=sorted(IMAGE_TYPES | VIDEO_TYPES),
            accept_multiple_files=False,
        )
        st.divider()
        st.markdown("<div class='sidebar-section-title'>Processing</div>", unsafe_allow_html=True)
        max_width = st.slider("Frame width", min_value=480, max_value=1280, value=860, step=40)
        min_confidence = st.slider("Face confidence", min_value=0.30, max_value=0.90, value=0.55, step=0.05)
        with st.expander("Video sampling", expanded=False):
            frame_step = st.slider("Analyze every N frames", min_value=1, max_value=15, value=5)
            max_frames = st.slider("Maximum sampled frames", min_value=10, max_value=160, value=80, step=10)
        sidebar_note(
            "Performance tip",
            ["Lower frame width and higher frame step are faster on low-end laptops."],
        )

    if uploaded_file is None:
        empty_state()
        return
    if uploaded_file.size == 0:
        status_box("error", "Empty upload. Please choose a non-empty media file.")
        return

    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else ""
    if suffix not in IMAGE_TYPES | VIDEO_TYPES:
        status_box("error", "Unsupported media type. Please upload jpg, jpeg, png, mp4, avi, or mov.")
        return

    upload_path = BASE_DIR / "uploads" / Path(uploaded_file.name).name
    save_upload(uploaded_file, upload_path)

    workflow_steps(active=2)
    preview_col, status_col = st.columns([2, 1])
    with preview_col:
        panel_title("Uploaded Media Preview")
        if suffix in IMAGE_TYPES:
            image_bgr = read_image_upload(uploaded_file)
            if image_bgr is None:
                status_box("error", "Corrupted image or unsupported image payload.")
                return
            st.image(bgr_to_rgb(resize_for_processing(image_bgr, max_width=max_width)), use_container_width=True)
        else:
            st.video(uploaded_file)
    with status_col:
        metric_card("File", uploaded_file.name, f"{uploaded_file.size / 1024:.1f} KB")
        metric_card("Mode", "Image" if suffix in IMAGE_TYPES else "Video", "Single face analysis")
        metric_card("Processing Width", f"{max_width}px", "Resize limit for speed")

    st.markdown("<div class='analysis-action'>", unsafe_allow_html=True)
    analyze_now = st.button("Analyze uploaded media", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if not analyze_now:
        workflow_steps(active=3)
        status_box("ok", "Ready to analyze. Review the preview and settings, then press Analyze uploaded media.")
        return

    workflow_steps(active=4)
    if suffix in IMAGE_TYPES:
        render_image_analysis(image_bgr, min_confidence=min_confidence, max_width=max_width)
    else:
        render_video_analysis(uploaded_file, frame_step=frame_step, max_frames=max_frames, max_width=max_width)


if __name__ == "__main__":
    main()
