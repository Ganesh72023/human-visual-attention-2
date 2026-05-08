# Human Visual Attention Intelligence System

A Streamlit dashboard for local CPU-friendly image and video analysis. It detects one primary face with MediaPipe, renders facial landmarks, estimates emotion with DeepFace, computes explainable attention heuristics, and produces deterministic cognitive interpretation text.

## Features

- Image upload: `jpg`, `jpeg`, `png`
- Video upload: `mp4`, `avi`, `mov`
- MediaPipe primary-face detection and neon annotation
- MediaPipe Face Mesh with eye, iris, mouth, and contour rendering
- DeepFace pretrained emotion confidence analysis
- Rule-based attention score from face centeredness, eye alignment, yaw, pitch, eye openness, and gaze centering
- Sampled video summaries for low-end laptop performance
- Futuristic dark Streamlit UI with Plotly charts

## Setup

Use Python 3.10 on Windows.

```powershell
cd "C:\human visual attention\human_visual_attention"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

DeepFace may download pretrained model weights during first use. No model training is performed.

## Run

```powershell
streamlit run app.py
```

## Project Structure

```text
human_visual_attention/
  app.py
  requirements.txt
  README.md
  assets/
    styles.css
    animations/
  outputs/
  temp/
  uploads/
  utils/
    attention_analysis.py
    cognitive_interpreter.py
    emotion_analysis.py
    face_detection.py
    ui_components.py
    video_processor.py
    visualization.py
```

## Manual Test Plan

- Valid face image: shows bounding box, face mesh, emotion chart, attention gauge, and interpretation.
- No-face image: shows `⚠ No face detected in uploaded media`.
- Corrupted image: shows a clean image error instead of crashing.
- Valid face video: processes sampled frames and shows aggregated results.
- No-face video: shows the no-face warning.
- Corrupted or unreadable video: shows a video read failure message.
- Performance: confirm large images are resized and videos process every 5th frame by default.

## Notes

The attention engine is heuristic and explainable. It is not a clinical assessment, identity system, or trained attention model. It analyzes only the highest-confidence detected face.
