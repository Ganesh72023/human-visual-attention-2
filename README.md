# Human Visual Attention Intelligence System

A Streamlit dashboard for local CPU-friendly image and video analysis.
Detects one primary face with MediaPipe, renders facial landmarks, estimates emotion with DeepFace, computes explainable attention heuristics, and generates deterministic cognitive interpretation text.

## Features

- Upload images: `jpg`, `jpeg`, `png`
- Upload videos: `mp4`, `avi`, `mov`
- MediaPipe primary face detection and neon annotation
- MediaPipe Face Mesh with eye, iris, mouth, and contour rendering
- DeepFace pretrained emotion confidence analysis
- Rule-based attention score from face centeredness, eye alignment, yaw, pitch, eye openness, and gaze centering
- Sampled video summaries for low-end laptop performance
- Dark Streamlit UI with Plotly charts

## Quick Start

```bash
git clone https://github.com/<username>/<repo>.git
cd "c:\human visual attention"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

> On macOS/Linux, activate the venv with `source .venv/bin/activate`.

DeepFace may download pretrained model weights during first use. No model training is performed.

## Repository Layout

```text
.
  app.py
  requirements.txt
  README.md
  .gitignore
  .github/workflows/python-app.yml
  .streamlit/
  assets/
  input_images/
  input_video/
  outputs/
  temp/
  uploads/
  utils/
```

## GitHub Ready

- Flattened project layout for a clean repository root
- `.gitignore` excludes local virtual environments and runtime artifacts
- GitHub Actions workflow validates dependency installation and Python syntax

## Notes

This project is an explainable attention estimation demo. It is not a clinical assessment, identity system, or trained attention model. It analyzes only the highest-confidence detected face.
