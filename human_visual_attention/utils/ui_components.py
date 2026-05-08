from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st


def load_css(css_path: str | Path) -> None:
    path = Path(css_path)
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
          <div>
            <div class="hero-kicker">Local CPU Vision Dashboard</div>
            <h1>Human Visual Attention Intelligence</h1>
            <p>Upload an image or video, review the preview, then run face, emotion, and attention analysis from one focused workspace.</p>
          </div>
          <div class="hero-status">
            <span>MediaPipe</span>
            <strong>DeepFace</strong>
            <span>Explainable Attention</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <span>{escape(label)}</span>
          <strong>{escape(value)}</strong>
          <small>{escape(detail)}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_title(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='panel-title'><h3>{escape(title)}</h3>{subtitle_html}</div>", unsafe_allow_html=True)


def status_box(kind: str, message: str) -> None:
    st.markdown(f"<div class='status-box {kind}'>{escape(message)}</div>", unsafe_allow_html=True)


def workflow_steps(active: int = 1) -> None:
    labels = ["Upload media", "Review settings", "Run analysis", "Read signals"]
    items = []
    for index, label in enumerate(labels, start=1):
        state = "active" if index == active else "done" if index < active else ""
        items.append(f"<div class='workflow-step {state}'><b>{index}</b><span>{label}</span></div>")
    st.markdown(f"<div class='workflow'>{''.join(items)}</div>", unsafe_allow_html=True)


def empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
          <div>
            <span class="empty-kicker">Start here</span>
            <h2>Upload a face image or short video</h2>
            <p>The app will keep processing local, resize large media for speed, and analyze only the most confident face.</p>
          </div>
          <div class="empty-list">
            <span>Supported: JPG, PNG, MP4, AVI, MOV</span>
            <span>Default video sampling: every 5th frame</span>
            <span>No training or cloud API dependency</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_note(title: str, lines: list[str]) -> None:
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    st.markdown(
        f"""
        <div class="sidebar-note">
          <strong>{escape(title)}</strong>
          <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
