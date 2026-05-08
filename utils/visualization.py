from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def emotion_bar_chart(emotions: dict[str, float]) -> go.Figure:
    data = pd.DataFrame(
        {"emotion": list(emotions.keys()), "confidence": [float(v) for v in emotions.values()]}
    ).sort_values("confidence", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=data["confidence"],
            y=data["emotion"],
            orientation="h",
            marker=dict(color=data["confidence"], colorscale="Tealgrn", line=dict(color="#19d5ff", width=1)),
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(l=12, r=12, t=24, b=18),
        xaxis=dict(range=[0, 100], title="Confidence %", gridcolor="rgba(25,213,255,.15)"),
        yaxis=dict(title=""),
        font=dict(color="#d9fbff"),
    )
    return fig


def attention_gauge(score: int) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"color": "#e7fdff", "size": 38}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#7eeaff"},
                "bar": {"color": "#19d5ff"},
                "bgcolor": "rgba(255,255,255,.05)",
                "borderwidth": 1,
                "bordercolor": "rgba(126,234,255,.35)",
                "steps": [
                    {"range": [0, 35], "color": "rgba(255,82,119,.35)"},
                    {"range": [35, 55], "color": "rgba(255,188,66,.32)"},
                    {"range": [55, 78], "color": "rgba(95,220,170,.28)"},
                    {"range": [78, 100], "color": "rgba(25,213,255,.28)"},
                ],
            },
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=290,
        margin=dict(l=18, r=18, t=20, b=8),
        font=dict(color="#d9fbff"),
    )
    return fig


def video_summary_chart(summary: dict) -> go.Figure:
    labels = ["Attention", "Face detection", "Frames analyzed"]
    values = [
        float(summary.get("average_attention", 0)),
        float(summary.get("face_detection_rate", 0)),
        min(100.0, float(summary.get("sampled_frames", 0)) * 4.0),
    ]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=["#19d5ff", "#7cffcb", "#ffcf5a"],
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
        margin=dict(l=12, r=12, t=24, b=28),
        yaxis=dict(range=[0, 100], gridcolor="rgba(25,213,255,.15)"),
        font=dict(color="#d9fbff"),
    )
    return fig
