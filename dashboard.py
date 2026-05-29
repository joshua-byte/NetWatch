"""
dashboard.py
------------
Plotly figure builders for the Streamlit dashboard.

All functions accept a DetectionResult and return a go.Figure
(or list of figures) ready to be passed to st.plotly_chart().
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from detector import DetectionResult, AnomalyEvent

# ------------------------------------------------------------------
# Theme constants
# ------------------------------------------------------------------

PALETTE = {
    "bg": "#0f1117",
    "surface": "#1a1d27",
    "border": "#2e3147",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "signal": "#4a9eff",
    "background_line": "#38bdf8",
    "sigma_band": "rgba(56,189,248,0.08)",
    "support": "rgba(251,191,36,0.15)",
    "event_critical": "#ef4444",
    "event_high": "#f97316",
    "event_medium": "#eab308",
    "event_low": "#22c55e",
}

SEVERITY_COLORS = {
    "CRITICAL": PALETTE["event_critical"],
    "HIGH": PALETTE["event_high"],
    "MEDIUM": PALETTE["event_medium"],
    "LOW": PALETTE["event_low"],
}

LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'JetBrains Mono', 'Courier New', monospace", color=PALETTE["text"], size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(
        showgrid=True, gridcolor=PALETTE["border"], gridwidth=0.5,
        zeroline=False, linecolor=PALETTE["border"],
    ),
    yaxis=dict(
        showgrid=True, gridcolor=PALETTE["border"], gridwidth=0.5,
        zeroline=False, linecolor=PALETTE["border"],
    ),
)


def _apply_base(fig: go.Figure, title: str = "", height: int = 380) -> go.Figure:
    fig.update_layout(**LAYOUT_BASE, title=dict(text=title, font=dict(size=13, color=PALETTE["text"])), height=height)
    return fig


# ------------------------------------------------------------------
# 1. Main signal + singular measure overview
# ------------------------------------------------------------------

def fig_singular_measure(result: DetectionResult) -> go.Figure:
    """
    Dual-axis chart:
      Left  → raw signal + robust background ± 3σ band
      Right → singular event measure (Dirac analog impulses)
    """
    n = result.n_samples
    idx = np.arange(n)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ±3σ confidence band
    upper = result.background + result.params.k_threshold * result.sigma
    lower = result.background - result.params.k_threshold * result.sigma

    fig.add_trace(go.Scatter(
        x=np.concatenate([idx, idx[::-1]]),
        y=np.concatenate([upper, lower[::-1]]),
        fill="toself",
        fillcolor=PALETTE["sigma_band"],
        line=dict(color="rgba(0,0,0,0)"),
        name=f"±{result.params.k_threshold}σ band",
        hoverinfo="skip",
    ), secondary_y=False)

    # Raw signal
    fig.add_trace(go.Scatter(
        x=idx, y=result.signal,
        line=dict(color=PALETTE["signal"], width=1, dash="solid"),
        opacity=0.7,
        name=f"Signal ({result.signal_column})",
    ), secondary_y=False)

    # Background
    fig.add_trace(go.Scatter(
        x=idx, y=result.background,
        line=dict(color=PALETTE["background_line"], width=2),
        name="Background μ(t)",
    ), secondary_y=False)

    # Singular measure impulses — colour-coded by severity
    if result.events:
        event_idx = np.array([e.index for e in result.events])
        event_mass = np.array([e.mass for e in result.events])
        event_colors = [SEVERITY_COLORS[e.severity] for e in result.events]
        event_text = [
            f"Index: {e.index}<br>Severity: {e.severity}<br>"
            f"z-score: {e.z_score:.2f}σ<br>Mass: {e.mass:.3f}<br>"
            f"Raw: {e.raw_value:,.0f}"
            for e in result.events
        ]

        fig.add_trace(go.Bar(
            x=event_idx,
            y=event_mass,
            marker_color=event_colors,
            name="δ-Events (singular measure)",
            hovertext=event_text,
            hoverinfo="text",
            width=3,
            opacity=0.9,
        ), secondary_y=True)

    fig.update_yaxes(
        title_text=result.signal_column,
        secondary_y=False,
        gridcolor=PALETTE["border"],
        showgrid=True,
        zeroline=False,
        linecolor=PALETTE["border"],
        color=PALETTE["text"],
    )
    fig.update_yaxes(
        title_text="Event mass (σ-normalized)",
        secondary_y=True,
        gridcolor="rgba(0,0,0,0)",
        showgrid=False,
        zeroline=False,
        color=PALETTE["text"],
    )
    fig.update_xaxes(title_text="Sample index", color=PALETTE["text"])

    return _apply_base(
        fig,
        title="Singular Measure Decomposition  ·  x(t) = f(t)dt + Σ aᵢ δ(t − tᵢ)",
        height=420,
    )


# ------------------------------------------------------------------
# 2. Deviation field z(t)
# ------------------------------------------------------------------

def fig_deviation_field(result: DetectionResult) -> go.Figure:
    n = result.n_samples
    idx = np.arange(n)
    K = result.params.k_threshold
    z = result.z_field

    fig = go.Figure()

    # Support region fill (yellow wash where |z|>K)
    support_y = np.where(result.support_mask, np.abs(z), 0)
    fig.add_trace(go.Scatter(
        x=idx, y=support_y,
        fill="tozeroy",
        fillcolor="rgba(251,191,36,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Singular support |z|>K",
        hoverinfo="skip",
    ))

    # z-field
    fig.add_trace(go.Scatter(
        x=idx, y=z,
        line=dict(color=PALETTE["signal"], width=1),
        name="Deviation z(t)",
    ))

    # ±K threshold lines
    for sign, label in [(K, f"+{K}σ threshold"), (-K, f"−{K}σ threshold")]:
        fig.add_hline(
            y=sign,
            line_dash="dash",
            line_color=PALETTE["event_medium"],
            line_width=1,
            annotation_text=label,
            annotation_font_color=PALETTE["event_medium"],
        )

    fig.update_xaxes(title_text="Sample index")
    fig.update_yaxes(title_text="z(t) = (x − μ) / σ")
    return _apply_base(fig, title="Deviation Field  z(t)", height=320)


# ------------------------------------------------------------------
# 3. Event severity distribution (donut)
# ------------------------------------------------------------------

def fig_severity_donut(result: DetectionResult) -> go.Figure:
    if not result.events:
        fig = go.Figure()
        fig.add_annotation(
            text="No anomalies detected",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=PALETTE["muted"], size=14),
        )
        return _apply_base(fig, height=280)

    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for e in result.events:
        counts[e.severity] += 1

    labels = [k for k, v in counts.items() if v > 0]
    values = [counts[k] for k in labels]
    colors = [SEVERITY_COLORS[k] for k in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="%{label}: %{value} events<extra></extra>",
    ))

    fig.add_annotation(
        text=f"<b>{result.n_events}</b><br>events",
        xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=PALETTE["text"], size=13),
        align="center",
    )

    return _apply_base(fig, title="Severity Distribution", height=300)


# ------------------------------------------------------------------
# 4. Top-N events bar chart
# ------------------------------------------------------------------

def fig_top_events(result: DetectionResult, top_n: int = 15) -> go.Figure:
    events = result.events[:top_n]
    if not events:
        fig = go.Figure()
        fig.add_annotation(text="No events", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=PALETTE["muted"]))
        return _apply_base(fig, height=300)

    labels = [f"#{i+1}  idx={e.index}" for i, e in enumerate(events)]
    masses = [e.mass for e in events]
    colors = [SEVERITY_COLORS[e.severity] for e in events]
    hover = [
        f"Rank #{i+1}<br>Index: {e.index}<br>z-score: {e.z_score:.2f}σ<br>"
        f"Mass: {e.mass:.3f}<br>Severity: {e.severity}"
        for i, e in enumerate(events)
    ]

    fig = go.Figure(go.Bar(
        x=masses,
        y=labels,
        orientation="h",
        marker_color=colors,
        hovertext=hover,
        hoverinfo="text",
        text=[f"{m:.2f}" for m in masses],
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.update_xaxes(title_text="Event mass (σ-normalized)")
    fig.update_yaxes(autorange="reversed")
    return _apply_base(fig, title=f"Top {len(events)} Events by Mass", height=max(300, len(events) * 28 + 60))


# ------------------------------------------------------------------
# 5. Rolling anomaly rate over time
# ------------------------------------------------------------------

def fig_anomaly_rate(result: DetectionResult, bin_size: int = 100) -> go.Figure:
    if result.n_samples < bin_size:
        fig = go.Figure()
        fig.add_annotation(text="Need more samples", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=PALETTE["muted"]))
        return _apply_base(fig, height=220)

    # Count events per bin
    n_bins = result.n_samples // bin_size
    bins = np.arange(n_bins) * bin_size
    counts = np.zeros(n_bins)
    for e in result.events:
        b = e.index // bin_size
        if b < n_bins:
            counts[b] += 1

    fig = go.Figure(go.Scatter(
        x=bins,
        y=counts,
        fill="tozeroy",
        fillcolor="rgba(74,158,255,0.12)",
        line=dict(color=PALETTE["signal"], width=1.5),
        mode="lines",
        name="Events per window",
    ))

    fig.update_xaxes(title_text="Sample index")
    fig.update_yaxes(title_text="Event count")
    return _apply_base(fig, title=f"Event Density  (bin = {bin_size} samples)", height=240)


# ------------------------------------------------------------------
# 6. Signal statistics histogram
# ------------------------------------------------------------------

def fig_signal_histogram(result: DetectionResult) -> go.Figure:
    x = result.signal
    z = result.z_field

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Signal distribution", "z-field distribution"])

    fig.add_trace(go.Histogram(
        x=x, nbinsx=80,
        marker_color=PALETTE["signal"], opacity=0.7,
        name="Signal",
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=z, nbinsx=80,
        marker_color=PALETTE["background_line"], opacity=0.7,
        name="z-field",
    ), row=1, col=2)

    # Vertical lines for thresholds on z histogram
    K = result.params.k_threshold
    for sign in [K, -K]:
        fig.add_vline(
            x=sign, line_dash="dash",
            line_color=PALETTE["event_medium"],
            line_width=1, row=1, col=2
        )

    fig.update_layout(**LAYOUT_BASE, height=300, showlegend=False)
    return fig


# ------------------------------------------------------------------
# 7. Live capture sparkline
# ------------------------------------------------------------------

def fig_live_sparkline(
    timestamps: np.ndarray,
    byte_series: np.ndarray,
    events_idx: list[int] | None = None,
) -> go.Figure:
    """Compact live-mode sparkline for the capture dashboard."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=byte_series,
        line=dict(color=PALETTE["signal"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(74,158,255,0.08)",
        name="Bytes/s",
    ))

    if events_idx:
        ev_x = [timestamps[i] for i in events_idx if i < len(timestamps)]
        ev_y = [byte_series[i] for i in events_idx if i < len(timestamps)]
        fig.add_trace(go.Scatter(
            x=ev_x, y=ev_y,
            mode="markers",
            marker=dict(color=PALETTE["event_critical"], size=8, symbol="x"),
            name="Anomaly",
        ))

    fig.update_xaxes(title_text="Time", showticklabels=False)
    fig.update_yaxes(title_text="Bytes/s")
    return _apply_base(fig, title="Live Traffic  (Bytes/s)", height=200)


# ------------------------------------------------------------------
# 8. Events table helper
# ------------------------------------------------------------------

def events_dataframe(result: DetectionResult) -> pd.DataFrame:
    """Build a clean DataFrame from detected events for st.dataframe()."""
    if not result.events:
        return pd.DataFrame(columns=["Rank", "Index", "Severity", "z-score", "Mass", "Raw value", "Background"])

    rows = []
    for i, e in enumerate(result.events, 1):
        rows.append({
            "Rank": i,
            "Index": e.index,
            "Severity": e.severity,
            "z-score": round(e.z_score, 2),
            "Mass": round(e.mass, 4),
            "Raw value": round(e.raw_value, 2),
            "Background": round(e.background, 2),
        })
    return pd.DataFrame(rows)