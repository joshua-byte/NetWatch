"""
app.py
------
Measure-Based Anomaly Detection Tool
Streamlit application entry point.

Run:
    streamlit run app.py
"""

import time
import numpy as np
import pandas as pd
import streamlit as st

from detector import (
    MeasureAnomalyDetector,
    DetectionParams,
    DetectionResult,
    load_csv,
    get_numeric_columns,
)
from capture import generate_synthetic_traffic, SCAPY_AVAILABLE
from dashboard import (
    fig_singular_measure,
    fig_deviation_field,
    fig_severity_donut,
    fig_top_events,
    fig_anomaly_rate,
    fig_signal_histogram,
    fig_live_sparkline,
    events_dataframe,
)
from report import generate_pdf_report


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Anomaly Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #2e3147;
}

/* Main background */
.stApp { background: #0f1117; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2e3147;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* Tab bar */
[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid #2e3147;
}

/* Buttons */
.stButton > button {
    background: #1a1d27;
    border: 1px solid #4a9eff;
    color: #4a9eff;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    padding: 8px 20px;
    transition: all 0.15s ease;
}
.stButton > button:hover {
    background: #4a9eff;
    color: #0f1117;
}

/* Alert badge */
.threat-badge {
    display: inline-block;
    padding: 4px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* Section header */
.section-head {
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a9eff;
    margin-bottom: 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid #2e3147;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #2e3147 !important;
    border-radius: 6px;
}

/* Scrollable code block */
.log-box {
    background: #0d1117;
    border: 1px solid #2e3147;
    border-radius: 6px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #64748b;
    max-height: 180px;
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------

def _init_state():
    defaults = {
        "result": None,
        "df": None,
        "mode": "csv",
        "live_running": False,
        "live_logs": [],
        "live_cap": None,          # PacketCapture instance persisted across reruns
        "live_ts": [],             # rolling timestamp list
        "live_bps": [],            # rolling bytes/s list
        "live_last_detect": 0.0,   # time of last detection run
        "live_render_tick": 0,      # increments each rerun to guarantee unique keys
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH": "🟠",
    "MEDIUM": "🟡",
    "LOW": "🟢",
}

THREAT_CSS = {
    "NONE":     "background:#14532d; color:#4ade80;",
    "LOW":      "background:#14532d; color:#4ade80;",
    "MEDIUM":   "background:#713f12; color:#fde047;",
    "HIGH":     "background:#7c2d12; color:#fb923c;",
    "CRITICAL": "background:#7f1d1d; color:#f87171;",
}

def _overall_threat(result: DetectionResult) -> str:
    if result.n_events == 0:
        return "NONE"
    if result.critical_count > 0:
        return "CRITICAL"
    if result.high_count > 0:
        return "HIGH"
    medium = sum(1 for e in result.events if e.severity == "MEDIUM")
    if medium > 0:
        return "MEDIUM"
    return "LOW"


def _run_detection(x: np.ndarray, params: DetectionParams, signal_col: str, timestamps=None):
    """Run detection and store result in session state."""
    with st.spinner("Running measure decomposition…"):
        detector = MeasureAnomalyDetector(params)
        result = detector.detect(x, timestamps=timestamps, signal_column=signal_col)
    st.session_state.result = result


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div style="font-size:20px; font-weight:700; color:#4a9eff; margin-bottom:2px;">⚡ ANOMALY DETECTOR</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:10px; color:#64748b; letter-spacing:1px; margin-bottom:16px;">MEASURE-BASED · DIRAC DELTA FRAMEWORK</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Mode selector
    mode = st.radio(
        "Input mode",
        ["📂  CSV file", "🎲  Synthetic demo", "📡  Live capture"],
        index=0,
    )

    st.markdown("---")
    st.markdown('<div class="section-head">Detection Parameters</div>', unsafe_allow_html=True)

    bg_window = st.slider("Background window", 50, 2000, 500, 50,
                          help="Coarse-graining scale for μ(t) and σ(t)")
    ev_window = st.slider("Event separation", 10, 500, 200, 10,
                          help="Minimum samples between impulses")
    k_thresh = st.slider("K threshold (σ)", 1.5, 8.0, 3.0, 0.5,
                         help="Singular support condition |z(t)| > K")

    params = DetectionParams(
        background_window=bg_window,
        event_window=ev_window,
        k_threshold=k_thresh,
    )

    st.markdown("---")
    st.markdown('<div class="section-head">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px; color:#64748b; line-height:1.8;">
    x(t) = f(t)dt + Σ aᵢ δ(t−tᵢ)<br><br>
    • Robust rolling median background<br>
    • MAD-based scale (Gaussian consistent)<br>
    • Contiguous support extraction<br>
    • Sparse singular measure output<br><br>
    <a href="https://github.com/joshua-byte/Measure-Based-Anomaly-Detection"
       style="color:#4a9eff; text-decoration:none;">
    ↗ GitHub
    </a>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------

st.markdown(
    '<h1 style="font-size:26px; font-weight:700; color:#e2e8f0; margin-bottom:2px;">'
    'Network Traffic Anomaly Detection'
    '</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:11px; color:#64748b; margin-bottom:20px;">'
    'Singular measure decomposition · Dirac delta analog framework'
    '</p>',
    unsafe_allow_html=True,
)

# ── CSV Mode ──────────────────────────────────────────────────────

if "CSV" in mode:
    uploaded = st.file_uploader(
        "Upload a CSV (CICIDS format or any numeric traffic log)",
        type=["csv"],
        help="Works with CICIDS2017, custom traffic exports, or any CSV with numeric columns.",
    )

    if uploaded:
        with st.spinner("Loading CSV…"):
            df = pd.read_csv(uploaded, low_memory=False)
            df.columns = df.columns.str.strip()
        st.session_state.df = df

        numeric_cols = get_numeric_columns(df)
        if not numeric_cols:
            st.error("No numeric columns found in the CSV.")
        else:
            col_choice = st.selectbox("Signal column", numeric_cols,
                                      index=0 if "Flow Bytes/s" not in numeric_cols else numeric_cols.index("Flow Bytes/s"))

            st.markdown(f'<div style="font-size:10px; color:#64748b; margin-bottom:8px;">'
                        f'{df.shape[0]:,} rows · {df.shape[1]} columns · '
                        f'Column: <b style="color:#4a9eff">{col_choice}</b></div>',
                        unsafe_allow_html=True)

            if st.button("▶  Run Detection", use_container_width=True):
                x = df[col_choice].dropna().values.astype(float)
                _run_detection(x, params, col_choice)

# ── Synthetic Demo Mode ───────────────────────────────────────────

elif "Synthetic" in mode:
    c1, c2, c3 = st.columns(3)
    n_samples = c1.number_input("Samples", 500, 20000, 3000, 500)
    n_anomalies = c2.number_input("Planted anomalies", 1, 50, 10, 1)
    noise_level = c3.slider("Noise level", 0.1, 3.0, 1.0, 0.1)

    if st.button("▶  Generate & Detect", use_container_width=True):
        ts, x = generate_synthetic_traffic(
            n=n_samples,
            noise_scale=8000 * noise_level,
            n_anomalies=n_anomalies,
        )
        _run_detection(x, params, "Bytes/s (synthetic)", timestamps=ts)

# ── Live Capture Mode ─────────────────────────────────────────────

elif "Live" in mode:
    if not SCAPY_AVAILABLE:
        st.warning(
            "**Scapy is not installed.** Install with:\n\n"
            "```\npip install scapy\n```\n\n"
            "Live capture also requires root / administrator privileges."
        )
    else:
        st.info(
            "⚠️  Live capture requires **root / sudo** on Linux/macOS, "
            "or **admin privileges** on Windows."
        )

        ifaces = ["(auto)"]
        try:
            from scapy.all import get_if_list
            ifaces += get_if_list()
        except Exception:
            pass

        iface_choice = st.selectbox("Interface", ifaces)
        iface = None if iface_choice == "(auto)" else iface_choice
        detect_interval = st.slider("Detection interval (s)", 2, 30, 5, 1)

        col_start, col_stop = st.columns(2)

        if col_start.button("▶  Start Capture", use_container_width=True):
            st.session_state.live_running = True
            st.session_state.live_logs = []

        if col_stop.button("⏹  Stop", use_container_width=True):
            st.session_state.live_running = False

        # ── Start / Stop ──────────────────────────────────────────────
        if st.session_state.live_running and st.session_state.live_cap is None:
            from capture import PacketCapture
            cap = PacketCapture(iface=iface, detect_interval=detect_interval, params=params)
            cap.start()
            st.session_state.live_cap = cap
            st.session_state.live_ts = []
            st.session_state.live_bps = []
            st.session_state.live_last_detect = time.time()

        if not st.session_state.live_running and st.session_state.live_cap is not None:
            st.session_state.live_cap.stop()
            st.session_state.live_cap = None

        # ── Live dashboard (renders on every rerun) ────────────────
        if st.session_state.live_running and st.session_state.live_cap is not None:
            st.session_state.live_render_tick += 1
            _tick = st.session_state.live_render_tick
            cap = st.session_state.live_cap

            # Drain new per-second buckets from the capture deque into session lists
            ts_deque, bps_deque = cap.get_series()
            st.session_state.live_ts = list(ts_deque)
            st.session_state.live_bps = list(bps_deque)

            # Run detection if enough data and interval has passed
            now = time.time()
            bps_arr = np.array(st.session_state.live_bps, dtype=float)
            min_samples = max(10, params.background_window // 10)
            if (len(bps_arr) >= min_samples and
                    now - st.session_state.live_last_detect >= detect_interval):
                try:
                    from detector import MeasureAnomalyDetector
                    live_detector = MeasureAnomalyDetector(params)
                    live_result = live_detector.detect(bps_arr, signal_column="Bytes/s (live)")
                    st.session_state.result = live_result
                    st.session_state.live_last_detect = now
                    if live_result.n_events:
                        st.session_state.live_logs.append(
                            f"[{time.strftime('%H:%M:%S')}] {live_result.n_events} event(s) "
                            f"| critical={live_result.critical_count} "
                            f"high={live_result.high_count}"
                        )
                except Exception as e:
                    st.session_state.live_logs.append(f"[ERR] {e}")

            # ── Live stats bar ─────────────────────────────────────
            s = cap.stats
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Packets", f"{s.total_packets:,}")
            m2.metric("Bytes", f"{s.total_bytes / 1e6:.1f} MB")
            m3.metric("Anomalies", s.anomalies_detected)
            m4.metric("Elapsed", f"{s.elapsed:.0f}s")
            m5.metric("Samples", len(st.session_state.live_bps))

            # ── Live sparkline (always visible) ───────────────────
            ts_arr = np.array(st.session_state.live_ts, dtype=float)
            bps_arr = np.array(st.session_state.live_bps, dtype=float)
            ev_idx = [e.index for e in (st.session_state.result.events
                                        if st.session_state.result else [])]
            if len(bps_arr) > 2:
                st.plotly_chart(
                    fig_live_sparkline(ts_arr, bps_arr, ev_idx),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"live_sparkline_{_tick}",
                )

            # ── Analysis tabs (update each detection cycle) ────────
            if st.session_state.result is not None:
                live_r = st.session_state.result
                st.markdown(
                    f'<div style="font-size:11px; color:#64748b; margin: 4px 0 8px;">' +
                    f'Last analysis: {live_r.n_samples} samples · ' +
                    f'{live_r.n_events} event(s) · {live_r.duration_ms:.1f} ms' +
                    '</div>',
                    unsafe_allow_html=True,
                )
                lt1, lt2, lt3 = st.tabs(["📈 Signal", "📊 Deviation", "📋 Events"])
                with lt1:
                    st.plotly_chart(fig_singular_measure(live_r),
                                    use_container_width=True, key=f"live_singular_{_tick}")
                with lt2:
                    st.plotly_chart(fig_deviation_field(live_r),
                                    use_container_width=True, key=f"live_deviation_{_tick}")
                with lt3:
                    st.dataframe(events_dataframe(live_r),
                                 use_container_width=True, hide_index=True)

            # ── Event log ─────────────────────────────────────────
            logs = st.session_state.live_logs[-20:]
            log_html = "".join(
                f'<div style="color:{"#f87171" if "critical" in l.lower() else "#64748b"}">' +
                l + '</div>'
                for l in reversed(logs)
            )
            st.markdown(
                f'<div class="log-box">{log_html or "<i>Waiting for traffic…</i>"}</div>',
                unsafe_allow_html=True,
            )

            # ── Schedule next rerun ───────────────────────────────
            time.sleep(1)
            st.rerun()


# ------------------------------------------------------------------
# Results display
# ------------------------------------------------------------------

result: DetectionResult | None = st.session_state.result

# In live mode the inline tabs already show everything — skip to avoid duplicate keys
if result is not None and not st.session_state.live_running:
    st.markdown("---")

    # Threat banner
    threat = _overall_threat(result)
    css = THREAT_CSS[threat]
    st.markdown(
        f'<div class="threat-badge" style="{css}">⚡ Threat level: {threat}</div>'
        f'<span style="font-size:11px; color:#64748b; margin-left:12px;">'
        f'{result.n_events} event(s) in {result.n_samples:,} samples · '
        f'{result.duration_ms:.1f} ms analysis time</span>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Total events", result.n_events)
    k2.metric("🔴 Critical", result.critical_count)
    k3.metric("🟠 High", result.high_count)
    k4.metric("🟡 Medium", sum(1 for e in result.events if e.severity == "MEDIUM"))
    k5.metric("🟢 Low", sum(1 for e in result.events if e.severity == "LOW"))
    k6.metric("Anomaly rate", f"{result.anomaly_rate:.3f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab_main, tab_dev, tab_stats, tab_events, tab_report = st.tabs([
        "📈  Singular Measure",
        "📊  Deviation Field",
        "🔬  Statistics",
        "📋  Events Table",
        "📄  Report",
    ])

    with tab_main:
        st.plotly_chart(fig_singular_measure(result), use_container_width=True, config={"displayModeBar": True}, key="chart_singular_measure")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(fig_severity_donut(result), use_container_width=True, config={"displayModeBar": False}, key="chart_severity_donut")
        with c2:
            st.plotly_chart(fig_top_events(result, top_n=12), use_container_width=True, config={"displayModeBar": False}, key="chart_top_events")

    with tab_dev:
        st.plotly_chart(fig_deviation_field(result), use_container_width=True, config={"displayModeBar": True}, key="chart_deviation_field")
        st.plotly_chart(fig_anomaly_rate(result), use_container_width=True, config={"displayModeBar": False}, key="chart_anomaly_rate")

    with tab_stats:
        st.plotly_chart(fig_signal_histogram(result), use_container_width=True, config={"displayModeBar": False}, key="chart_signal_histogram")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-head">Signal statistics</div>', unsafe_allow_html=True)
            stats = {
                "Mean": f"{np.mean(result.signal):,.2f}",
                "Median": f"{np.median(result.signal):,.2f}",
                "Std dev": f"{np.std(result.signal):,.2f}",
                "Min": f"{np.min(result.signal):,.2f}",
                "Max": f"{np.max(result.signal):,.2f}",
            }
            for k, v in stats.items():
                c1, c2 = st.columns([2, 3])
                c1.markdown(f'<span style="font-size:11px; color:#64748b;">{k}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span style="font-size:11px; color:#e2e8f0;">{v}</span>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="section-head">z-field statistics</div>', unsafe_allow_html=True)
            z = result.z_field
            zstats = {
                "Peak |z|": f"{np.max(np.abs(z)):.2f}σ",
                "Mean |z|": f"{np.mean(np.abs(z)):.2f}σ",
                "Support fraction": f"{result.support_mask.mean() * 100:.2f}%",
                "K threshold": f"{result.params.k_threshold}σ",
            }
            for k, v in zstats.items():
                c1, c2 = st.columns([2, 3])
                c1.markdown(f'<span style="font-size:11px; color:#64748b;">{k}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span style="font-size:11px; color:#e2e8f0;">{v}</span>', unsafe_allow_html=True)

    with tab_events:
        df_events = events_dataframe(result)

        sev_filter = st.multiselect(
            "Filter by severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        )
        df_filtered = df_events[df_events["Severity"].isin(sev_filter)] if not df_events.empty else df_events

        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Severity": st.column_config.TextColumn(width="small"),
                "z-score": st.column_config.NumberColumn(format="%.2f"),
                "Mass": st.column_config.NumberColumn(format="%.4f"),
            },
        )

        if not df_filtered.empty:
            csv_bytes = df_filtered.to_csv(index=False).encode()
            st.download_button(
                "⬇  Download events CSV",
                data=csv_bytes,
                file_name="anomaly_events.csv",
                mime="text/csv",
            )

    with tab_report:
        st.markdown(
            '<p style="font-size:12px; color:#64748b; margin-bottom:16px;">'
            'Generate a full PDF report with summary, methodology, and events table.</p>',
            unsafe_allow_html=True,
        )
        report_title = st.text_input("Report title", "Network Anomaly Detection Report")
        top_n = st.slider("Max events in report", 5, 100, 25, 5)

        if st.button("📄  Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF…"):
                pdf_bytes = generate_pdf_report(result, title=report_title, top_n_events=top_n)

            st.download_button(
                "⬇  Download PDF",
                data=pdf_bytes,
                file_name="anomaly_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("Report ready — click above to download.")

else:
    # Landing state
    st.markdown("""
    <div style="
        text-align: center;
        padding: 60px 40px;
        border: 1px dashed #2e3147;
        border-radius: 12px;
        margin-top: 40px;
    ">
        <div style="font-size: 48px; margin-bottom: 16px;">⚡</div>
        <div style="font-size: 18px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px;">
            Ready to analyse
        </div>
        <div style="font-size: 12px; color: #64748b; max-width: 460px; margin: 0 auto; line-height: 1.8;">
            Upload a CSV, run the synthetic demo, or start a live capture.<br>
            The singular measure decomposition pipeline will detect anomalous events<br>
            as Dirac-analog impulses relative to the background traffic field.
        </div>
    </div>
    """, unsafe_allow_html=True)