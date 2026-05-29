"""
report.py
---------
PDF report generator for anomaly detection results.

Produces a structured report with:
    • Executive summary (threat level, event counts)
    • Detection parameters used
    • Top-N events table
    • Severity breakdown
    • Methodology section

Usage:
    from report import generate_pdf_report
    pdf_bytes = generate_pdf_report(result, title="Traffic Analysis Report")
"""

import io
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable

from detector import DetectionResult, AnomalyEvent


# ------------------------------------------------------------------
# Colour palette
# ------------------------------------------------------------------

C_BG        = colors.HexColor("#0f1117")
C_SURFACE   = colors.HexColor("#1a1d27")
C_ACCENT    = colors.HexColor("#4a9eff")
C_TEXT      = colors.HexColor("#e2e8f0")
C_MUTED     = colors.HexColor("#64748b")
C_CRITICAL  = colors.HexColor("#ef4444")
C_HIGH      = colors.HexColor("#f97316")
C_MEDIUM    = colors.HexColor("#eab308")
C_LOW       = colors.HexColor("#22c55e")
C_WHITE     = colors.white
C_DARK      = colors.HexColor("#0d1117")

SEV_COLORS = {
    "CRITICAL": C_CRITICAL,
    "HIGH":     C_HIGH,
    "MEDIUM":   C_MEDIUM,
    "LOW":      C_LOW,
}

THREAT_LEVELS = {
    "NONE":     (colors.HexColor("#22c55e"), "NO THREAT DETECTED"),
    "LOW":      (colors.HexColor("#22c55e"), "LOW"),
    "MEDIUM":   (colors.HexColor("#eab308"), "MEDIUM"),
    "HIGH":     (colors.HexColor("#f97316"), "HIGH"),
    "CRITICAL": (colors.HexColor("#ef4444"), "CRITICAL"),
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


# ------------------------------------------------------------------
# Style helpers
# ------------------------------------------------------------------

def _build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["title"] = ParagraphStyle(
        "ReportTitle",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=C_TEXT,
        leading=28,
        alignment=TA_LEFT,
    )
    styles["subtitle"] = ParagraphStyle(
        "ReportSubtitle",
        fontName="Helvetica",
        fontSize=11,
        textColor=C_MUTED,
        leading=16,
        spaceAfter=6,
    )
    styles["section"] = ParagraphStyle(
        "SectionHead",
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=C_ACCENT,
        leading=18,
        spaceBefore=14,
        spaceAfter=6,
        borderPad=4,
    )
    styles["body"] = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=C_TEXT,
        leading=14,
        spaceAfter=4,
    )
    styles["body_muted"] = ParagraphStyle(
        "BodyMuted",
        fontName="Helvetica",
        fontSize=9,
        textColor=C_MUTED,
        leading=13,
        spaceAfter=3,
    )
    styles["mono"] = ParagraphStyle(
        "Mono",
        fontName="Courier",
        fontSize=8.5,
        textColor=C_TEXT,
        leading=12,
        spaceAfter=2,
    )
    styles["table_header"] = ParagraphStyle(
        "TH",
        fontName="Helvetica-Bold",
        fontSize=8.5,
        textColor=C_WHITE,
        alignment=TA_CENTER,
    )
    styles["table_cell"] = ParagraphStyle(
        "TC",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=C_TEXT,
        alignment=TA_LEFT,
    )
    styles["table_cell_center"] = ParagraphStyle(
        "TCC",
        fontName="Helvetica",
        fontSize=8.5,
        textColor=C_TEXT,
        alignment=TA_CENTER,
    )

    return styles


# ------------------------------------------------------------------
# Report sections
# ------------------------------------------------------------------

def _header_section(result: DetectionResult, title: str, styles: dict) -> list:
    threat = _overall_threat(result)
    threat_color, threat_label = THREAT_LEVELS[threat]
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    elements = []

    # Dark header bar via a 1-row table
    header_data = [[
        Paragraph(title, styles["title"]),
        Paragraph(
            f'<font color="#64748b">Generated: {ts}</font>',
            ParagraphStyle("RH", fontName="Helvetica", fontSize=9, textColor=C_MUTED,
                           alignment=TA_RIGHT),
        ),
    ]]
    t = Table(header_data, colWidths=["70%", "30%"])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_DARK),
        ("TOPPADDING", (0, 0), (-1, -1), 16),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 16),
        ("LEFTPADDING", (0, 0), (-1, -1), 16),
        ("RIGHTPADDING", (0, 0), (-1, -1), 16),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROUNDEDCORNERS", [6, 6, 6, 6]),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 10))

    # Threat level badge
    badge_data = [[
        Paragraph("OVERALL THREAT LEVEL", ParagraphStyle(
            "BadgeLabel", fontName="Helvetica-Bold", fontSize=8,
            textColor=C_MUTED, alignment=TA_CENTER)),
        Paragraph(threat_label, ParagraphStyle(
            "BadgeVal", fontName="Helvetica-Bold", fontSize=16,
            textColor=threat_color, alignment=TA_CENTER)),
    ]]
    bt = Table(badge_data, colWidths=["50%", "50%"])
    bt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_SURFACE),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BOX", (0, 0), (-1, -1), 0.5, C_MUTED),
    ]))
    elements.append(bt)
    elements.append(Spacer(1, 12))

    return elements


def _summary_cards(result: DetectionResult, styles: dict) -> list:
    critical = result.critical_count
    high = result.high_count
    medium = sum(1 for e in result.events if e.severity == "MEDIUM")
    low = sum(1 for e in result.events if e.severity == "LOW")

    top_z = max((e.z_score for e in result.events), default=0)

    def card(label, value, color):
        return [
            Paragraph(label, ParagraphStyle("CL", fontName="Helvetica", fontSize=7.5,
                                            textColor=C_MUTED, alignment=TA_CENTER)),
            Paragraph(str(value), ParagraphStyle("CV", fontName="Helvetica-Bold", fontSize=20,
                                                 textColor=color, alignment=TA_CENTER)),
        ]

    data = [
        card("TOTAL EVENTS", result.n_events, C_ACCENT),
        card("CRITICAL", critical, C_CRITICAL),
        card("HIGH", high, C_HIGH),
        card("MEDIUM", medium, C_MEDIUM),
        card("LOW", low, C_LOW),
        card("PEAK z-SCORE", f"{top_z:.1f}σ", C_TEXT),
        card("SAMPLES", f"{result.n_samples:,}", C_TEXT),
        card("ANOMALY RATE", f"{result.anomaly_rate:.3f}%", C_TEXT),
    ]

    row = [item for pair in data for item in [pair]]  # flatten outer list

    # Two rows of 4 cards each
    rows = [data[i : i + 4] for i in range(0, len(data), 4)]
    elements = []
    for row in rows:
        flat = [[cell for pair in row for cell in pair]]
        # Build two-row cell: label on top, value below
        # Use nested tables for each card
        card_tables = []
        for pair in row:
            ct = Table([[pair[0]], [pair[1]]])
            ct.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_SURFACE),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("BOX", (0, 0), (-1, -1), 0.5, C_MUTED),
            ]))
            card_tables.append(ct)

        row_table = Table([card_tables], colWidths=["25%"] * len(card_tables))
        row_table.setStyle(TableStyle([
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(row_table)
        elements.append(Spacer(1, 6))

    return elements


def _parameters_section(result: DetectionResult, styles: dict) -> list:
    p = result.params
    elements = [
        Paragraph("Detection Parameters", styles["section"]),
        HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceAfter=8),
    ]

    rows = [
        ["Parameter", "Value", "Description"],
        ["Background window", str(p.background_window), "Coarse-graining scale (samples)"],
        ["Event window", str(p.event_window), "Minimum impulse separation (samples)"],
        ["K threshold", f"{p.k_threshold}σ", "Singular support condition |z| > K"],
        ["Regularization ε", str(p.eps), "MAD floor to prevent divide-by-zero"],
        ["Signal column", result.signal_column, "Input feature analysed"],
        ["Total samples", f"{result.n_samples:,}", "Length of analysed time series"],
        ["Analysis time", f"{result.duration_ms:.1f} ms", "Wall-clock time for full pipeline"],
    ]

    t = Table(rows, colWidths=["30%", "20%", "50%"])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 0), (0, -1), "Courier"),
        ("BACKGROUND", (0, 1), (-1, -1), C_SURFACE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_SURFACE, C_DARK]),
        ("TEXTCOLOR", (0, 1), (-1, -1), C_TEXT),
        ("GRID", (0, 0), (-1, -1), 0.25, C_MUTED),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(t)
    return elements


def _events_table_section(result: DetectionResult, styles: dict, top_n: int = 25) -> list:
    elements = [
        Spacer(1, 6),
        Paragraph(f"Top {min(top_n, result.n_events)} Anomalous Events", styles["section"]),
        HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceAfter=8),
    ]

    if not result.events:
        elements.append(Paragraph("No anomalous events detected.", styles["body_muted"]))
        return elements

    header = ["#", "Index", "Severity", "z-score", "Mass", "Raw value", "Background"]
    rows = [header]
    for i, e in enumerate(result.events[:top_n], 1):
        rows.append([
            str(i),
            str(e.index),
            e.severity,
            f"{e.z_score:.2f}σ",
            f"{e.mass:.4f}",
            f"{e.raw_value:,.1f}",
            f"{e.background:,.1f}",
        ])

    col_w = ["5%", "10%", "14%", "12%", "12%", "24%", "23%"]
    t = Table(rows, colWidths=col_w)

    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), C_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), C_ACCENT),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BACKGROUND", (0, 1), (-1, -1), C_SURFACE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [C_SURFACE, C_DARK]),
        ("TEXTCOLOR", (0, 1), (-1, -1), C_TEXT),
        ("GRID", (0, 0), (-1, -1), 0.25, C_MUTED),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ALIGN", (2, 1), (2, -1), "CENTER"),
    ]

    # Severity colour coding
    for i, e in enumerate(result.events[:top_n], 1):
        c = SEV_COLORS[e.severity]
        style_cmds.append(("TEXTCOLOR", (2, i), (2, i), c))
        style_cmds.append(("FONTNAME", (2, i), (2, i), "Helvetica-Bold"))

    t.setStyle(TableStyle(style_cmds))
    elements.append(t)
    return elements


def _methodology_section(styles: dict) -> list:
    elements = [
        Spacer(1, 8),
        Paragraph("Methodology", styles["section"]),
        HRFlowable(width="100%", thickness=0.5, color=C_MUTED, spaceAfter=8),
        Paragraph(
            "This report was generated by the <b>Measure-Based Anomaly Detection</b> framework. "
            "Network traffic is modelled as a finite measure on the time axis:",
            styles["body"],
        ),
        Spacer(1, 4),
        Paragraph("x(t)  =  f(t) dt  +  Σᵢ aᵢ δ(t − tᵢ)", styles["mono"]),
        Spacer(1, 6),
        Paragraph(
            "The pipeline executes three traversals over the signal:",
            styles["body"],
        ),
        Paragraph(
            "<b>Traversal 1 — Robust background (absolutely continuous component):</b> "
            "A rolling median f(t) and scale estimate σ(t) are computed using the "
            "Median Absolute Deviation (MAD), scaled by 1.4826 for Gaussian consistency. "
            "This ensures robustness under heavy-tailed bursts.",
            styles["body"],
        ),
        Paragraph(
            "<b>Traversal 2 — Deviation field:</b> "
            "The normalised deviation z(t) = (x − μ) / σ is computed. "
            "Any region where |z(t)| > K (the threshold parameter) constitutes "
            "the singular support of the measure.",
            styles["body"],
        ),
        Paragraph(
            "<b>Traversal 3 — Singular measure extraction:</b> "
            "Each contiguous support region is collapsed to a single impulse δ(t − tᵢ) "
            "at the point of maximal deviation. The event mass aᵢ is the mean |z| "
            "over the region, providing a resolution-normalised anomaly score. "
            "A minimum separation window enforces sparsity of the resulting measure.",
            styles["body"],
        ),
        Spacer(1, 6),
        Paragraph(
            "Severity is assigned by peak z-score: "
            "LOW (≥3σ), MEDIUM (≥5σ), HIGH (≥7σ), CRITICAL (≥10σ).",
            styles["body_muted"],
        ),
    ]
    return elements


def _footer_line(styles: dict) -> list:
    return [
        Spacer(1, 14),
        HRFlowable(width="100%", thickness=0.5, color=C_MUTED),
        Spacer(1, 4),
        Paragraph(
            "Generated by Measure-Based Anomaly Detection Tool  ·  "
            "github.com/joshua-byte/Measure-Based-Anomaly-Detection",
            ParagraphStyle("Footer", fontName="Helvetica", fontSize=7.5,
                           textColor=C_MUTED, alignment=TA_CENTER),
        ),
    ]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def generate_pdf_report(
    result: DetectionResult,
    title: str = "Network Anomaly Detection Report",
    top_n_events: int = 25,
) -> bytes:
    """
    Build a complete PDF report and return it as bytes.

    Parameters
    ----------
    result        : DetectionResult from MeasureAnomalyDetector.detect()
    title         : Report title string
    top_n_events  : Max events to include in the table

    Returns
    -------
    bytes  — PDF content, ready for st.download_button() or file write
    """
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=title,
        author="Measure-Based Anomaly Detector",
        subject="Network Traffic Anomaly Detection Report",
    )

    styles = _build_styles()
    story = []

    story += _header_section(result, title, styles)
    story.append(Spacer(1, 4))
    story += _summary_cards(result, styles)
    story.append(Spacer(1, 6))
    story += _parameters_section(result, styles)
    story += _events_table_section(result, styles, top_n=top_n_events)
    story += _methodology_section(styles)
    story += _footer_line(styles)

    doc.build(story)
    return buf.getvalue()


def save_pdf_report(
    result: DetectionResult,
    path: str,
    title: str = "Network Anomaly Detection Report",
) -> str:
    """Write PDF report to disk. Returns the path written."""
    pdf_bytes = generate_pdf_report(result, title=title)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path