"""AI-SOC design system: palette, CSS shell, and reusable UI components.

Presentation-only module. It contains no machine-learning logic and never
invents data - every value shown through these components is passed in by a
page module that read it from the real backend (models, metrics, alerts).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config

# ---------------------------------------------------------------------------
# Palette (fixed by the design specification)
# ---------------------------------------------------------------------------
BG = "#080F1D"
BG_SIDEBAR = "#0B1424"
CARD = "#111D30"
CARD_ELEVATED = "#17253A"
ACCENT = "#38BDF8"
ACCENT_2 = "#6366F1"
TEXT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
BORDER = "#26364B"
GRID = "#1B2941"

SEVERITY_COLORS = {
    "Critical": "#EF4444",
    "High": "#F97316",
    "Medium": "#EAB308",
    "Low": "#22C55E",
    "Informational": "#64748B",
}
SEVERITY_GLYPH = {
    "Critical": "&#9679;",   # filled circle
    "High": "&#9650;",       # triangle
    "Medium": "&#9632;",     # square
    "Low": "&#9660;",        # inverted triangle
    "Informational": "&#9432;",
}
CATEGORY_COLORS = {
    "Normal": "#22C55E",
    "DoS": "#EF4444",
    "Probe": "#EAB308",
    "R2L": "#F97316",
    "U2R": "#A78BFA",
    "Unknown": "#64748B",
}
STATUS_COLORS = {
    "New": "#38BDF8",
    "Investigating": "#EAB308",
    "Resolved": "#22C55E",
    "False Positive": "#94A3B8",
}
STATE_COLORS = {"ok": "#22C55E", "warn": "#EAB308", "bad": "#EF4444",
                "idle": "#64748B"}

# ---------------------------------------------------------------------------
# Inline SVG icon set (Lucide-style stroke icons; no external requests)
# ---------------------------------------------------------------------------
_ICONS: dict[str, str] = {
    "shield": "M12 3 5 6v6c0 4.2 2.9 7.6 7 9 4.1-1.4 7-4.8 7-9V6z",
    "shield-check": "M12 3 5 6v6c0 4.2 2.9 7.6 7 9 4.1-1.4 7-4.8 7-9V6z M9 12l2 2 4-4",
    "gauge": "M12 14 16 9 M4.5 18a9 9 0 1 1 15 0",
    "activity": "M3 12h4l3-8 4 16 3-8h4",
    "network": "M12 3v5 M5 21v-4h14v4 M12 17V8 M5 17V8 M19 17V8 M3 8h18 M12 3a1 1 0 1 0 0 .01",
    "bell": "M6 9a6 6 0 1 1 12 0c0 4 1.5 5.5 2 6H4c.5-.5 2-2 2-6 M10 19a2 2 0 0 0 4 0",
    "crosshair": "M12 3v4 M12 17v4 M3 12h4 M17 12h4 M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8",
    "sparkles": "M12 4l1.6 4.4L18 10l-4.4 1.6L12 16l-1.6-4.4L6 10l4.4-1.6z M18 16l.8 2.2L21 19l-2.2.8L18 22l-.8-2.2L15 19l2.2-.8z",
    "chart": "M4 20V10 M10 20V4 M16 20v-7 M22 20H2",
    "cpu": "M6 6h12v12H6z M9 9h6v6H9z M9 2v4 M15 2v4 M9 18v4 M15 18v4 M2 9h4 M2 15h4 M18 9h4 M18 15h4",
    "file": "M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z M14 3v5h5 M9 13h6 M9 17h4",
    "settings": "M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6 M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-2.9 1.2 2 2 0 1 1-4 0 1.7 1.7 0 0 0-2.9-1.2l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1A1.7 1.7 0 0 0 3 15a2 2 0 1 1 0-4 1.7 1.7 0 0 0 1.4-2.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1A1.7 1.7 0 0 0 10 4.6 2 2 0 1 1 14 4.6a1.7 1.7 0 0 0 2.9 1.2l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1A1.7 1.7 0 0 0 21 11a2 2 0 1 1 0 4z",
    "database": "M12 3c4.4 0 8 1.3 8 3s-3.6 3-8 3-8-1.3-8-3 3.6-3 8-3 M4 6v6c0 1.7 3.6 3 8 3s8-1.3 8-3V6 M4 12v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6",
    "clock": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M12 7v5l3 2",
    "check": "M4 12.5 9.5 18 20 6.5",
    "check-circle": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M8.5 12.5l2.5 2.5 4.5-5",
    "alert": "M12 4 2.8 20h18.4z M12 10v4 M12 17h.01",
    "x-circle": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M9 9l6 6 M15 9l-6 6",
    "info": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M12 11v5 M12 8h.01",
    "search": "M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14 M20 20l-4-4",
    "refresh": "M20 11a8 8 0 1 0-2.3 6.3 M20 5v6h-6",
    "download": "M12 4v11 M8 11l4 4 4-4 M5 19h14",
    "upload": "M12 20V9 M8 13l4-4 4 4 M5 5h14",
    "eye": "M2.5 12S6 6 12 6s9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6 M12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5",
    "zap": "M13 3 5 14h6l-1 7 8-11h-6z",
    "layers": "M12 3 3 8l9 5 9-5z M3 13l9 5 9-5",
    "server": "M4 4h16v6H4z M4 14h16v6H4z M7.5 7h.01 M7.5 17h.01",
    "drive": "M4 5h16l2 9v5H2v-5z M4 14h16 M7 17h.01 M11 17h4",
    "trend": "M3 17l6-6 4 4 8-8 M15 7h6v6",
    "filter": "M3 5h18l-7 8v6l-4-2v-4z",
    "user": "M12 4a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M4.5 20a7.5 7.5 0 0 1 15 0",
    "play": "M7 4.5 19 12 7 19.5z",
    "list": "M8 6h13 M8 12h13 M8 18h13 M3.5 6h.01 M3.5 12h.01 M3.5 18h.01",
    "brain": "M9 4a3 3 0 0 0-3 3 3 3 0 0 0-1 5.8V15a3 3 0 0 0 4 2.8V20h3V4z M15 4a3 3 0 0 1 3 3 3 3 0 0 1 1 5.8V15a3 3 0 0 1-4 2.8V20h-3",
    "chevron": "M9 6l6 6-6 6",
    "note": "M5 4h14v11l-5 5H5z M19 15h-5v5 M8 8h8 M8 12h5",
    "lock": "M6 11h12v9H6z M9 11V8a3 3 0 0 1 6 0v3",
    "box": "M12 3 3 7.5v9L12 21l9-4.5v-9z M3 7.5 12 12l9-4.5 M12 12v9",
    "grid": "M3 3h8v8H3z M13 3h8v8h-8z M3 13h8v8H3z M13 13h8v8h-8z",
    "target": "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18 M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8 M12 11.5a.5.5 0 1 0 0 1 .5.5 0 0 0 0-1",
}


def icon(name: str, size: int = 16, color: str | None = None,
         stroke: float = 1.7) -> str:
    path = _ICONS.get(name, _ICONS["info"])
    col = color or "currentColor"
    return (
        f'<svg class="soc-icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{col}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
        f'<path d="{path}"/></svg>'
    )


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_CSS = f"""
<style>
:root {{
  --soc-bg: {BG}; --soc-sidebar: {BG_SIDEBAR}; --soc-card: {CARD};
  --soc-card-2: {CARD_ELEVATED}; --soc-accent: {ACCENT}; --soc-accent-2: {ACCENT_2};
  --soc-text: {TEXT}; --soc-muted: {TEXT_MUTED}; --soc-border: {BORDER};
}}

html, body, .stApp, div[data-testid="stAppViewContainer"] {{
  background-color: {BG};
  color: {TEXT};
}}

/* ------------------------------------------------------------- app shell */
header[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stMainBlockContainer"],
div[data-testid="stAppViewBlockContainer"] {{
  padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1680px;
}}
section[data-testid="stSidebar"] {{
  background-color: {BG_SIDEBAR};
  border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] > div:first-child {{ padding-top: 0.4rem; }}
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {{
  padding-top: 0.25rem;
}}

/* nav links */
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"],
section[data-testid="stSidebar"] a[href^="./"] {{
  border-radius: 8px; color: {TEXT_MUTED}; font-size: 0.9rem;
  transition: background-color .15s ease, color .15s ease;
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"]:hover {{
  background-color: {CARD}; color: {TEXT};
}}
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background-color: {CARD_ELEVATED}; color: {TEXT}; font-weight: 600;
  box-shadow: inset 2px 0 0 {ACCENT};
}}

/* ------------------------------------------------------------- typography */
h1, h2, h3, h4 {{ color: {TEXT}; letter-spacing: -0.01em; }}
p, li, label, span, div {{ font-family: "Inter", "Segoe UI", system-ui, sans-serif; }}
small, .stCaption, div[data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED}; }}
hr {{ border-color: {BORDER}; }}
a {{ color: {ACCENT}; }}

/* --------------------------------------------------------------- widgets */
.stButton > button, .stDownloadButton > button, button[kind="secondary"] {{
  background-color: {CARD}; border: 1px solid {BORDER}; color: {TEXT};
  border-radius: 8px; font-weight: 500; transition: all .15s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
  border-color: {ACCENT}; color: {ACCENT}; background-color: {CARD_ELEVATED};
}}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{
  background-color: {ACCENT}; border-color: {ACCENT}; color: #06121F; font-weight: 600;
}}
.stButton > button[kind="primary"]:hover {{ background-color: #5CC9FA; color: #06121F; }}

div[data-testid="stTextInput"] input, div[data-testid="stNumberInput"] input,
textarea, div[data-baseweb="select"] > div {{
  background-color: {BG_SIDEBAR} !important; border: 1px solid {BORDER} !important;
  color: {TEXT} !important; border-radius: 8px !important;
}}
div[data-baseweb="select"] > div:hover {{ border-color: {ACCENT} !important; }}
ul[data-baseweb="menu"], ul[data-baseweb="select"] {{
  background-color: {CARD} !important; border: 1px solid {BORDER} !important;
}}
li[data-baseweb="menu-item"] {{ color: {TEXT} !important; }}
li[data-baseweb="menu-item"]:hover {{ background-color: {CARD_ELEVATED} !important; }}

div[data-testid="stTabs"] button[data-baseweb="tab"] {{
  background: transparent; color: {TEXT_MUTED}; border-radius: 8px 8px 0 0;
}}
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {{
  color: {ACCENT}; background-color: {CARD};
}}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}

div[data-testid="stExpander"] {{
  border: 1px solid {BORDER}; border-radius: 10px; background-color: {CARD};
}}
div[data-testid="stExpander"] summary {{ color: {TEXT}; font-weight: 500; }}

input[type="checkbox"]:checked, .stToggle div[role="checkbox"][aria-checked="true"] {{
  background-color: {ACCENT} !important; border-color: {ACCENT} !important;
}}
div[data-baseweb="slider"] [role="slider"] {{ background-color: {ACCENT}; border-color: {ACCENT}; }}
div[data-baseweb="slider"] > div > div {{ background-color: {BORDER}; }}

/* alerts / callouts */
div[data-testid="stAlert"] {{ border-radius: 10px; border: 1px solid {BORDER}; }}
div[data-testid="stAlertContainer"] {{ background-color: {CARD}; border-radius: 10px; }}

/* dataframes & tables */
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
div[data-testid="stTable"] table {{ border-collapse: separate; border-spacing: 0; }}
div[data-testid="stTable"] th {{
  background-color: {CARD_ELEVATED} !important; color: {TEXT} !important;
  border-bottom: 1px solid {BORDER} !important; font-weight: 600; font-size: 0.8rem;
}}
div[data-testid="stTable"] td {{ border-bottom: 1px solid #1A2740 !important; font-size: 0.85rem; }}

/* ---------------------------------------------------------- card system */
[class*="st-key-soc_card"] {{
  background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
  padding: 0.1rem 1rem 0.9rem 1rem; margin-bottom: 0.2rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.25);
}}
[class*="st-key-soc_card"] > div {{ gap: 0.55rem; }}
[class*="st-key-soc_card_elev"], .st-key-soc_card_elev {{
  background-color: {CARD_ELEVATED} !important;
}}
.soc-card-title {{
  display: flex; align-items: center; gap: 8px; font-size: 0.95rem;
  font-weight: 600; color: {TEXT}; margin: 0.6rem 0 0.15rem 0;
}}
.soc-card-title .soc-icon {{ color: {ACCENT}; }}
.soc-card-sub {{ font-size: 0.78rem; color: {TEXT_MUTED}; margin: 0 0 .35rem 0; }}

/* --------------------------------------------------------- metric cards */
.soc-metric-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  align-items: stretch;
}}
.soc-metric {{
  background-color: {CARD}; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 14px 16px 13px; height: 100%; display: flex; flex-direction: column;
  gap: 9px; min-width: 0; transition: border-color .15s ease, transform .15s ease;
}}
.soc-metric:hover {{ border-color: #33465F; transform: translateY(-1px); }}
.soc-metric-head {{ display: flex; align-items: center; gap: 11px; min-width: 0; }}
.soc-metric-label {{
  font-size: 0.73rem; text-transform: uppercase; letter-spacing: .05em;
  color: {TEXT_MUTED}; font-weight: 600; line-height: 1.3;
  flex: 1 1 auto; min-width: 0; overflow-wrap: break-word;
}}
.soc-metric-icon {{
  display: flex; align-items: center; justify-content: center; width: 36px; height: 36px;
  border-radius: 11px; background-color: {CARD_ELEVATED};
  color: var(--soc-metric-accent, {ACCENT}); flex: 0 0 auto;
}}
.soc-metric-value {{
  font-size: 1.65rem; font-weight: 700; color: {TEXT}; line-height: 1.1;
  margin-top: 0; min-width: 0; overflow-wrap: break-word;
  font-variant-numeric: tabular-nums;
}}
.soc-metric-value small {{ font-size: 0.85rem; color: {TEXT_MUTED}; font-weight: 500; }}
.soc-metric-foot {{
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 10px; margin-top: auto;
}}
.soc-metric-caption {{ font-size: 0.74rem; color: {TEXT_MUTED}; line-height: 1.45; }}
.soc-metric-delta {{
  display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px;
  border-radius: 999px; border: 1px solid currentColor; font-size: 0.72rem;
  font-weight: 600; white-space: nowrap; flex: 0 0 auto;
}}

/* ------------------------------------------------------ progress bars */
.soc-progress {{ margin: 9px 0 3px 0; }}
.soc-progress-head {{
  display: flex; align-items: baseline; justify-content: space-between; gap: 8px;
  font-size: 0.8rem; color: {TEXT_MUTED}; margin-bottom: 6px;
}}
.soc-progress-head b {{
  color: {TEXT}; font-weight: 600; font-variant-numeric: tabular-nums; font-size: 0.85rem;
}}
.soc-progress-track {{
  height: 7px; border-radius: 999px; background-color: {CARD_ELEVATED};
  border: 1px solid #1E2C44; overflow: hidden;
}}
.soc-progress-fill {{ height: 100%; border-radius: 999px; }}
.soc-progress-sub {{ font-size: 0.72rem; color: {TEXT_MUTED}; margin-top: 5px; }}

/* ----------------------------------------------------- rich HTML table */
.soc-table-wrap {{ overflow-x: auto; }}
.soc-table {{ width: 100%; min-width: 620px; border-collapse: collapse; font-size: 0.84rem; }}
.soc-table th {{
  text-align: left; font-size: 0.68rem; text-transform: uppercase; letter-spacing: .05em;
  color: {TEXT_MUTED}; font-weight: 600; padding: 8px 10px;
  border-bottom: 1px solid {BORDER};
}}
.soc-table td {{
  padding: 9px 10px; border-bottom: 1px solid #1A2740; color: {TEXT}; vertical-align: middle;
}}
.soc-table tbody tr:hover td {{ background-color: {BG_SIDEBAR}; }}
.soc-table td.num {{ font-variant-numeric: tabular-nums; color: {TEXT_MUTED}; }}
.soc-risk {{ display: flex; align-items: center; gap: 8px; }}
.soc-risk b {{ font-variant-numeric: tabular-nums; color: {TEXT}; font-weight: 600; }}
.soc-riskbar {{
  height: 5px; border-radius: 999px; background-color: {CARD_ELEVATED};
  overflow: hidden; min-width: 58px; flex: 1 1 auto;
}}
.soc-riskbar span {{ display: block; height: 100%; border-radius: 999px; }}

/* -------------------------------------------------------- page header */
.soc-head-row {{ display: flex; align-items: flex-start; gap: 13px; }}
.soc-head-tile {{
  width: 44px; height: 44px; border-radius: 13px; display: flex; align-items: center;
  justify-content: center; flex: 0 0 auto; color: {ACCENT};
  background: linear-gradient(150deg, {ACCENT}26 0%, {ACCENT_2}26 100%);
  border: 1px solid {BORDER};
}}

/* ------------------------------------------------------ badges & pills */
.soc-badge {{
  display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px;
  border-radius: 999px; font-size: 0.74rem; font-weight: 600;
  border: 1px solid currentColor; line-height: 1.5;
  max-width: 100%; white-space: normal; text-align: left;
}}
.soc-chip {{
  display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
  border-radius: 8px; background-color: {CARD}; border: 1px solid {BORDER};
  font-size: 0.78rem; color: {TEXT_MUTED}; white-space: nowrap;
}}
.soc-chip b {{ color: {TEXT}; font-weight: 600; }}
.soc-chip .soc-icon {{ color: {ACCENT}; }}
.soc-dot {{ width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: 0 0 auto; }}

/* ------------------------------------------------------------- top bar */
.soc-rule {{ border-bottom: 1px solid {BORDER}; margin: 10px 0 16px 0; }}
.soc-crumb {{ font-size: 0.76rem; color: {TEXT_MUTED}; display: flex; align-items: center; gap: 6px; }}
.soc-crumb span.sep {{ opacity: .55; display: inline-flex; }}
.soc-crumb span.current {{ color: {ACCENT}; font-weight: 600; }}
.soc-title {{ font-size: 1.5rem; font-weight: 700; color: {TEXT}; margin: 4px 0 0 0; }}
.soc-subtitle {{ font-size: 0.86rem; color: {TEXT_MUTED}; margin-top: 4px; max-width: 900px; line-height: 1.5; }}
.soc-chip-row {{
  display: flex; align-items: center; flex-wrap: wrap; gap: 8px;
  margin: 2px 0 0 0;
}}

/* ------------------------------------------------------------ misc bits */
.soc-kv {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
.soc-kv th {{
  text-align: left; color: {TEXT_MUTED}; font-weight: 500; padding: 7px 10px 7px 0;
  width: 42%; vertical-align: top; border-bottom: 1px solid #1A2740;
}}
.soc-kv td {{ color: {TEXT}; padding: 7px 0; border-bottom: 1px solid #1A2740; vertical-align: top; }}
.soc-note {{
  display: flex; gap: 9px; align-items: flex-start; background-color: {BG_SIDEBAR};
  border: 1px solid {BORDER}; border-left: 3px solid {ACCENT_2}; border-radius: 8px;
  padding: 10px 12px; font-size: 0.82rem; color: {TEXT_MUTED}; margin: 6px 0;
}}
.soc-note b {{ color: {TEXT}; }}
.soc-note .soc-icon {{ margin-top: 1px; flex: 0 0 auto; }}
.soc-empty {{ text-align: center; padding: 34px 18px; color: {TEXT_MUTED}; }}
.soc-empty .soc-icon {{ color: #3A4C66; margin-bottom: 10px; }}
.soc-empty h4 {{ color: {TEXT}; margin: 0 0 6px 0; font-size: 1rem; }}
.soc-empty p {{ margin: 0; font-size: 0.85rem; }}

.soc-stage {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 2px 0; }}
.soc-stage-item {{
  display: flex; align-items: center; gap: 7px; padding: 6px 11px; border-radius: 999px;
  font-size: 0.76rem; font-weight: 500; border: 1px solid {BORDER};
  background-color: {BG_SIDEBAR}; color: {TEXT_MUTED};
}}
.soc-stage-item.done {{ border-color: #1E5B3A; color: #6EE7A0; background-color: #0E2018; }}
.soc-stage-item.active {{ border-color: {ACCENT}; color: {ACCENT}; background-color: #0D2135; }}
.soc-stage-item .num {{
  width: 17px; height: 17px; border-radius: 50%; display: inline-flex; align-items: center;
  justify-content: center; font-size: 0.66rem; border: 1px solid currentColor;
}}

.soc-brand {{ display: flex; align-items: center; gap: 11px; padding: 6px 4px 14px 4px; }}
.soc-brand-mark {{
  width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center;
  justify-content: center; background: linear-gradient(150deg, {ACCENT} 0%, {ACCENT_2} 100%);
  color: #06121F; flex: 0 0 auto;
}}
.soc-brand-name {{ font-size: 1.05rem; font-weight: 700; color: {TEXT}; letter-spacing: .01em; }}
.soc-brand-sub {{ font-size: 0.7rem; color: {TEXT_MUTED}; letter-spacing: .04em; text-transform: uppercase; }}

.soc-foot {{ border-top: 1px solid {BORDER}; margin-top: 12px; padding-top: 10px; }}
.soc-foot-row {{
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  font-size: 0.74rem; color: {TEXT_MUTED}; padding: 3px 0;
}}
.soc-foot-row b {{ color: {TEXT}; font-weight: 600; }}
.soc-ver {{ font-size: 0.68rem; color: #5A6B82; padding-top: 8px; }}

.soc-plain {{ font-size: 0.88rem; line-height: 1.6; color: #D7E0EC; }}
.soc-plain b {{ color: {TEXT}; }}
code {{ background-color: {CARD_ELEVATED}; color: {ACCENT}; padding: 1px 5px; border-radius: 5px; }}

/* responsive */
@media (max-width: 900px) {{
  .soc-metric-value {{ font-size: 1.35rem; }}
  .soc-title {{ font-size: 1.25rem; }}
  div[data-testid="stMainBlockContainer"], div[data-testid="stAppViewBlockContainer"] {{
    padding-left: 0.9rem; padding-right: 0.9rem;
  }}
}}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shell components
# ---------------------------------------------------------------------------
def sidebar_brand() -> None:
    st.sidebar.markdown(
        f"""
        <div class="soc-brand">
          <div class="soc-brand-mark">{icon("shield-check", 21, "#06121F", 2.0)}</div>
          <div>
            <div class="soc-brand-name">AI-SOC</div>
            <div class="soc-brand-sub">Intelligent Security Operations</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer(system_state: str, model_state: str, dataset_state: str,
                   version: str = config.APP_VERSION) -> None:
    def row(label: str, value: str) -> str:
        state = "ok"
        low = value.lower()
        if any(w in low for w in ("missing", "error", "fail", "unavailable", "not ")):
            state = "bad"
        elif any(w in low for w in ("warn", "partial", "stale", "idle", "not run")):
            state = "warn"
        colour = STATE_COLORS[state]
        return (
            f'<div class="soc-foot-row"><span>{label}</span>'
            f'<span style="display:flex;align-items:center;gap:6px;">'
            f'<span class="soc-dot" style="background:{colour}"></span>'
            f'<b>{value}</b></span></div>'
        )

    st.sidebar.markdown(
        f"""
        <div class="soc-foot">
          {row("System", system_state)}
          {row("Models", model_state)}
          {row("Dataset", dataset_state)}
          <div class="soc-ver">AI-SOC research prototype &middot; v{version}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def topbar(breadcrumb: Sequence[str], title: str, subtitle: str,
           chips: Sequence[str] = (), notifications: int | None = None,
           refresh_key: str = "topbar_refresh",
           icon_name: str | None = None) -> None:
    """Page header + information/action bar.

    `notifications` must be a real count (e.g. untriaged critical alerts) or
    None - never a simulated value.
    """
    crumbs = "".join(
        f'<span>{c}</span><span class="sep">{icon("chevron", 11)}</span>'
        for c in breadcrumb[:-1]
    ) + f'<span class="current">{breadcrumb[-1]}</span>'
    notif = ""
    if notifications:
        notif = (
            f'<span class="soc-chip" title="{notifications:,} alerts are still '
            f'New and rated Critical or High">'
            f'{icon("bell", 13, "#F97316")}<b>{notifications:,}</b></span>'
        )
    tile = (
        f'<div class="soc-head-tile">{icon(icon_name, 22)}</div>'
        if icon_name else ""
    )
    left, right = st.columns([5.2, 1.3], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div class="soc-crumb">{icon("shield", 12)}{crumbs}</div>'
            f'<div class="soc-head-row">{tile}'
            f'<div><div class="soc-title">{title}</div>'
            f'<div class="soc-subtitle">{subtitle}</div></div></div>',
            unsafe_allow_html=True,
        )
    with right:
        st.button(
            "Refresh data", key=refresh_key, icon=":material/refresh:",
            help="Re-read alerts, metrics and settings from disk.",
            on_click=_refresh_caches, width="stretch",
        )
    strip = "".join(chips) + notif
    if strip:
        st.markdown(f'<div class="soc-chip-row">{strip}</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="soc-rule"></div>', unsafe_allow_html=True)


def _refresh_caches() -> None:
    st.cache_data.clear()
    from src import settings
    settings.log("Dashboard caches refreshed by analyst.")


def chip(label: str, value: str, icon_name: str = "info",
         value_color: str | None = None) -> str:
    colour = value_color or TEXT
    return (
        f'<span class="soc-chip">{icon(icon_name, 13)}'
        f'{label} <b style="color:{colour}">{value}</b></span>'
    )


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
@contextmanager
def card(title: str | None = None, icon_name: str | None = None,
         subtitle: str | None = None, key: str = "card",
         elevated: bool = False):
    """Bordered surface with an optional header. Usage: with card(...): ..."""
    suffix = "_elev" if elevated else ""
    container = st.container(key=f"soc_card_{key}{suffix}")
    with container:
        if title:
            head = icon(icon_name, 15) if icon_name else ""
            st.markdown(
                f'<div class="soc-card-title">{head}<span>{title}</span></div>',
                unsafe_allow_html=True,
            )
        if subtitle:
            st.markdown(
                f'<div class="soc-card-sub">{subtitle}</div>',
                unsafe_allow_html=True,
            )
        yield container


def metric_card(label: str, value: str, caption: str = "",
                icon_name: str = "activity", accent: str = ACCENT,
                delta: str | None = None, delta_color: str = TEXT_MUTED) -> str:
    """KPI card: tinted icon tile + label, large value, caption and delta chip.

    `delta` must describe a real, measured change - never an invented trend.
    """
    delta_html = (
        f'<span class="soc-metric-delta" style="color:{delta_color};'
        f'background:{delta_color}14">{delta}</span>'
        if delta else ""
    )
    return (
        f'<div class="soc-metric" style="--soc-metric-accent:{accent}">'
        f'<div class="soc-metric-head">'
        f'<span class="soc-metric-icon" style="background:{accent}1F;'
        f'color:{accent}">{icon(icon_name, 17)}</span>'
        f'<div class="soc-metric-label">{label}</div></div>'
        f'<div class="soc-metric-value">{value}</div>'
        f'<div class="soc-metric-foot">'
        f'<div class="soc-metric-caption">{caption}</div>{delta_html}</div></div>'
    )


def metric_row(cards: Sequence[str], gap: str = "small") -> None:
    px = {"xsmall": 8, "small": 12, "medium": 16, "large": 24}.get(gap, 12)
    st.markdown(
        f'<div class="soc-metric-grid" style="gap:{px}px">'
        f'{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def badge(text: str, color: str, glyph: str | None = None) -> str:
    inner = f'<span>{glyph}</span>' if glyph else ""
    return (
        f'<span class="soc-badge" style="color:{color};background:{color}1A">'
        f'{inner}{text}</span>'
    )


def severity_badge(severity: str) -> str:
    colour = SEVERITY_COLORS.get(severity, TEXT_MUTED)
    return badge(severity, colour, SEVERITY_GLYPH.get(severity))


def status_pill(status: str) -> str:
    return badge(status, STATUS_COLORS.get(status, TEXT_MUTED))


def category_badge(category: str) -> str:
    return badge(category, CATEGORY_COLORS.get(category, TEXT_MUTED))


def note(text: str, icon_name: str = "info", accent: str | None = None) -> None:
    colour = accent or ACCENT_2
    st.markdown(
        f'<div class="soc-note" style="border-left-color:{colour}">'
        f'<span style="color:{colour}">{icon(icon_name, 15)}</span>'
        f'<div>{text}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str, icon_name: str = "box") -> None:
    st.markdown(
        f'<div class="soc-empty">{icon(icon_name, 34)}'
        f'<h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def kv_table(rows: Iterable[tuple[str, str]]) -> None:
    body = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows)
    st.markdown(f'<table class="soc-kv">{body}</table>', unsafe_allow_html=True)


def stage_tracker(stages: Sequence[tuple[str, float | None]],
                  active_index: int | None = None) -> None:
    """Pipeline progress: (label, seconds|None). Done stages show timing."""
    items = []
    for i, (label, secs) in enumerate(stages):
        if secs is not None:
            cls, extra = "done", f"{secs:.2f}s"
        elif active_index == i:
            cls, extra = "active", "running"
        else:
            cls, extra = "", "pending"
        items.append(
            f'<div class="soc-stage-item {cls}">'
            f'<span class="num">{i + 1}</span>{label}'
            f'<span style="opacity:.7">{extra}</span></div>'
        )
    st.markdown(f'<div class="soc-stage">{"".join(items)}</div>',
                unsafe_allow_html=True)


def plain_language(text: str) -> None:
    st.markdown(f'<div class="soc-plain">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def style_fig(fig: go.Figure, height: int | None = None,
              title: str | None = None) -> go.Figure:
    fig.update_layout(
        template=None,
        height=height,
        title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', sans-serif", size=12.5,
                  color=TEXT_MUTED),
        margin=dict(l=16, r=44, t=46 if title else 28, b=16),
        xaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER,
                   automargin=True,
                   tickfont=dict(size=11.5, color=TEXT_MUTED)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=BORDER, linecolor=BORDER,
                   automargin=True,
                   tickfont=dict(size=11.5, color=TEXT_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11.5, color=TEXT_MUTED),
                    orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0),
        hoverlabel=dict(bgcolor=CARD_ELEVATED, bordercolor=BORDER,
                        font=dict(size=12, color=TEXT)),
        colorway=[ACCENT, ACCENT_2, "#22C55E", "#EAB308", "#F97316", "#EF4444",
                  "#A78BFA", "#2DD4BF"],
    )
    return fig


def show(fig: go.Figure, height: int | None = None, key: str | None = None,
         caption: str | None = None) -> None:
    style_fig(fig, height=height)
    st.plotly_chart(fig, theme=None, width="stretch", key=key,
                    config={"displaylogo": False})
    if caption:
        st.caption(caption)


def doughnut(labels: Sequence[str], values: Sequence[float],
             colors: Sequence[str] | None = None, hole: float = 0.62,
             height: int = 320, center_label: str | None = None,
             center_sub: str | None = None,
             legend_percent: bool = True) -> go.Figure:
    """Donut chart with an optional centre figure and percentages in the legend.

    The percentages are computed from `values` (the caller's real counts) and
    shown in the legend instead of on the slices, matching the design spec.
    """
    raw = [str(x) for x in labels]
    vals = [float(v) for v in values]
    total = sum(vals)
    if legend_percent and total > 0:
        shown = [f"{lbl}   {v / total:.1%}" for lbl, v in zip(raw, vals)]
    else:
        shown = raw
    fig = go.Figure(go.Pie(
        labels=shown, values=vals, hole=hole, sort=False, textinfo="none",
        marker=dict(colors=list(colors) if colors else None,
                    line=dict(color=BORDER, width=1)),
        customdata=raw,
        hovertemplate="%{customdata}<br>%{value:,.0f} records "
                      "(%{percent})<extra></extra>",
    ))
    fig.update_layout(showlegend=True)
    style_fig(fig, height=height)
    if center_label:
        # Centre the ring and put the legend in a row beneath it so neither
        # overlaps nor clips inside narrow cards.
        fig.update_traces(domain=dict(x=[0.16, 0.84], y=[0.22, 1.0]))
        fig.update_layout(
            margin=dict(l=8, r=8, t=8, b=52),
            legend=dict(orientation="h", x=0.5, xanchor="center",
                        y=-0.02, yanchor="top"),
        )
        text = f'<b style="font-size:19px;color:{TEXT}">{center_label}</b>'
        if center_sub:
            text += (f'<br><span style="font-size:11px;color:{TEXT_MUTED}">'
                     f'{center_sub}</span>')
        fig.add_annotation(text=text, xref="paper", yref="paper", x=0.5,
                           y=0.61, showarrow=False, align="center",
                           font=dict(size=13, color=TEXT))
    else:
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5,
                                      yanchor="middle"))
    return fig


def progress(label: str, value: float, color: str = ACCENT,
             max_value: float = 100.0, sub: str | None = None,
             digits: int = 1, suffix: str = "") -> None:
    """Horizontal bar for a measured quantity (CPU %, memory %, risk 0-100)."""
    try:
        ratio = float(value) / float(max_value) if max_value else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = 0.0
    width = max(0.0, min(1.0, ratio)) * 100
    sub_html = f'<div class="soc-progress-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="soc-progress">'
        f'<div class="soc-progress-head"><span>{label}</span>'
        f'<b>{float(value):,.{digits}f}{suffix}</b></div>'
        f'<div class="soc-progress-track">'
        f'<div class="soc-progress-fill" style="width:{width:.1f}%;'
        f'background:{color}"></div></div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def ring_gauge(value: float, label: str, suffix: str = "%",
               color: str = ACCENT, value_range: tuple[float, float] = (0, 100),
               height: int = 200, digits: int = 1) -> go.Figure:
    """Ring gauge for one measured value.

    `value_range` must be the quantity's real scale (a percentage, or a
    measured maximum stated by the caller) - the filled arc is value/range.
    """
    lo, hi = float(value_range[0]), float(value_range[1])
    span = (hi - lo) or 1.0
    fraction = max(0.0, min(1.0, (float(value) - lo) / span))
    fig = go.Figure(go.Pie(
        values=[fraction, 1.0 - fraction], labels=["measured", "remaining"],
        hole=0.76, sort=False, direction="clockwise", rotation=-90,
        marker=dict(colors=[color, CARD_ELEVATED],
                    line=dict(color=BORDER, width=1)),
        textinfo="none", hoverinfo="skip", showlegend=False,
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        template=None, height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', sans-serif", size=12,
                  color=TEXT_MUTED),
    )
    fig.add_annotation(
        text=f'<b style="font-size:20px;color:{TEXT}">'
             f'{float(value):,.{digits}f}{suffix}</b>'
             f'<br><span style="font-size:11px;color:{TEXT_MUTED}">{label}</span>',
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        align="center", font=dict(size=12, color=TEXT_MUTED),
    )
    return fig


def alert_table(alerts: pd.DataFrame, max_rows: int | None = None) -> None:
    """Rich alert list: severity dots, category and status pills, risk bar.

    Renders only the columns present in `alerts`; risk bars are scaled to the
    documented 0-100 risk score, never to an arbitrary maximum.
    """
    rows = alerts.head(max_rows) if max_rows else alerts
    if rows.empty:
        return
    has_conf = "model_confidence" in rows.columns
    body = []
    for r in rows.itertuples():
        sev = str(getattr(r, "severity", ""))
        sev_colour = SEVERITY_COLORS.get(sev, TEXT_MUTED)
        cat = str(getattr(r, "predicted_category", ""))
        status = str(getattr(r, "investigation_status", ""))
        risk = float(getattr(r, "risk_score", 0.0) or 0.0)
        width = max(0.0, min(100.0, risk))
        conf_html = ""
        if has_conf:
            conf = getattr(r, "model_confidence", None)
            conf_html = ("<td class='num'>n/a</td>"
                         if conf is None or pd.isna(conf)
                         else f"<td class='num'>{float(conf):.1%}</td>")
        body.append(
            "<tr>"
            f'<td><b>{r.alert_id}</b><div style="font-size:.72rem;'
            f'color:{TEXT_MUTED}">record {int(r.record_index):,}</div></td>'
            f'<td>{category_badge(cat)}</td>'
            f'<td><span style="display:inline-flex;align-items:center;gap:7px;'
            f'color:{sev_colour};font-weight:600">'
            f'<span class="soc-dot" style="background:{sev_colour}"></span>'
            f'{sev}</span></td>'
            f'<td><div class="soc-risk"><b>{risk:.1f}</b>'
            f'<div class="soc-riskbar"><span style="width:{width:.1f}%;'
            f'background:{sev_colour}"></span></div></div></td>'
            f"{conf_html}"
            f'<td>{status_pill(status)}</td>'
            "</tr>"
        )
    conf_head = "<th>Confidence</th>" if has_conf else ""
    st.markdown(
        f'<div class="soc-table-wrap"><table class="soc-table"><thead><tr>'
        f'<th>Alert</th><th>Category</th><th>Severity</th>'
        f'<th>Risk score</th>{conf_head}<th>Status</th>'
        f'</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def hbar(labels: Sequence[str], values: Sequence[float],
         colors: Sequence[str] | None = None, height: int = 300,
         title: str | None = None, text_auto: bool = True,
         text: Sequence[str] | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(values), y=list(labels), orientation="h",
        marker=dict(color=list(colors) if colors else ACCENT,
                    line=dict(color=BORDER, width=0.5)),
        text=list(text) if text else (
            [f"{v:,.0f}" for v in values] if text_auto else None),
        textposition="outside", textfont=dict(size=11.5, color=TEXT_MUTED),
        hovertemplate="%{y}: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(title=None), yaxis=dict(automargin=True))
    return style_fig(fig, height=height, title=title)


def vbar(labels: Sequence[str], values: Sequence[float],
         colors: Sequence[str] | None = None, height: int = 300,
         title: str | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker=dict(color=list(colors) if colors else ACCENT,
                    line=dict(color=BORDER, width=0.5)),
        text=[f"{v:,.2f}" if isinstance(v, float) and v < 10 else f"{v:,.0f}"
              for v in values],
        textposition="outside", textfont=dict(size=11.5, color=TEXT_MUTED),
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    return style_fig(fig, height=height, title=title)


def grouped_bar(labels: Sequence[str], series: dict[str, Sequence[float]],
                height: int = 340, title: str | None = None,
                y_range: tuple[float, float] | None = None,
                colors: Sequence[str] | None = None) -> go.Figure:
    fig = go.Figure()
    palette = colors or [ACCENT, ACCENT_2, "#22C55E", "#EAB308"]
    for (name, values), colour in zip(series.items(), palette):
        fig.add_trace(go.Bar(
            name=name, x=list(labels), y=list(values),
            marker=dict(color=colour, line=dict(color=BORDER, width=0.5)),
            hovertemplate=f"{name}<br>%{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    if y_range:
        fig.update_yaxes(range=list(y_range))
    return style_fig(fig, height=height, title=title)


def line_chart(x: Sequence, series: dict[str, Sequence[float]],
               height: int = 300, title: str | None = None,
               y_title: str | None = None, fill: bool = False) -> go.Figure:
    fig = go.Figure()
    palette = [ACCENT, ACCENT_2, "#22C55E", "#EAB308", "#EF4444"]
    for (name, values), colour in zip(series.items(), palette):
        fig.add_trace(go.Scatter(
            x=list(x), y=list(values), name=name, mode="lines+markers",
            line=dict(color=colour, width=2),
            marker=dict(size=5),
            fill="tozeroy" if fill else None,
            fillcolor=f"{colour}22" if fill else None,
            hovertemplate=f"{name}<br>%{{x}}: %{{y}}<extra></extra>",
        ))
    fig.update_layout(yaxis_title=y_title)
    return style_fig(fig, height=height, title=title)


def heatmap(matrix, labels_x: Sequence[str], labels_y: Sequence[str],
            height: int = 420, title: str | None = None,
            colorscale: str = "Blues", text: bool = True) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=matrix, x=list(labels_x), y=list(labels_y),
        colorscale=colorscale, showscale=True,
        colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=10, color=TEXT_MUTED)),
        text=[[f"{v:,}" for v in row] for row in matrix] if text else None,
        texttemplate="%{text}" if text else None,
        textfont=dict(size=11, color="#E8EEF7"),
        hovertemplate="true: %{y}<br>pred: %{x}<br>count: %{z}<extra></extra>",
    ))
    fig.update_layout(xaxis=dict(side="bottom", title="Predicted"),
                      yaxis=dict(title="Actual", autorange="reversed"))
    return style_fig(fig, height=height, title=title)


def shap_bars(explanation, top_k: int, height: int | None = None) -> go.Figure:
    """Signed SHAP contributions of one record: green pushes toward the
    predicted class, red pushes against it."""
    contrib = sorted(explanation.contributions, key=lambda c: abs(c[2]),
                     reverse=True)[:top_k]
    contrib = contrib[::-1]
    labels = [f"{f} = {v:g}" for f, v, _ in contrib]
    values = [float(c[2]) for c in contrib]
    colors = ["#EF4444" if v < 0 else "#22C55E" for v in values]
    fig = hbar(labels, values, colors=colors,
               height=height or int(34 * len(values) + 110),
               text=[f"{v:+.3f}" for v in values], text_auto=False)
    fig.update_traces(hovertemplate="%{y}<br>SHAP %{x:+.4f}<extra></extra>")
    return fig


def shap_waterfall(explanation, top_k: int, height: int = 430) -> go.Figure:
    """Base value -> top-k contributions -> reconstructed model output."""
    contrib = sorted(explanation.contributions, key=lambda c: abs(c[2]),
                     reverse=True)[:top_k]
    labels = [f"{f} ({v:g})" for f, v, _ in contrib]
    values = [float(c[2]) for c in contrib]
    total = float(sum(c[2] for c in explanation.contributions))
    fig = go.Figure(go.Waterfall(
        orientation="v",
        x=["Base value"] + labels + ["Reconstructed output"],
        measure=["absolute"] + ["relative"] * len(labels) + ["total"],
        y=[explanation.base_value] + values + [0],
        connector=dict(line=dict(color=BORDER, width=1)),
        decreasing=dict(marker=dict(color="#EF4444")),
        increasing=dict(marker=dict(color="#22C55E")),
        totals=dict(marker=dict(color=ACCENT)),
        hovertemplate="%{x}<br>%{y:+.4f}<extra></extra>",
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=10, color=TEXT_MUTED),
    ))
    fig = style_fig(fig, height=height)
    fig.update_layout(xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
                      margin=dict(b=110))
    fig.add_annotation(
        text=f"All {len(explanation.contributions)} contributions sum to "
             f"{total:+.4f}; base + sum = "
             f"{explanation.base_value + total:.4f} (explainer units).",
        xref="paper", yref="paper", x=0, y=-0.28, showarrow=False,
        font=dict(size=11, color=TEXT_MUTED), align="left",
    )
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def severity_styler(df: pd.DataFrame):
    """Colour-code severity / status / category columns for st.dataframe."""
    styler = df.style
    if "severity" in df.columns:
        styler = styler.map(
            lambda v: f"color:{SEVERITY_COLORS.get(str(v), TEXT)};font-weight:600",
            subset=["severity"],
        )
    for col, palette in (("investigation_status", STATUS_COLORS),
                         ("predicted_category", CATEGORY_COLORS),
                         ("true_category", CATEGORY_COLORS),
                         ("detection_status", {"Anomaly": "#F97316", "Normal": "#22C55E"})):
        if col in df.columns:
            styler = styler.map(
                lambda v, p=palette: f"color:{p.get(str(v), TEXT)};font-weight:600",
                subset=[col],
            )
    return styler


def show_table(df: pd.DataFrame, height: int = 380, key: str | None = None,
               hide_index: bool = True) -> None:
    st.dataframe(severity_styler(df), width="stretch", height=height,
                 hide_index=hide_index, key=key)


def paginate(df: pd.DataFrame, page_size: int, key: str) -> pd.DataFrame:
    """Client-side pagination control; returns the visible slice."""
    total = len(df)
    if total == 0:
        return df
    pages = max(1, -(-total // page_size))
    c1, c2, c3 = st.columns([1, 3, 3])
    with c1:
        page = st.number_input("Page", min_value=1, max_value=pages, value=1,
                               step=1, key=f"{key}_page", label_visibility="collapsed")
    start = (int(page) - 1) * page_size
    stop = min(start + page_size, total)
    with c2:
        st.caption(f"Showing records {start + 1:,}-{stop:,} of {total:,} "
                   f"(page {int(page)} of {pages})")
    return df.iloc[start:stop]


def timestamp_label(iso: str | None, fallback: str = "not recorded") -> str:
    if not iso:
        return fallback
    try:
        dt = datetime.fromisoformat(str(iso))
    except ValueError:
        return str(iso)
    if dt.tzinfo is None:
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fmt(value, digits: int = 2, suffix: str = "", missing: str = "not measured") -> str:
    if value is None:
        return missing
    try:
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def pct(value, digits: int = 1, missing: str = "not measured") -> str:
    if value is None:
        return missing
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)
