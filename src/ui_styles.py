"""Design tokens CSS adaptif (Light & Dark mode), tipografi, dan aksesibilitas."""

import streamlit as st


def apply_custom_styles() -> None:
    """Terapkan design tokens adaptif (Light & Dark mode), aksesibilitas, dan tipografi."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

        :root {
            --accent-gold: #F59E0B;
            --accent-gold-hover: #D97706;
            --accent-ink: #0F172A;
            --pos-bg: rgba(13, 148, 136, 0.14);
            --pos-border: rgba(13, 148, 136, 0.45);
            --neg-bg: rgba(220, 38, 38, 0.14);
            --neg-border: rgba(220, 38, 38, 0.45);
            --info-bg: rgba(245, 158, 11, 0.14);
            --info-border: rgba(245, 158, 11, 0.50);
        }

        .stApp {
            --surface-bg: color-mix(in srgb, currentColor 7%, transparent);
            --ink-primary: inherit;
            --ink-secondary: color-mix(in srgb, currentColor 88%, transparent);
            --ink-muted: color-mix(in srgb, currentColor 76%, transparent);
            --border-subtle: color-mix(in srgb, currentColor 18%, transparent);
        }

        html, body, .stApp, [class*="css"] {
            font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.75rem;
            padding-bottom: 2.75rem;
        }

        .hero-header {
            background-color: var(--surface-bg);
            border: 1px solid var(--border-subtle);
            border-top: 4px solid var(--accent-gold);
            border-radius: 10px;
            padding: 1.35rem 1.65rem;
            margin-bottom: 1.15rem;
        }

        .hero-title {
            margin: 0 0 0.3rem 0;
            font-size: 1.85rem;
            font-weight: 700;
            line-height: 1.25;
            color: var(--ink-primary);
            letter-spacing: -0.015em;
        }

        .hero-subtitle {
            margin: 0;
            font-size: 0.98rem;
            line-height: 1.55;
            color: var(--ink-muted);
        }

        .section-title {
            margin: 0.25rem 0 0.75rem 0;
            font-size: 1.25rem;
            font-weight: 600;
            line-height: 1.35;
            color: var(--ink-primary);
        }

        .subsection-title {
            margin: 1rem 0 0.5rem 0;
            font-size: 1.05rem;
            font-weight: 600;
            line-height: 1.35;
            color: var(--ink-primary);
        }

        .chart-title {
            margin: 0.35rem 0 0.25rem 0;
            font-size: 1.02rem;
            font-weight: 600;
            line-height: 1.35;
            color: var(--ink-primary);
        }

        .empty-guide {
            background-color: var(--surface-bg);
            border: 1px dashed var(--border-subtle);
            border-radius: 8px;
            padding: 0.95rem 1.15rem;
            color: var(--ink-secondary);
            font-size: 0.94rem;
            line-height: 1.55;
            margin: 0.65rem 0 0.85rem 0;
        }

        .sentiment-card {
            border-radius: 8px;
            padding: 0.95rem 1.15rem;
            border: 1px solid;
            color: var(--ink-primary);
        }

        .sentiment-card--pos {
            background-color: var(--pos-bg);
            border-color: var(--pos-border);
        }

        .sentiment-card--neg {
            background-color: var(--neg-bg);
            border-color: var(--neg-border);
        }

        .sentiment-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--ink-muted);
            margin-bottom: 0.2rem;
        }

        .sentiment-value {
            font-size: 1.35rem;
            font-weight: 700;
            line-height: 1.2;
            color: var(--ink-primary);
        }

        .topic-card {
            background-color: var(--info-bg);
            border: 1px solid var(--info-border);
            border-radius: 8px;
            padding: 0.95rem 1.15rem;
            color: var(--ink-primary);
            margin-bottom: 0.65rem;
            overflow-wrap: anywhere;
        }

        .topic-card-id {
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            text-transform: uppercase;
            color: var(--ink-muted);
            margin-bottom: 0.2rem;
        }

        .topic-card-name {
            font-size: 1.08rem;
            font-weight: 700;
            color: var(--ink-primary);
            margin: 0;
        }

        .keyword-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            margin: 0.45rem 0 0.75rem 0;
        }

        .keyword-chip {
            display: inline-block;
            background-color: var(--surface-bg);
            border: 1px solid var(--border-subtle);
            border-radius: 6px;
            padding: 0.2rem 0.55rem;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--ink-secondary);
            overflow-wrap: anywhere;
        }

        .prob-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--ink-secondary);
            margin-top: 0.35rem;
            font-variant-numeric: tabular-nums;
        }

        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        [data-testid="stDataFrame"] {
            font-variant-numeric: tabular-nums;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height: 44px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 180ms ease, border-color 180ms ease;
        }

        .stButton > button[kind="primary"] {
            background-color: var(--accent-gold);
            color: var(--accent-ink);
            border: 1px solid var(--accent-gold-hover);
        }

        .stButton > button[kind="primary"]:hover {
            background-color: var(--accent-gold-hover);
            color: var(--accent-ink);
            border-color: #B45309;
        }

        .stButton > button:focus-visible,
        .stDownloadButton > button:focus-visible,
        [role="tab"]:focus-visible,
        textarea:focus-visible,
        input:focus-visible,
        select:focus-visible {
            outline: 2px solid currentColor !important;
            outline-offset: 2px !important;
        }

        [role="tab"] {
            cursor: pointer;
            font-weight: 600;
            min-height: 44px;
        }

        .app-footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-subtle);
            text-align: center;
            color: var(--ink-muted);
            font-size: 0.85rem;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-top: 1rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .hero-header {
                padding: 1rem 1.1rem;
            }
            .hero-title {
                font-size: 1.4rem;
            }
            textarea, input, select {
                font-size: 16px !important;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                transition-duration: 0.01ms !important;
                animation-duration: 0.01ms !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
