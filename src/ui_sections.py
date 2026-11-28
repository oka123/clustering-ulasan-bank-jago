"""Facade modul antarmuka Streamlit untuk menjaga kompatibilitas impor."""

from src.ui_dashboard import render_dashboard_charts, render_dashboard_tab  # noqa: F401
from src.ui_playstore import render_playstore_tab, render_topic_reference_tab  # noqa: F401
from src.ui_styles import apply_custom_styles  # noqa: F401
from src.ui_tabs import (  # noqa: F401
    render_batch_analysis_results,
    render_text_tab,
    render_upload_tab,
)
