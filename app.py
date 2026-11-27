"""
==============================================================================
STREAMLIT APP: Klasifikasi Sentimen + Clustering Topik Ulasan Bank Jago
==============================================================================
Alur:
  1. Dashboard dinamis ulasan Google Play Store per Tahun & Bulan
  2. Analisis satu ulasan dengan IndoBERT + BERTopic
  3. Analisis batch dari file CSV/Excel atau langsung dari Google Play Store

Jalankan: streamlit run app.py
==============================================================================
"""

import os

# Nonaktifkan inisialisasi TensorFlow yang tidak terpakai (aplikasi memakai PyTorch)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings

import streamlit as st

warnings.filterwarnings("ignore")

# Re-export konstanta dan fungsi publik agar kompatibel dengan test & pemanggil eksternal
from src.config import (  # noqa: E402, F401
    BERTOPIC_NEG_DIR,
    BERTOPIC_POS_DIR,
    DASHBOARD_CACHE_PATH,
    DASHBOARD_COLUMNS,
    MAX_LEN,
    MODEL_DIR,
    MONTH_NAMES_ID,
    MONTH_SHORT_ID,
    PLAYSTORE_APP_ID,
    SLANG_DICT,
    TOPIC_LABELS,
)
from src.models import (  # noqa: E402, F401
    DefaultBinaryLabelEncoder,
    _encode_for_bertopic,
    _extract_binary_probs,
    analyze_reviews_batch,
    build_topic_summary_df,
    load_bertopic_models,
    load_classifier,
    load_sentence_embedding_model,
    predict_sentiment,
    predict_sentiments_batch,
    predict_topic,
    resolve_topic_name,
)
from src.playstore_dashboard import (  # noqa: E402, F401
    build_dashboard_records_from_playstore,
    build_monthly_trend_df,
    build_topic_monthly_trend_df,
    build_topic_ranking_df,
    filter_dashboard_data,
    load_dashboard_cache,
    merge_dashboard_cache,
    save_dashboard_cache,
    scrape_playstore_reviews,
    summarize_dashboard_metrics,
)
from src.preprocessing import (  # noqa: E402, F401
    clean_text,
    detect_text_column,
    load_uploaded_table,
)
from src.ui_sections import (  # noqa: E402
    render_dashboard_tab,
    render_playstore_tab,
    render_text_tab,
    render_topic_reference_section,
    render_upload_tab,
)


def main() -> None:
    """Entry point utama aplikasi Streamlit."""
    st.set_page_config(
        page_title="Analisis Ulasan Bank Jago",
        page_icon="🏦",
        layout="wide",
    )

    st.markdown(
        """
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <h2 style="margin-bottom: 0.2rem;">🏦 Analisis Ulasan Bank Jago</h2>
            <p style="font-size: 0.95rem; color: #666; margin-top: 0;">
              Clustering Topik (BERTopic)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### ℹ️ Info Model")
        st.caption(
            "• **Sentimen**: IndoBERT (`indobert-base-p1`)\n"
            "• **Topik**: BERTopic (`paraphrase-multilingual-MiniLM-L12-v2`)"
        )
        st.markdown("---")
        st.caption("Kelompok 4 - Kelas A")

    model_exists = MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()
    if not model_exists:
        st.error(f"⚠️ Model IndoBERT belum ditemukan di `{MODEL_DIR}/`!")
        st.stop()

    tokenizer, classifier, le, device = load_classifier()
    bertopic_available = BERTOPIC_NEG_DIR.exists() or BERTOPIC_POS_DIR.exists()
    bertopic_models = {}
    embedding_model = None

    if bertopic_available:
        embedding_model = load_sentence_embedding_model()
        bertopic_models = load_bertopic_models(embedding_model)

    tab_dashboard, tab_text, tab_file, tab_playstore = st.tabs(
        [
            "📊 Dashboard",
            "✍️ Input Teks",
            "📁 Upload CSV / Excel",
            "📱 Ambil Data Play Store",
        ]
    )

    with tab_dashboard:
        render_dashboard_tab(
            tokenizer, classifier, le, device, bertopic_models, embedding_model
        )

    with tab_text:
        render_text_tab(
            tokenizer,
            classifier,
            le,
            device,
            bertopic_available,
            bertopic_models,
            embedding_model,
        )

    with tab_file:
        render_upload_tab(
            tokenizer, classifier, le, device, bertopic_models, embedding_model
        )

    with tab_playstore:
        render_playstore_tab(
            tokenizer, classifier, le, device, bertopic_models, embedding_model
        )

    if bertopic_available and bertopic_models:
        render_topic_reference_section(bertopic_models)

    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
        "Tugas Akhir — Analisis Ulasan Bank Jago | Kelompok 4"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
