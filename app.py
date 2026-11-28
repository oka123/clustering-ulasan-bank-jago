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
    SAMPLE_REVIEWS,
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
    build_batch_distribution_df,
    build_dashboard_records_from_playstore,
    build_monthly_trend_df,
    build_topic_monthly_trend_df,
    build_topic_ranking_df,
    filter_batch_results,
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
    apply_custom_styles,
    render_dashboard_tab,
    render_playstore_tab,
    render_text_tab,
    render_topic_reference_tab,
    render_upload_tab,
)


def main() -> None:
    """Entry point utama aplikasi Streamlit."""
    st.set_page_config(
        page_title="Analisis Ulasan Bank Jago",
        layout="wide",
    )
    apply_custom_styles()

    st.markdown(
        """
        <header class="hero-header">
            <h1 class="hero-title">Analisis Ulasan Bank Jago</h1>
            <p class="hero-subtitle">
                Klasifikasi sentimen IndoBERT dan clustering topik otomatis menggunakan BERTopic.
            </p>
        </header>
        """,
        unsafe_allow_html=True,
    )

    model_exists = MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()
    if not model_exists:
        st.error(
            f"Model IndoBERT tidak ditemukan di folder `{MODEL_DIR}/`. "
            "Pastikan file `config.json` dan bobot model sudah tersedia."
        )
        st.stop()

    bertopic_available = BERTOPIC_NEG_DIR.exists() or BERTOPIC_POS_DIR.exists()

    with st.sidebar:
        st.markdown("<h2 class='section-title'>Informasi Model</h2>", unsafe_allow_html=True)
        st.markdown(
            "**Klasifikasi Sentimen**  \n"
            "`agufsamudra/indo-sentiment-analysis` (IndoBERT)"
        )
        st.markdown(
            "**Clustering Topik**  \n"
            "`BERTopic` dengan embedding `paraphrase-multilingual-MiniLM-L12-v2`"
        )
        st.markdown("---")
        bertopic_status = (
            "Tersedia (Negatif & Positif)"
            if (BERTOPIC_NEG_DIR.exists() and BERTOPIC_POS_DIR.exists())
            else ("Parsial" if bertopic_available else "Tidak tersedia")
        )
        st.caption(f"Status BERTopic: {bertopic_status}")
        st.caption("Tugas Akhir | Kelompok 4 - Kelas A")

    with st.spinner("Memuat model IndoBERT dan BERTopic..."):
        tokenizer, classifier, le, device = load_classifier()
        bertopic_models = {}
        embedding_model = None

        if bertopic_available:
            embedding_model = load_sentence_embedding_model()
            bertopic_models = load_bertopic_models(embedding_model)

    tab_dashboard, tab_text, tab_file, tab_playstore, tab_topics = st.tabs(
        [
            "Dashboard",
            "Analisis Ulasan",
            "Upload CSV / Excel",
            "Ambil Data Play Store",
            "Referensi Topik",
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

    with tab_topics:
        render_topic_reference_tab(bertopic_available, bertopic_models)

    st.markdown(
        """
        <footer class="app-footer">
            Tugas Akhir | Analisis Ulasan Bank Jago | Kelompok 4A
        </footer>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
