"""Komponen UI untuk tab Analisis Ulasan dan Upload CSV/Excel."""

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import SAMPLE_REVIEWS
from src.models import (
    analyze_reviews_batch,
    predict_sentiment,
    predict_topic,
)
from src.playstore_dashboard import (
    build_batch_distribution_df,
    filter_batch_results,
)
from src.preprocessing import clean_text, detect_text_column, load_uploaded_table


def _get_sample_csv_bytes() -> bytes:
    """Ambil bytes contoh CSV untuk tombol unduh pada empty state tab upload."""
    sample_path = Path("samples/contoh_ulasan_bank_jago.csv")
    if sample_path.exists():
        return sample_path.read_bytes()
    fallback_df = pd.DataFrame(
        {
            "ulasan": [
                "Aplikasinya sangat membantu, transfer gratis ke mana saja dan cepat masuk",
                "Susah banget login di hp baru, verifikasi email tidak masuk padahal data sudah benar",
                "Fitur QRIS nya mantap, bayar di kasir tinggal scan langsung beres",
            ],
            "rating": [5, 1, 5],
        }
    )
    return fallback_df.to_csv(index=False).encode("utf-8-sig")


def _set_sample_review(sample_text: str) -> None:
    """Callback untuk mengisi input ulasan dari tombol contoh cepat."""
    st.session_state["single_review_input"] = sample_text


def render_batch_analysis_results(
    result_df: pd.DataFrame, out_stem: str, key_prefix: str
) -> None:
    """Tampilkan ringkasan metrik, grafik cluster, filter tabel, dan tombol unduh CSV."""
    import plotly.express as px

    st.markdown("---")
    st.markdown(
        "<h3 class='subsection-title'>Ringkasan Hasil Analisis</h3>",
        unsafe_allow_html=True,
    )

    total_rows = len(result_df)
    pos_count = int((result_df["Sentimen"] == "Positif").sum())
    neg_count = int((result_df["Sentimen"] == "Negatif").sum())

    col_tot, col_pos, col_neg = st.columns(3)
    col_tot.metric("Total Ulasan", f"{total_rows:,}")
    col_pos.metric(
        "Sentimen Positif",
        f"{pos_count:,}",
        f"{(pos_count / total_rows * 100):.1f}%" if total_rows else "0.0%",
    )
    col_neg.metric(
        "Sentimen Negatif",
        f"{neg_count:,}",
        f"{(neg_count / total_rows * 100):.1f}%" if total_rows else "0.0%",
    )

    dist_df = build_batch_distribution_df(result_df)
    if not dist_df.empty:
        st.markdown(
            "<h3 class='chart-title'>Distribusi Topik Cluster</h3>",
            unsafe_allow_html=True,
        )
        fig_batch = px.bar(
            dist_df.sort_values("Jumlah Ulasan", ascending=True),
            x="Jumlah Ulasan",
            y="Label Topik",
            color="Sentimen",
            orientation="h",
            color_discrete_map={"Positif": "#0D9488", "Negatif": "#DC2626"},
            hover_data={"Sentimen": True, "Cluster": True, "Nama Topik": True},
        )
        fig_batch.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif"),
            height=max(260, len(dist_df) * 34 + 80),
            margin=dict(t=40, b=24, l=10, r=10),
            yaxis_title="",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                title_text="",
            ),
        )
        st.plotly_chart(fig_batch, width="stretch")

        with st.expander("Tabel Ringkasan Distribusi Topik Cluster"):
            st.dataframe(dist_df, width="stretch", hide_index=True)

    st.markdown(
        "<h3 class='subsection-title'>Hasil Lengkap per Ulasan</h3>",
        unsafe_allow_html=True,
    )

    f_col1, f_col2 = st.columns(2)
    sentiment_options = ["Semua"] + sorted(result_df["Sentimen"].dropna().unique().tolist())
    cluster_options = ["Semua"] + sorted(result_df["Cluster"].dropna().unique().tolist())
    selected_sentiment = f_col1.selectbox(
        "Filter Sentimen",
        options=sentiment_options,
        key=f"{key_prefix}_filter_sentiment",
    )
    selected_cluster = f_col2.selectbox(
        "Filter Cluster",
        options=cluster_options,
        key=f"{key_prefix}_filter_cluster",
    )

    filtered_df = filter_batch_results(result_df, selected_sentiment, selected_cluster)
    st.caption(
        f"Menampilkan **{len(filtered_df):,}** dari **{total_rows:,}** baris hasil analisis."
    )
    if filtered_df.empty:
        st.info("Tidak ada baris yang cocok dengan kombinasi filter yang dipilih.")
    else:
        st.dataframe(filtered_df, width="stretch", hide_index=True)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Unduh Hasil Analisis (CSV)",
        data=csv_bytes,
        file_name=f"hasil_analisis_{out_stem}.csv",
        mime="text/csv",
        width="stretch",
        key=f"{key_prefix}_download_csv",
    )


def render_text_tab(
    tokenizer, classifier, le, device, bertopic_available, bertopic_models, embedding_model
) -> None:
    """Render tab analisis satu ulasan teks dengan contoh cepat dan loading feedback."""
    st.markdown("<h2 class='section-title'>Analisis Ulasan Tunggal</h2>", unsafe_allow_html=True)

    if "single_review_input" not in st.session_state:
        st.session_state["single_review_input"] = ""

    st.caption("Contoh cepat (klik untuk mengisi kolom ulasan):")
    preset_cols = st.columns(len(SAMPLE_REVIEWS))
    for col, (label, sample_text) in zip(preset_cols, SAMPLE_REVIEWS):
        col.button(
            label,
            key=f"preset_{label}",
            width="stretch",
            on_click=_set_sample_review,
            args=(sample_text,),
        )

    user_input = st.text_area(
        "Teks ulasan pengguna Bank Jago",
        key="single_review_input",
        height=120,
        placeholder="Ketik atau tempel kalimat ulasan di sini...",
    )

    analyze_btn = st.button("Analisis Ulasan", type="primary", width="stretch")
    if not analyze_btn:
        st.markdown(
            """
            <div class="empty-guide">
                <strong>Panduan singkat:</strong> Masukkan satu kalimat ulasan atau pilih salah satu contoh di atas,
                lalu tekan <strong>Analisis Ulasan</strong> untuk melihat klasifikasi sentimen dan topik cluster.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    if not user_input.strip():
        st.warning("Masukkan teks ulasan terlebih dahulu sebelum memulai analisis.")
        return

    cleaned = clean_text(user_input)
    if not cleaned.strip():
        st.warning(
            "Teks ulasan menjadi kosong setelah dibersihkan (hanya berisi emoji atau tanda baca). "
            "Tambahkan kata ulasan yang bermakna."
        )
        return

    with st.spinner("Menganalisis sentimen dan topik ulasan..."):
        sentimen, confidence, probs = predict_sentiment(
            cleaned, tokenizer, classifier, le, device
        )
        topic_result = None
        if bertopic_available and bertopic_models:
            topic_result = predict_topic(
                cleaned, sentimen, bertopic_models, embedding_model
            )

    st.markdown("---")
    st.markdown("<h3 class='subsection-title'>Hasil Analisis</h3>", unsafe_allow_html=True)

    col_sent, col_conf = st.columns(2)
    with col_sent:
        variant_cls = "sentiment-card--pos" if sentimen == "Positif" else "sentiment-card--neg"
        safe_sentimen = html.escape(str(sentimen))
        st.markdown(
            f"""
            <div class="sentiment-card {variant_cls}">
                <div class="sentiment-label">Klasifikasi Sentimen</div>
                <div class="sentiment-value">{safe_sentimen}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_conf:
        st.metric("Tingkat Keyakinan", f"{confidence * 100:.1f}%")

    for i, cls in enumerate(le.classes_):
        prob_val = float(probs[i])
        safe_cls = html.escape(str(cls))
        st.markdown(
            f"""
            <div class="prob-row">
                <span>Probabilitas {safe_cls}</span>
                <span><strong>{prob_val * 100:.1f}%</strong></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(prob_val)

    if topic_result is not None:
        st.markdown("<h3 class='subsection-title'>Topik Ulasan</h3>", unsafe_allow_html=True)
        topic_id, topic_name, topic_info = topic_result

        if topic_id is not None and topic_id != -1:
            safe_name = html.escape(str(topic_name))
            st.markdown(
                f"""
                <div class="topic-card">
                    <div class="topic-card-id">Cluster #{int(topic_id)}</div>
                    <p class="topic-card-name">{safe_name}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if topic_info and isinstance(topic_info, list):
                chips_html = "".join(
                    f"<span class='keyword-chip'>{html.escape(str(word))}</span>"
                    for word, _ in topic_info[:7]
                )
                st.markdown(
                    f"<div><strong>Kata kunci terkait:</strong></div>"
                    f"<div class='keyword-row'>{chips_html}</div>",
                    unsafe_allow_html=True,
                )
        elif topic_id == -1:
            st.warning(
                "Ulasan ini dikategorikan sebagai outlier karena tidak masuk kelompok topik dominan."
            )
        else:
            st.info(str(topic_info))

    with st.expander("Detail Preprocessing Teks"):
        st.write("**Teks Bersih:**", cleaned)
        st.caption(f"Jumlah kata setelah pembersihan: {len(cleaned.split())} kata")


def render_upload_tab(
    tokenizer, classifier, le, device, bertopic_models, embedding_model
) -> None:
    """Render tab upload file CSV/Excel untuk analisis batch."""
    st.markdown(
        "<h2 class='section-title'>Analisis Batch dari File CSV / Excel</h2>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Unggah file berisi ulasan (.csv, .xlsx, atau .xls)",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded_file is None:
        st.markdown(
            """
            <div class="empty-guide">
                <strong>Format yang didukung:</strong> File <code>.csv</code> (UTF-8 atau Latin-1), <code>.xlsx</code>, atau <code>.xls</code>
                dengan minimal satu kolom teks ulasan (misalnya bernama <code>ulasan</code>, <code>review</code>, atau <code>komentar</code>).
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            label="Unduh File Contoh (CSV)",
            data=_get_sample_csv_bytes(),
            file_name="contoh_ulasan_bank_jago.csv",
            mime="text/csv",
        )
        return

    try:
        uploaded_df = load_uploaded_table(uploaded_file, uploaded_file.name)
    except ValueError as err:
        st.error(f"Gagal memproses file: {err}")
        return

    st.caption(
        f"File **{uploaded_file.name}** terbaca: **{len(uploaded_df):,}** baris, **{len(uploaded_df.columns)}** kolom."
    )
    with st.expander("Preview Data (5 baris pertama)", expanded=True):
        st.dataframe(uploaded_df.head(5), width="stretch", hide_index=True)

    columns_list = list(uploaded_df.columns)
    default_col = detect_text_column(uploaded_df)
    default_idx = (
        columns_list.index(default_col)
        if default_col is not None and default_col in columns_list
        else 0
    )
    selected_col = st.selectbox(
        "Pilih kolom yang berisi teks ulasan:",
        options=columns_list,
        index=default_idx,
    )

    analyze_file_btn = st.button("Analisis File", type="primary", width="stretch")

    batch_cache_key = (uploaded_file.name, uploaded_file.size, str(selected_col))
    if analyze_file_btn:
        progress_bar = st.progress(0.0, text="Memulai analisis ulasan...")

        def _update_progress(val):
            pct = int(min(max(val, 0.0), 1.0) * 100)
            progress_bar.progress(
                min(max(val, 0.0), 1.0), text=f"Memproses ulasan... ({pct}%)"
            )

        result_df = analyze_reviews_batch(
            uploaded_df[selected_col].tolist(),
            tokenizer,
            classifier,
            le,
            device,
            bertopic_models,
            embedding_model,
            progress_callback=_update_progress,
        )
        progress_bar.empty()
        st.session_state["batch_result_key"] = batch_cache_key
        st.session_state["batch_result_df"] = result_df

    if (
        st.session_state.get("batch_result_key") == batch_cache_key
        and "batch_result_df" in st.session_state
    ):
        render_batch_analysis_results(
            st.session_state["batch_result_df"],
            Path(uploaded_file.name).stem,
            "upload_batch",
        )

