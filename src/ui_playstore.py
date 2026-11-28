"""Komponen UI untuk tab Ambil Data Play Store dan Referensi Cluster Topik."""

from datetime import date, timedelta

import streamlit as st

from src.config import PLAYSTORE_APP_ID
from src.models import analyze_reviews_batch, build_topic_summary_df
from src.playstore_dashboard import (
    build_dashboard_records_from_playstore,
    load_dashboard_cache,
    merge_dashboard_cache,
    save_dashboard_cache,
    scrape_playstore_reviews,
)
from src.ui_tabs import render_batch_analysis_results


def render_playstore_tab(
    tokenizer, classifier, le, device, bertopic_models, embedding_model
) -> None:
    """Render tab pengambilan data ulasan Play Store berfilter tanggal + analisis batch."""
    st.markdown(
        "<h2 class='section-title'>Ambil Data Ulasan Google Play Store</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Mengambil ulasan aplikasi Bank Jago (`{PLAYSTORE_APP_ID}`) berdasarkan rentang tanggal sebagai input analisis."
    )

    today = date.today()
    default_start = today - timedelta(days=30)

    p_col1, p_col2, p_col3 = st.columns(3)
    start_date = p_col1.date_input(
        "Tanggal Mulai",
        value=default_start,
        max_value=today,
        key="ps_start_date",
    )
    end_date = p_col2.date_input(
        "Tanggal Akhir",
        value=today,
        max_value=today,
        key="ps_end_date",
    )
    max_reviews = int(
        p_col3.number_input(
            "Maksimal Ulasan",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            key="ps_max_reviews",
        )
    )

    fetch_btn = st.button(
        "Ambil Ulasan Play Store",
        type="primary",
        width="stretch",
        key="ps_fetch_btn",
    )

    if fetch_btn:
        if not isinstance(start_date, date) or not isinstance(end_date, date):
            st.error("Pilih tanggal mulai dan tanggal akhir yang valid.")
        elif start_date > end_date:
            st.error("Tanggal mulai harus lebih kecil atau sama dengan tanggal akhir.")
        else:
            fetch_progress = st.progress(0.0, text="Menghubungi Google Play Store...")

            def _update_fetch_progress(val):
                pct = int(min(max(val, 0.0), 1.0) * 100)
                fetch_progress.progress(
                    min(max(val, 0.0), 1.0),
                    text=f"Mengambil ulasan Play Store... ({pct}%)",
                )

            try:
                scraped_df = scrape_playstore_reviews(
                    start_date=start_date,
                    end_date=end_date,
                    max_reviews=max_reviews,
                    progress_callback=_update_fetch_progress,
                )
            except (ValueError, RuntimeError) as exc:
                fetch_progress.empty()
                st.error(str(exc))
            else:
                fetch_progress.empty()
                st.session_state["ps_scraped_df"] = scraped_df
                st.session_state["ps_scraped_range"] = (str(start_date), str(end_date))
                st.session_state.pop("ps_result_df", None)

    if "ps_scraped_df" not in st.session_state:
        st.markdown(
            """
            <div class="empty-guide">
                <strong>Langkah penggunaan:</strong> Tentukan <strong>Tanggal Mulai</strong>, <strong>Tanggal Akhir</strong>,
                dan batas maksimal ulasan, lalu klik <strong>Ambil Ulasan Play Store</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    scraped_df = st.session_state["ps_scraped_df"]
    r_start, r_end = st.session_state.get(
        "ps_scraped_range", (str(start_date), str(end_date))
    )

    if scraped_df.empty:
        st.info(
            f"Tidak ada ulasan ditemukan pada rentang tanggal **{r_start}** sampai **{r_end}**. "
            "Coba perluas rentang tanggal."
        )
        return

    st.caption(
        f"Berhasil mengambil **{len(scraped_df):,}** ulasan pada rentang **{r_start}** s/d **{r_end}**."
    )
    with st.expander("Preview Data Ulasan Play Store", expanded=True):
        st.dataframe(scraped_df, width="stretch", hide_index=True)

    act_col1, act_col2 = st.columns(2)
    analyze_ps_btn = act_col1.button(
        "Analisis Data Play Store",
        type="primary",
        width="stretch",
        key="ps_analyze_btn",
    )
    raw_csv_bytes = scraped_df.to_csv(index=False).encode("utf-8-sig")
    act_col2.download_button(
        label="Unduh Data Mentah (CSV)",
        data=raw_csv_bytes,
        file_name=f"ulasan_playstore_{r_start}_{r_end}.csv",
        mime="text/csv",
        width="stretch",
        key="ps_download_raw",
    )

    if analyze_ps_btn:
        ps_progress = st.progress(0.0, text="Memulai analisis ulasan Play Store...")

        def _update_ps_analysis(val):
            pct = int(min(max(val, 0.0), 1.0) * 100)
            ps_progress.progress(
                min(max(val, 0.0), 1.0),
                text=f"Menganalisis ulasan... ({pct}%)",
            )

        ps_result_df = analyze_reviews_batch(
            scraped_df["ulasan"].tolist(),
            tokenizer,
            classifier,
            le,
            device,
            bertopic_models,
            embedding_model,
            progress_callback=_update_ps_analysis,
        )
        ps_result_df.insert(0, "Tanggal", scraped_df["tanggal"].values)
        ps_result_df.insert(1, "Rating", scraped_df["score"].values)

        new_dash_rows = build_dashboard_records_from_playstore(scraped_df, ps_result_df)
        existing_dash = st.session_state.get("dashboard_df", load_dashboard_cache())
        updated_dash = merge_dashboard_cache(existing_dash, new_dash_rows)
        save_dashboard_cache(updated_dash)
        st.session_state["dashboard_df"] = updated_dash

        ps_progress.empty()
        st.session_state["ps_result_df"] = ps_result_df
        st.session_state["ps_result_range"] = (r_start, r_end)
        st.success(
            f"Data selesai dianalisis dan disimpan ke cache Dashboard (total {len(updated_dash):,} ulasan)."
        )

    if "ps_result_df" in st.session_state:
        res_start, res_end = st.session_state.get("ps_result_range", (r_start, r_end))
        render_batch_analysis_results(
            st.session_state["ps_result_df"],
            f"playstore_{res_start}_{res_end}",
            "ps_batch",
        )


def render_topic_reference_tab(bertopic_available: bool, bertopic_models: dict) -> None:
    """Render tab khusus daftar seluruh cluster topik bernama."""
    st.markdown(
        "<h2 class='section-title'>Referensi Cluster Topik</h2>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Daftar seluruh cluster topik dari model BERTopic beserta nama label dan kata kunci utamanya."
    )
    if not (bertopic_available and bertopic_models):
        st.warning(
            "Folder model BERTopic (`model_bertopic_negatif/` atau `model_bertopic_positif/`) tidak ditemukan."
        )
        return

    available_sentiments = [s for s in ("Negatif", "Positif") if s in bertopic_models]
    if not available_sentiments:
        st.info("Model BERTopic tidak memiliki topik yang termuat.")
        return

    tabs = st.tabs([f"Sentimen {s}" for s in available_sentiments])
    for tab, sentimen_key in zip(tabs, available_sentiments):
        with tab:
            summary_df = build_topic_summary_df(
                sentimen_key, bertopic_models[sentimen_key]
            )
            st.dataframe(summary_df, width="stretch", hide_index=True)
