"""Komponen antarmuka Streamlit untuk setiap tab dan grafik Plotly."""

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import MONTH_NAMES_ID, PLAYSTORE_APP_ID
from src.models import (
    analyze_reviews_batch,
    build_topic_summary_df,
    predict_sentiment,
    predict_topic,
)
from src.playstore_dashboard import (
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
from src.preprocessing import clean_text, detect_text_column, load_uploaded_table


def render_batch_analysis_results(
    result_df: pd.DataFrame, out_stem: str, key_prefix: str
) -> None:
    """Tampilkan ringkasan sentimen, distribusi topik cluster, tabel hasil, dan tombol download CSV."""
    st.markdown("---")
    st.markdown("### 📊 Ringkasan Hasil Analisis")

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

    valid_topic_df = result_df[result_df["Sentimen"].isin(["Positif", "Negatif"])]
    if not valid_topic_df.empty:
        dist_df = (
            valid_topic_df.groupby(["Sentimen", "Cluster", "Nama Topik"], as_index=False)
            .size()
            .rename(columns={"size": "Jumlah Ulasan"})
            .sort_values(["Sentimen", "Jumlah Ulasan"], ascending=[True, False])
        )
        with st.expander("📈 Distribusi Topik Cluster", expanded=True):
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📄 Hasil Lengkap per Ulasan")
    st.dataframe(result_df, use_container_width=True, hide_index=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Download Hasil Analisis (CSV)",
        data=csv_bytes,
        file_name=f"hasil_analisis_{out_stem}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{key_prefix}_download_csv",
    )


def render_dashboard_charts(
    filtered_df: pd.DataFrame,
    selected_year: int,
    selected_month,
) -> None:
    """Render grafik interaktif Plotly untuk Dashboard ulasan."""
    import plotly.express as px

    sentiment_colors = {"Positif": "#10B981", "Negatif": "#EF4444"}
    period_title = (
        f"Tahun {selected_year}"
        if selected_month == "Semua Bulan"
        else f"{MONTH_NAMES_ID.get(int(selected_month), selected_month)} {selected_year}"
    )

    if selected_month == "Semua Bulan":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            monthly_df = build_monthly_trend_df(filtered_df)
            if not monthly_df.empty:
                fig_sent_trend = px.bar(
                    monthly_df,
                    x="bulan_label",
                    y="Jumlah Ulasan",
                    color="Sentimen",
                    barmode="group",
                    color_discrete_map=sentiment_colors,
                    title=f"Tren Sentimen Bulanan ({period_title})",
                    labels={"bulan_label": "Bulan"},
                )
                fig_sent_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=45, b=20, l=10, r=10),
                    legend=dict(orientation="h", y=1.12),
                )
                st.plotly_chart(fig_sent_trend, use_container_width=True)
            else:
                st.info("Belum ada data sentimen bulanan untuk ditampilkan.")

        with col_t2:
            topic_trend_df = build_topic_monthly_trend_df(filtered_df, top_n=5)
            if not topic_trend_df.empty:
                fig_topic_trend = px.line(
                    topic_trend_df,
                    x="bulan_label",
                    y="Jumlah Ulasan",
                    color="Label Topik",
                    markers=True,
                    title=f"Tren 5 Topik Teratas per Bulan ({period_title})",
                    labels={"bulan_label": "Bulan"},
                )
                fig_topic_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=45, b=20, l=10, r=10),
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_topic_trend, use_container_width=True)
            else:
                st.info("Belum ada topik cluster non-outlier untuk tren bulanan.")

    col_c1, col_c2 = st.columns([3, 2])
    ranking_df = build_topic_ranking_df(filtered_df)
    with col_c1:
        if not ranking_df.empty:
            fig_topics = px.bar(
                ranking_df.sort_values("Jumlah Ulasan", ascending=True),
                x="Jumlah Ulasan",
                y="Label Topik",
                color="Sentimen",
                orientation="h",
                color_discrete_map=sentiment_colors,
                title=f"Ranking Topik Cluster ({period_title})",
                hover_data={
                    "Sentimen": True,
                    "Cluster": True,
                    "Nama Topik": True,
                    "Jumlah Ulasan": True,
                },
            )
            fig_topics.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(320, len(ranking_df) * 34 + 90),
                margin=dict(t=45, b=20, l=10, r=10),
                yaxis_title="",
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_topics, use_container_width=True)
        else:
            st.info("Tidak ada data topik untuk periode ini.")

    with col_c2:
        rating_valid = filtered_df[
            filtered_df["score"].between(1, 5)
            & filtered_df["Sentimen"].isin(["Positif", "Negatif"])
        ]
        if not rating_valid.empty:
            rating_df = (
                rating_valid.groupby(["score", "Sentimen"], as_index=False)
                .size()
                .rename(columns={"size": "Jumlah Ulasan"})
            )
            rating_df["Rating"] = rating_df["score"].astype(str) + " Bintang"
            fig_rating = px.bar(
                rating_df.sort_values("score"),
                x="Rating",
                y="Jumlah Ulasan",
                color="Sentimen",
                barmode="stack",
                color_discrete_map=sentiment_colors,
                title=f"Distribusi Rating Bintang ({period_title})",
            )
            fig_rating.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=max(320, len(ranking_df) * 34 + 90) if not ranking_df.empty else 340,
                margin=dict(t=45, b=20, l=10, r=10),
                legend=dict(orientation="h", y=1.08),
            )
            st.plotly_chart(fig_rating, use_container_width=True)
        else:
            st.info("Tidak ada data rating untuk periode ini.")


def render_dashboard_tab(
    tokenizer, classifier, le, device, bertopic_models, embedding_model
) -> None:
    """Render tab Dashboard dinamis (sinkronisasi Play Store, filter Tahun->Bulan, KPI, Plotly)."""
    if "dashboard_df" not in st.session_state:
        st.session_state["dashboard_df"] = load_dashboard_cache()

    dashboard_df = st.session_state["dashboard_df"]

    with st.expander(
        "🔄 Sinkronkan Data Ulasan dari Google Play Store",
        expanded=dashboard_df.empty,
    ):
        st.caption(
            f"Tarik ulasan aplikasi Bank Jago (`{PLAYSTORE_APP_ID}`) pada rentang tanggal tertentu, "
            "otomatis klasifikasi sentimen & topik, lalu simpan ke cache Dashboard (tanpa duplikat)."
        )
        dash_today = date.today()
        dash_default_start = dash_today - timedelta(days=90)

        s_col1, s_col2, s_col3 = st.columns(3)
        sync_start = s_col1.date_input(
            "Tanggal Mulai Sinkronisasi",
            value=dash_default_start,
            max_value=dash_today,
            key="dash_sync_start",
        )
        sync_end = s_col2.date_input(
            "Tanggal Akhir Sinkronisasi",
            value=dash_today,
            max_value=dash_today,
            key="dash_sync_end",
        )
        sync_limit = int(
            s_col3.number_input(
                "Maksimal Ulasan per Sinkronisasi",
                min_value=20,
                max_value=500,
                value=150,
                step=10,
                key="dash_sync_limit",
            )
        )

        sync_btn = st.button(
            "🔄 Tarik & Update Dashboard",
            type="primary",
            use_container_width=True,
            key="dash_sync_btn",
        )

        if sync_btn:
            if sync_start > sync_end:
                st.error("⚠️ Tanggal mulai tidak boleh lebih besar daripada tanggal akhir.")
            else:
                sync_prog = st.progress(0.0, text="Mengambil ulasan dari Google Play Store...")

                def _sync_scrape_cb(val):
                    pct = int(min(max(val, 0.0), 1.0) * 50)
                    sync_prog.progress(
                        min(max(val, 0.0), 1.0) * 0.5,
                        text=f"Mengambil ulasan Play Store... ({pct}%)",
                    )

                def _sync_analyze_cb(val):
                    pct = 50 + int(min(max(val, 0.0), 1.0) * 50)
                    sync_prog.progress(
                        0.5 + min(max(val, 0.0), 1.0) * 0.5,
                        text=f"Menganalisis sentimen & topik... ({pct}%)",
                    )

                try:
                    raw_sync_df = scrape_playstore_reviews(
                        start_date=sync_start,
                        end_date=sync_end,
                        max_reviews=sync_limit,
                        progress_callback=_sync_scrape_cb,
                    )
                except (ValueError, RuntimeError) as exc:
                    sync_prog.empty()
                    st.error(f"⚠️ {exc}")
                else:
                    if raw_sync_df.empty:
                        sync_prog.empty()
                        st.info(
                            f"Tidak ditemukan ulasan baru di Play Store pada rentang {sync_start} s/d {sync_end}."
                        )
                    else:
                        analyzed_sync_df = analyze_reviews_batch(
                            raw_sync_df["ulasan"].tolist(),
                            tokenizer,
                            classifier,
                            le,
                            device,
                            bertopic_models,
                            embedding_model,
                            progress_callback=_sync_analyze_cb,
                        )
                        new_records = build_dashboard_records_from_playstore(
                            raw_sync_df, analyzed_sync_df
                        )
                        updated_dash_df = merge_dashboard_cache(dashboard_df, new_records)
                        save_dashboard_cache(updated_dash_df)
                        st.session_state["dashboard_df"] = updated_dash_df
                        dashboard_df = updated_dash_df
                        sync_prog.empty()
                        st.success(
                            f"✅ Sinkronisasi selesai: {len(raw_sync_df):,} ulasan ditarik, "
                            f"total cache Dashboard kini {len(updated_dash_df):,} ulasan."
                        )

    if dashboard_df.empty:
        st.info(
            "Belum ada data ulasan di cache Dashboard. Klik **🔄 Tarik & Update Dashboard** di atas "
            "atau analisis ulasan dari tab **📱 Ambil Data Play Store** untuk mengisi Dashboard."
        )
        return

    available_years = sorted(dashboard_df["tahun"].unique().tolist(), reverse=True)
    f_col1, f_col2, f_col3 = st.columns(3)
    selected_year = f_col1.selectbox(
        "1. Pilih Tahun (Wajib)",
        options=available_years,
        index=0,
        key="dash_year_select",
    )

    year_df = dashboard_df[dashboard_df["tahun"] == int(selected_year)]
    available_months = sorted(year_df["bulan"].unique().tolist())
    month_options = ["Semua Bulan"] + available_months
    selected_month = f_col2.selectbox(
        "2. Pilih Bulan",
        options=month_options,
        index=0,
        format_func=lambda m: (
            "Semua Bulan"
            if m == "Semua Bulan"
            else f"{int(m):02d} - {MONTH_NAMES_ID.get(int(m), str(m))}"
        ),
        key="dash_month_select",
    )
    selected_sentiment = f_col3.selectbox(
        "3. Filter Sentimen",
        options=["Semua", "Positif", "Negatif"],
        index=0,
        key="dash_sentiment_select",
    )

    filtered_dash_df = filter_dashboard_data(
        dashboard_df, selected_year, selected_month, selected_sentiment
    )

    if filtered_dash_df.empty:
        st.warning(
            "Tidak ada ulasan yang cocok dengan kombinasi filter Tahun, Bulan, dan Sentimen tersebut."
        )
        return

    metrics = summarize_dashboard_metrics(filtered_dash_df)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Ulasan", f"{metrics['total']:,}")
    k2.metric(
        "Sentimen Positif",
        f"{metrics['positif_count']:,}",
        f"{metrics['positif_pct']:.1f}%",
    )
    k3.metric(
        "Sentimen Negatif",
        f"{metrics['negatif_count']:,}",
        f"{metrics['negatif_pct']:.1f}%",
    )
    k4.metric("Rata-rata Rating", f"{metrics['avg_rating']:.2f} / 5.0")

    st.info(f"🏷️ **Topik Dominan Periode Ini:** {metrics['top_topic']}")

    render_dashboard_charts(filtered_dash_df, selected_year, selected_month)

    st.markdown("#### 🔍 Drill-down Sampel Ulasan per Topik")
    topic_choices_df = build_topic_ranking_df(filtered_dash_df)
    topic_labels_list = ["Semua Topik"] + topic_choices_df["Label Topik"].tolist()
    selected_topic_label = st.selectbox(
        "Pilih Topik untuk melihat kata kunci dan daftar ulasan:",
        options=topic_labels_list,
        key="dash_drilldown_topic",
    )

    drill_df = filtered_dash_df.copy()
    drill_df["Label Topik"] = drill_df["Cluster"] + " - " + drill_df["Nama Topik"]
    if selected_topic_label != "Semua Topik":
        drill_df = drill_df[drill_df["Label Topik"] == selected_topic_label]
        matched_row = topic_choices_df[
            topic_choices_df["Label Topik"] == selected_topic_label
        ]
        if not matched_row.empty:
            sel_sent = matched_row.iloc[0]["Sentimen"]
            sel_cluster = str(matched_row.iloc[0]["Cluster"])
            if (
                sel_cluster.startswith("#")
                and bertopic_models
                and sel_sent in bertopic_models
            ):
                tid = int(sel_cluster.lstrip("#"))
                words = bertopic_models[sel_sent].get_topic(tid) or []
                if words:
                    kw_text = " • ".join(f"`{w}`" for w, _ in words[:8])
                    st.caption(f"**Kata Kunci Cluster:** {kw_text}")

    display_cols = [
        "tanggal", "score", "Sentimen", "Keyakinan (%)",
        "Cluster", "Nama Topik", "ulasan",
    ]
    st.caption(f"Menampilkan **{len(drill_df):,}** ulasan.")
    st.dataframe(
        drill_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    dash_csv = drill_df[display_cols].to_csv(index=False).encode("utf-8-sig")
    month_slug = (
        "semua_bulan"
        if selected_month == "Semua Bulan"
        else f"bulan_{int(selected_month):02d}"
    )
    st.download_button(
        label="⬇️ Download Data Dashboard Periode Ini (CSV)",
        data=dash_csv,
        file_name=f"dashboard_ulasan_{selected_year}_{month_slug}.csv",
        mime="text/csv",
        use_container_width=True,
        key="dash_download_csv",
    )


def render_text_tab(
    tokenizer, classifier, le, device, bertopic_available, bertopic_models, embedding_model
) -> None:
    """Render tab analisis satu ulasan teks."""
    user_input = st.text_area(
        "Tulis ulasan Bank Jago:",
        height=110,
        placeholder="Ketik atau tempel kalimat ulasan di sini...",
    )

    analyze_btn = st.button("🔍 Analisis Ulasan", type="primary", use_container_width=True)
    if not analyze_btn:
        return

    if not user_input.strip():
        st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")
        return

    cleaned = clean_text(user_input)
    if not cleaned.strip():
        st.warning(
            "⚠️ Teks ulasan kosong setelah dibersihkan (hanya emoji/karakter khusus)."
        )
        return

    sentimen, confidence, probs = predict_sentiment(
        cleaned, tokenizer, classifier, le, device
    )

    st.markdown("---")
    st.markdown("### 📊 Hasil Analisis")

    col_sent, col_conf = st.columns(2)
    with col_sent:
        if sentimen == "Positif":
            st.success(f"**Sentimen:** ✅ **{sentimen}**")
        else:
            st.error(f"**Sentimen:** ❌ **{sentimen}**")

    with col_conf:
        st.metric("Tingkat Keyakinan", f"{confidence * 100:.1f}%")

    for i, cls in enumerate(le.classes_):
        prob_val = float(probs[i])
        color_dot = "🟢" if cls == "Positif" else "🔴"
        st.caption(f"{color_dot} Probabilitas {cls}: **{prob_val * 100:.1f}%**")
        st.progress(prob_val)

    if bertopic_available and bertopic_models:
        st.markdown("#### 🏷️ Topik Ulasan")
        topic_id, topic_name, topic_info = predict_topic(
            cleaned, sentimen, bertopic_models, embedding_model
        )

        if topic_id is not None and topic_id != -1:
            st.info(f"**Cluster #{topic_id}**: {topic_name}")
            if topic_info and isinstance(topic_info, list):
                keywords = " • ".join([f"`{word}`" for word, _ in topic_info[:7]])
                st.caption(f"**Kata Kunci:** {keywords}")
        elif topic_id == -1:
            st.warning(
                "Ulasan ini dikategorikan sebagai *outlier* (tidak masuk kelompok topik dominan)."
            )
        else:
            st.info(str(topic_info))

    with st.expander("🔍 Lihat Hasil Preprocessing Teks"):
        st.write("**Teks Bersih:**", cleaned)
        st.caption(f"Jumlah kata: {len(cleaned.split())} kata")


def render_upload_tab(
    tokenizer, classifier, le, device, bertopic_models, embedding_model
) -> None:
    """Render tab upload file CSV/Excel untuk analisis batch."""
    uploaded_file = st.file_uploader(
        "Upload file berisi ulasan (CSV atau Excel):",
        type=["csv", "xlsx", "xls"],
    )
    if uploaded_file is None:
        return

    try:
        uploaded_df = load_uploaded_table(uploaded_file, uploaded_file.name)
    except ValueError as err:
        st.error(f"⚠️ {err}")
        return

    st.caption(
        f"File **{uploaded_file.name}** terbaca: **{len(uploaded_df):,}** baris, **{len(uploaded_df.columns)}** kolom."
    )
    with st.expander("👀 Preview Data (5 baris pertama)", expanded=True):
        st.dataframe(uploaded_df.head(5), use_container_width=True, hide_index=True)

    columns_list = list(uploaded_df.columns)
    default_col = detect_text_column(uploaded_df)
    default_idx = columns_list.index(default_col) if default_col in columns_list else 0
    selected_col = st.selectbox(
        "Pilih kolom teks ulasan:",
        options=columns_list,
        index=default_idx,
    )

    analyze_file_btn = st.button(
        "🚀 Analisis File",
        type="primary",
        use_container_width=True,
    )

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


def render_playstore_tab(
    tokenizer, classifier, le, device, bertopic_models, embedding_model
) -> None:
    """Render tab pengambilan data ulasan Play Store berfilter tanggal + analisis batch."""
    st.caption(
        f"Ambil ulasan aplikasi Bank Jago (`{PLAYSTORE_APP_ID}`) dari Google Play Store "
        "berdasarkan filter tanggal sebagai input analisis."
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
        "🔄 Ambil Ulasan Play Store",
        type="primary",
        use_container_width=True,
        key="ps_fetch_btn",
    )

    if fetch_btn:
        if start_date > end_date:
            st.error("⚠️ Tanggal mulai tidak boleh lebih besar daripada tanggal akhir.")
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
                st.error(f"⚠️ {exc}")
            else:
                fetch_progress.empty()
                st.session_state["ps_scraped_df"] = scraped_df
                st.session_state["ps_scraped_range"] = (str(start_date), str(end_date))
                st.session_state.pop("ps_result_df", None)

    if "ps_scraped_df" not in st.session_state:
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
    with st.expander("👀 Preview Data Ulasan Play Store", expanded=True):
        st.dataframe(scraped_df, use_container_width=True, hide_index=True)

    act_col1, act_col2 = st.columns(2)
    analyze_ps_btn = act_col1.button(
        "🚀 Analisis Data Play Store",
        type="primary",
        use_container_width=True,
        key="ps_analyze_btn",
    )
    raw_csv_bytes = scraped_df.to_csv(index=False).encode("utf-8-sig")
    act_col2.download_button(
        label="⬇️ Download Data Mentah (CSV)",
        data=raw_csv_bytes,
        file_name=f"ulasan_playstore_{r_start}_{r_end}.csv",
        mime="text/csv",
        use_container_width=True,
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
            f"✅ Data juga disimpan ke cache Dashboard (total {len(updated_dash):,} ulasan)."
        )

    if "ps_result_df" in st.session_state:
        res_start, res_end = st.session_state.get("ps_result_range", (r_start, r_end))
        render_batch_analysis_results(
            st.session_state["ps_result_df"],
            f"playstore_{res_start}_{res_end}",
            "ps_batch",
        )


def render_topic_reference_section(bertopic_models: dict) -> None:
    """Render daftar seluruh cluster topik bernama di bagian bawah halaman."""
    st.markdown("---")
    st.markdown("### 📋 Daftar Cluster Topik")
    available_sentiments = [s for s in ("Negatif", "Positif") if s in bertopic_models]
    if not available_sentiments:
        return

    tab_labels = [
        f"{'❌' if s == 'Negatif' else '✅'} Sentimen {s}"
        for s in available_sentiments
    ]
    tabs = st.tabs(tab_labels)
    for tab, sentimen_key in zip(tabs, available_sentiments):
        with tab:
            summary_df = build_topic_summary_df(
                sentimen_key, bertopic_models[sentimen_key]
            )
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
