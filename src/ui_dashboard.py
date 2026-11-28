"""Komponen UI untuk tab Dashboard dinamis dan grafik interaktif Plotly."""

import html
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from src.config import MONTH_NAMES_ID, PLAYSTORE_APP_ID
from src.models import analyze_reviews_batch
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


def render_dashboard_charts(
    filtered_df: pd.DataFrame,
    selected_year: int,
    selected_month,
) -> None:
    """Render grafik interaktif Plotly untuk Dashboard ulasan."""
    import plotly.express as px

    sentiment_colors = {"Positif": "#0D9488", "Negatif": "#DC2626"}
    period_title = (
        f"Tahun {selected_year}"
        if selected_month is None or selected_month == "Semua Bulan"
        else f"{MONTH_NAMES_ID.get(int(selected_month), str(selected_month))} {selected_year}"
    )

    if selected_month == "Semua Bulan":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            monthly_df = build_monthly_trend_df(filtered_df)
            if not monthly_df.empty:
                st.markdown(
                    f"<h3 class='chart-title'>Tren Sentimen Bulanan ({html.escape(period_title)})</h3>",
                    unsafe_allow_html=True,
                )
                fig_sent_trend = px.bar(
                    monthly_df,
                    x="bulan_label",
                    y="Jumlah Ulasan",
                    color="Sentimen",
                    barmode="group",
                    color_discrete_map=sentiment_colors,
                    labels={"bulan_label": "Bulan"},
                )
                fig_sent_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="IBM Plex Sans, sans-serif"),
                    margin=dict(t=40, b=24, l=10, r=10),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                        title_text="",
                    ),
                )
                st.plotly_chart(fig_sent_trend, width="stretch")
            else:
                st.info("Belum ada data sentimen bulanan untuk ditampilkan.")

        with col_t2:
            topic_trend_df = build_topic_monthly_trend_df(filtered_df, top_n=5)
            if not topic_trend_df.empty:
                st.markdown(
                    f"<h3 class='chart-title'>Tren 5 Topik Teratas per Bulan ({html.escape(period_title)})</h3>",
                    unsafe_allow_html=True,
                )
                fig_topic_trend = px.line(
                    topic_trend_df,
                    x="bulan_label",
                    y="Jumlah Ulasan",
                    color="Label Topik",
                    markers=True,
                    labels={"bulan_label": "Bulan"},
                )
                fig_topic_trend.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="IBM Plex Sans, sans-serif"),
                    margin=dict(t=20, b=85, l=10, r=10),
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.28,
                        xanchor="left",
                        x=0,
                        title_text="",
                    ),
                )
                st.plotly_chart(fig_topic_trend, width="stretch")
            else:
                st.info("Belum ada topik cluster non-outlier untuk tren bulanan.")

    col_c1, col_c2 = st.columns([3, 2])
    ranking_df = build_topic_ranking_df(filtered_df)
    with col_c1:
        if not ranking_df.empty:
            st.markdown(
                f"<h3 class='chart-title'>Ranking Topik Cluster ({html.escape(period_title)})</h3>",
                unsafe_allow_html=True,
            )
            fig_topics = px.bar(
                ranking_df.sort_values("Jumlah Ulasan", ascending=True),
                x="Jumlah Ulasan",
                y="Label Topik",
                color="Sentimen",
                orientation="h",
                color_discrete_map=sentiment_colors,
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
                font=dict(family="IBM Plex Sans, sans-serif"),
                height=max(320, len(ranking_df) * 34 + 90),
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
            st.plotly_chart(fig_topics, width="stretch")
        else:
            st.info("Tidak ada data topik untuk periode ini.")

    with col_c2:
        rating_valid = filtered_df[
            filtered_df["score"].between(1, 5)
            & filtered_df["Sentimen"].isin(["Positif", "Negatif"])
        ]
        if not rating_valid.empty:
            st.markdown(
                f"<h3 class='chart-title'>Distribusi Rating Bintang ({html.escape(period_title)})</h3>",
                unsafe_allow_html=True,
            )
            rating_df = pd.DataFrame(
                {"Jumlah Ulasan": rating_valid.groupby(["score", "Sentimen"]).size()}
            ).reset_index()
            rating_df["Rating"] = rating_df["score"].astype(str) + " Bintang"
            fig_rating = px.bar(
                rating_df.sort_values("score"),
                x="Rating",
                y="Jumlah Ulasan",
                color="Sentimen",
                barmode="stack",
                color_discrete_map=sentiment_colors,
            )
            fig_rating.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="IBM Plex Sans, sans-serif"),
                height=max(320, len(ranking_df) * 34 + 90) if not ranking_df.empty else 340,
                margin=dict(t=40, b=24, l=10, r=10),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    title_text="",
                ),
            )
            st.plotly_chart(fig_rating, width="stretch")
        else:
            st.info("Tidak ada data rating untuk periode ini.")


def _render_dashboard_sync_panel(
    dashboard_df: pd.DataFrame,
    tokenizer,
    classifier,
    le,
    device,
    bertopic_models,
    embedding_model,
) -> pd.DataFrame:
    """Render panel sinkronisasi ulasan Google Play Store ke cache Dashboard."""
    with st.expander(
        "Sinkronkan Data Ulasan dari Google Play Store",
        expanded=dashboard_df.empty,
    ):
        st.caption(
            f"Tarik ulasan aplikasi Bank Jago (`{PLAYSTORE_APP_ID}`) pada rentang tanggal tertentu, "
            "klasifikasikan sentimen dan topiknya, lalu simpan ke cache Dashboard tanpa duplikat."
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
            "Tarik & Perbarui Dashboard",
            type="primary",
            width="stretch",
            key="dash_sync_btn",
        )

        if sync_btn:
            if not isinstance(sync_start, date) or not isinstance(sync_end, date):
                st.error("Pilih tanggal mulai dan tanggal akhir yang valid.")
            elif sync_start > sync_end:
                st.error("Tanggal mulai harus lebih kecil atau sama dengan tanggal akhir.")
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
                    st.error(str(exc))
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
                            f"Sinkronisasi selesai: {len(raw_sync_df):,} ulasan ditarik, "
                            f"total cache Dashboard kini {len(updated_dash_df):,} ulasan."
                        )

    return dashboard_df


def _render_dashboard_drilldown(
    filtered_dash_df: pd.DataFrame,
    bertopic_models: dict,
    selected_year: int,
    selected_month,
) -> None:
    """Render panel drill-down sampel ulasan per topik dan tombol unduh CSV."""
    st.markdown(
        "<h3 class='subsection-title'>Sampel Ulasan per Topik</h3>",
        unsafe_allow_html=True,
    )
    topic_choices_df = build_topic_ranking_df(filtered_dash_df)
    topic_labels_list = ["Semua Topik"] + topic_choices_df["Label Topik"].tolist()
    selected_topic_label = st.selectbox(
        "Pilih topik untuk melihat kata kunci dan daftar ulasan:",
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
                    chips_html = "".join(
                        f"<span class='keyword-chip'>{html.escape(str(w))}</span>"
                        for w, _ in words[:8]
                    )
                    st.markdown(
                        f"<div><strong>Kata kunci cluster:</strong></div>"
                        f"<div class='keyword-row'>{chips_html}</div>",
                        unsafe_allow_html=True,
                    )

    display_cols = [
        "tanggal", "score", "Sentimen", "Keyakinan (%)",
        "Cluster", "Nama Topik", "ulasan",
    ]
    display_df = pd.DataFrame(drill_df[display_cols]).reset_index(drop=True)
    st.caption(f"Menampilkan **{len(display_df):,}** ulasan.")
    st.dataframe(display_df, width="stretch", hide_index=True)

    dash_csv = display_df.to_csv(index=False).encode("utf-8-sig")
    month_slug = (
        "semua_bulan"
        if selected_month is None or selected_month == "Semua Bulan"
        else f"bulan_{int(selected_month):02d}"
    )
    st.download_button(
        label="Unduh Data Dashboard Periode Ini (CSV)",
        data=dash_csv,
        file_name=f"dashboard_ulasan_{selected_year}_{month_slug}.csv",
        mime="text/csv",
        width="stretch",
        key="dash_download_csv",
    )


def render_dashboard_tab(
    tokenizer, classifier, le, device, bertopic_models, embedding_model
) -> None:
    """Render tab Dashboard dinamis (sinkronisasi Play Store, filter Tahun->Bulan, KPI, Plotly)."""
    st.markdown("<h2 class='section-title'>Dashboard Monitoring Ulasan</h2>", unsafe_allow_html=True)

    if "dashboard_df" not in st.session_state:
        st.session_state["dashboard_df"] = load_dashboard_cache()

    dashboard_df = _render_dashboard_sync_panel(
        st.session_state["dashboard_df"],
        tokenizer,
        classifier,
        le,
        device,
        bertopic_models,
        embedding_model,
    )

    if dashboard_df.empty:
        st.markdown(
            """
            <div class="empty-guide">
                <strong>Belum ada data di Dashboard.</strong> Tekan <strong>Tarik &amp; Perbarui Dashboard</strong> di atas
                atau analisis ulasan melalui tab <strong>Ambil Data Play Store</strong> untuk mengisi ringkasan ini.
            </div>
            """,
            unsafe_allow_html=True,
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
    if selected_year is None:
        return

    year_df = dashboard_df[dashboard_df["tahun"] == int(selected_year)]
    available_months = sorted(pd.Series(year_df["bulan"]).dropna().unique().tolist())
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

    safe_top_topic = html.escape(str(metrics["top_topic"]))
    st.markdown(
        f"""
        <div class="topic-card">
            <div class="topic-card-id">Topik Dominan Periode Ini</div>
            <p class="topic-card-name">{safe_top_topic}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_dashboard_charts(filtered_dash_df, selected_year, selected_month)
    _render_dashboard_drilldown(
        filtered_dash_df, bertopic_models, selected_year, selected_month
    )
