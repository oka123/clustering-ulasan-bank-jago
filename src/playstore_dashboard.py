"""Scraping ulasan Google Play Store, manajemen cache, dan agregasi Dashboard."""

from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from src.config import (
    DASHBOARD_CACHE_PATH,
    DASHBOARD_COLUMNS,
    MONTH_SHORT_ID,
    PLAYSTORE_APP_ID,
)


def scrape_playstore_reviews(
    start_date,
    end_date,
    max_reviews: int | None = 100,
    app_id: str = PLAYSTORE_APP_ID,
    progress_callback=None,
) -> pd.DataFrame:
    """Ambil ulasan dari Google Play Store dengan filter rentang tanggal inklusif.

    Jika max_reviews bernilai None atau 0, seluruh ulasan pada rentang tanggal diambil tanpa batas.
    """
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    if start_date > end_date:
        raise ValueError("Tanggal mulai tidak boleh lebih besar daripada tanggal akhir.")
    if max_reviews is not None and max_reviews < 0:
        raise ValueError("Batas maksimal ulasan tidak boleh negatif.")

    review_limit: int | None = (
        None if (max_reviews is None or max_reviews == 0) else int(max_reviews)
    )

    from google_play_scraper import Sort, reviews

    columns = ["tanggal", "userName", "score", "ulasan", "thumbsUpCount"]
    collected = []
    continuation_token = None

    while True:
        try:
            review_kwargs: dict[str, Any] = {
                "lang": "id",
                "country": "id",
                "sort": Sort.NEWEST,
                "count": 100,
            }
            if continuation_token is not None:
                review_kwargs["continuation_token"] = continuation_token
            batch, continuation_token = reviews(app_id, **review_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Gagal mengambil data dari Google Play Store: {exc}") from exc

        if not batch:
            break

        oldest_in_batch = None
        for item in batch:
            review_dt = item.get("at")
            if isinstance(review_dt, datetime):
                review_d = review_dt.date()
            elif isinstance(review_dt, date):
                review_d = review_dt
            else:
                continue

            if oldest_in_batch is None or review_d < oldest_in_batch:
                oldest_in_batch = review_d

            if start_date <= review_d <= end_date:
                collected.append(
                    {
                        "tanggal": review_dt.strftime("%Y-%m-%d %H:%M")
                        if isinstance(review_dt, datetime)
                        else str(review_d),
                        "userName": item.get("userName", ""),
                        "score": int(item.get("score", 0) or 0),
                        "ulasan": str(item.get("content") or ""),
                        "thumbsUpCount": int(item.get("thumbsUpCount", 0) or 0),
                    }
                )
                if review_limit is not None and len(collected) >= review_limit:
                    break

        if progress_callback:
            if review_limit is not None:
                progress_callback(min(len(collected) / review_limit, 0.95))
            else:
                progress_callback(min(0.95, 1.0 - 1.0 / (1.0 + len(collected) / 500.0)))

        # Berhenti otomatis jika target tercapai, halaman habis, atau ulasan sudah lebih lama dari start_date
        if (
            (review_limit is not None and len(collected) >= review_limit)
            or continuation_token is None
            or (oldest_in_batch is not None and oldest_in_batch < start_date)
        ):
            break

    if progress_callback:
        progress_callback(1.0)

    if not collected:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(collected, columns=columns)


def _normalize_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan struktur kolom, tipe tahun/bulan, dan format tanggal konsisten."""
    if df is None or df.empty:
        return pd.DataFrame(columns=DASHBOARD_COLUMNS)

    work = df.copy()
    numeric_cols = ("tahun", "bulan", "score", "thumbsUpCount", "Keyakinan (%)")
    for col in DASHBOARD_COLUMNS:
        if col not in work.columns:
            work[col] = "" if col not in numeric_cols else 0

    parsed_dt = pd.to_datetime(work["tanggal"], errors="coerce")
    work = work[parsed_dt.notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=DASHBOARD_COLUMNS)

    parsed_dt = parsed_dt.loc[work.index]
    work["tahun"] = parsed_dt.dt.year.astype(int)
    work["bulan"] = parsed_dt.dt.month.astype(int)
    work["score"] = (
        pd.Series(pd.to_numeric(work["score"], errors="coerce"), index=work.index)
        .fillna(0)
        .astype(int)
    )
    work["thumbsUpCount"] = (
        pd.Series(pd.to_numeric(work["thumbsUpCount"], errors="coerce"), index=work.index)
        .fillna(0)
        .astype(int)
    )
    work["Keyakinan (%)"] = (
        pd.Series(pd.to_numeric(work["Keyakinan (%)"], errors="coerce"), index=work.index)
        .fillna(0.0)
        .astype(float)
    )
    return pd.DataFrame(work[DASHBOARD_COLUMNS]).reset_index(drop=True)


def load_dashboard_cache(cache_path: Path = DASHBOARD_CACHE_PATH) -> pd.DataFrame:
    """Muat dataset ulasan teranalisis dari file cache lokal."""
    if not cache_path.exists():
        return pd.DataFrame(columns=DASHBOARD_COLUMNS)
    try:
        df = pd.read_csv(cache_path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=DASHBOARD_COLUMNS)
    return _normalize_dashboard_df(df)


def save_dashboard_cache(df: pd.DataFrame, cache_path: Path = DASHBOARD_CACHE_PATH) -> None:
    """Simpan dataset Dashboard ke file cache lokal."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_dashboard_df(df)
    normalized.to_csv(cache_path, index=False, encoding="utf-8-sig")


def build_dashboard_records_from_playstore(
    scraped_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    """Gabungkan data mentah Play Store dengan output analyze_reviews_batch untuk Dashboard."""
    if scraped_df is None or scraped_df.empty or analysis_df is None or analysis_df.empty:
        return pd.DataFrame(columns=DASHBOARD_COLUMNS)

    combined = pd.DataFrame(
        {
            "tanggal": scraped_df["tanggal"].values,
            "userName": scraped_df["userName"].values,
            "score": scraped_df["score"].values,
            "ulasan": scraped_df["ulasan"].values,
            "thumbsUpCount": scraped_df["thumbsUpCount"].values,
            "Teks Bersih": analysis_df["Teks Bersih"].values,
            "Sentimen": analysis_df["Sentimen"].values,
            "Keyakinan (%)": analysis_df["Keyakinan (%)"].values,
            "Cluster": analysis_df["Cluster"].values,
            "Nama Topik": analysis_df["Nama Topik"].values,
        }
    )
    return _normalize_dashboard_df(combined)


def merge_dashboard_cache(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan data baru ke cache Dashboard tanpa duplikat dan urutkan dari terbaru."""
    norm_existing = _normalize_dashboard_df(existing_df)
    norm_new = _normalize_dashboard_df(new_df)

    if norm_existing.empty:
        merged = norm_new
    elif norm_new.empty:
        merged = norm_existing
    else:
        merged = pd.concat([norm_existing, norm_new], ignore_index=True)

    if merged.empty:
        return merged

    merged = merged.drop_duplicates(subset=["tanggal", "userName", "ulasan"], keep="last")
    return merged.sort_values("tanggal", ascending=False).reset_index(drop=True)


def filter_dashboard_data(
    df: pd.DataFrame,
    selected_year: int,
    selected_month: int | str | None = "Semua Bulan",
    selected_sentiment: str = "Semua",
) -> pd.DataFrame:
    """Filter data Dashboard berdasarkan tahun (wajib), bulan (opsional), dan sentimen."""
    if df is None or df.empty:
        return pd.DataFrame(columns=DASHBOARD_COLUMNS)

    filtered = cast(pd.DataFrame, df[df["tahun"] == int(selected_year)])
    if selected_month is not None and selected_month != "Semua Bulan":
        filtered = cast(pd.DataFrame, filtered[filtered["bulan"] == int(selected_month)])
    if selected_sentiment and selected_sentiment != "Semua":
        filtered = cast(pd.DataFrame, filtered[filtered["Sentimen"] == selected_sentiment])
    return pd.DataFrame(filtered).reset_index(drop=True)


def summarize_dashboard_metrics(df: pd.DataFrame) -> dict:
    """Hitung ringkasan KPI untuk periode Dashboard yang dipilih."""
    if df is None or df.empty:
        return {
            "total": 0,
            "positif_count": 0,
            "negatif_count": 0,
            "positif_pct": 0.0,
            "negatif_pct": 0.0,
            "avg_rating": 0.0,
            "top_topic": "-",
        }

    total = len(df)
    pos_count = int((df["Sentimen"] == "Positif").sum())
    neg_count = int((df["Sentimen"] == "Negatif").sum())
    score_series = pd.Series(pd.to_numeric(df["score"], errors="coerce")) if "score" in df.columns else pd.Series(dtype=float)
    avg_rating = float(score_series.mean()) if not score_series.dropna().empty else 0.0

    valid_topics = cast(
        pd.DataFrame,
        df[
            df["Sentimen"].isin(["Positif", "Negatif"])
            & ~df["Cluster"].isin(["-", "Outlier"])
        ],
    )
    if valid_topics.empty:
        top_topic = "-"
    else:
        top_group = (
            pd.DataFrame({"Jumlah Ulasan": valid_topics.groupby(["Cluster", "Nama Topik"]).size()})
            .reset_index()
            .sort_values("Jumlah Ulasan", ascending=False)
            .iloc[0]
        )
        top_topic = (
            f"{top_group['Cluster']} {top_group['Nama Topik']} "
            f"({int(top_group['Jumlah Ulasan']):,} ulasan)"
        )

    return {
        "total": total,
        "positif_count": pos_count,
        "negatif_count": neg_count,
        "positif_pct": (pos_count / total * 100) if total else 0.0,
        "negatif_pct": (neg_count / total * 100) if total else 0.0,
        "avg_rating": avg_rating,
        "top_topic": top_topic,
    }


def build_monthly_trend_df(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi jumlah ulasan Positif/Negatif per bulan dalam tahun terpilih."""
    columns = ["bulan", "bulan_label", "Sentimen", "Jumlah Ulasan"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    valid = cast(pd.DataFrame, df[df["Sentimen"].isin(["Positif", "Negatif"])])
    if valid.empty:
        return pd.DataFrame(columns=columns)

    agg = (
        pd.DataFrame({"Jumlah Ulasan": valid.groupby(["bulan", "Sentimen"]).size()})
        .reset_index()
        .sort_values("bulan")
    )
    agg["bulan_label"] = [MONTH_SHORT_ID.get(int(m), str(m)) for m in agg["bulan"]]
    return pd.DataFrame(agg[columns]).reset_index(drop=True)


def build_topic_ranking_df(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi ranking topik cluster pada periode Dashboard terpilih."""
    columns = ["Sentimen", "Cluster", "Nama Topik", "Label Topik", "Jumlah Ulasan"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    valid = cast(pd.DataFrame, df[df["Sentimen"].isin(["Positif", "Negatif"])])
    if valid.empty:
        return pd.DataFrame(columns=columns)

    agg = (
        pd.DataFrame(
            {"Jumlah Ulasan": valid.groupby(["Sentimen", "Cluster", "Nama Topik"]).size()}
        )
        .reset_index()
        .sort_values("Jumlah Ulasan", ascending=False)
    )
    agg["Label Topik"] = agg["Cluster"] + " - " + agg["Nama Topik"]
    return pd.DataFrame(agg[columns]).reset_index(drop=True)


def build_topic_monthly_trend_df(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Agregasi tren bulanan untuk top-N topik non-outlier pada tahun terpilih."""
    columns = ["bulan", "bulan_label", "Label Topik", "Jumlah Ulasan"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    valid = cast(
        pd.DataFrame,
        df[
            df["Sentimen"].isin(["Positif", "Negatif"])
            & ~df["Cluster"].isin(["-", "Outlier"])
        ],
    ).copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)

    valid["Label Topik"] = (
        valid["Sentimen"] + " | " + valid["Cluster"] + " - " + valid["Nama Topik"]
    )
    top_labels = pd.Series(valid["Label Topik"]).value_counts().head(top_n).index.tolist()
    filtered = cast(pd.DataFrame, valid[valid["Label Topik"].isin(top_labels)])

    agg = (
        pd.DataFrame({"Jumlah Ulasan": filtered.groupby(["bulan", "Label Topik"]).size()})
        .reset_index()
        .sort_values("bulan")
    )
    agg["bulan_label"] = [MONTH_SHORT_ID.get(int(m), str(m)) for m in agg["bulan"]]
    return pd.DataFrame(agg[columns]).reset_index(drop=True)


def build_batch_distribution_df(result_df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi jumlah ulasan valid per sentimen dan cluster topik dari hasil batch."""
    columns = ["Sentimen", "Cluster", "Nama Topik", "Label Topik", "Jumlah Ulasan"]
    if result_df is None or result_df.empty:
        return pd.DataFrame(columns=columns)

    valid_df = cast(pd.DataFrame, result_df[result_df["Sentimen"].isin(["Positif", "Negatif"])])
    if valid_df.empty:
        return pd.DataFrame(columns=columns)

    dist_df = (
        pd.DataFrame(
            {"Jumlah Ulasan": valid_df.groupby(["Sentimen", "Cluster", "Nama Topik"]).size()}
        )
        .reset_index()
        .sort_values(["Jumlah Ulasan", "Sentimen"], ascending=[False, True])
        .reset_index(drop=True)
    )
    dist_df["Label Topik"] = dist_df["Cluster"] + " - " + dist_df["Nama Topik"]
    return pd.DataFrame(dist_df[columns])


def filter_batch_results(
    result_df: pd.DataFrame,
    sentiment_filter: str = "Semua",
    cluster_filter: str = "Semua",
) -> pd.DataFrame:
    """Filter hasil analisis batch berdasarkan sentimen dan label cluster."""
    if result_df is None or result_df.empty:
        return pd.DataFrame(result_df)

    filtered = result_df
    if sentiment_filter and sentiment_filter != "Semua":
        filtered = cast(pd.DataFrame, filtered[filtered["Sentimen"] == sentiment_filter])
    if cluster_filter and cluster_filter != "Semua":
        filtered = cast(pd.DataFrame, filtered[filtered["Cluster"] == cluster_filter])
    return pd.DataFrame(filtered).reset_index(drop=True)
