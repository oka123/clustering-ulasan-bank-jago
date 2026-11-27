"""
==============================================================================
STREAMLIT APP: Klasifikasi Sentimen + Clustering Topik Ulasan Bank Jago
==============================================================================
Alur:
  1. User menginputkan kalimat ulasan
  2. IndoBERT mengklasifikasikan: Negatif atau Positif
  3. BERTopic meng-cluster ke topik yang sesuai

Jalankan: streamlit run app.py
==============================================================================
"""

import os

# Nonaktifkan inisialisasi TensorFlow yang tidak terpakai (aplikasi memakai PyTorch)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import re
import html
import unicodedata
import pickle
import warnings
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

# ==================== KONFIGURASI ====================
MODEL_DIR = Path("model_indobert_sentimen")
BERTOPIC_NEG_DIR = Path("model_bertopic_negatif")
BERTOPIC_POS_DIR = Path("model_bertopic_positif")
PLAYSTORE_APP_ID = "com.jago.digitalBanking"
MAX_LEN = 128

# Pemetaan nama cluster BERTopic berdasarkan kata kunci dominan di tiap topik
TOPIC_LABELS = {
    "Negatif": {
        0: "Kendala Login & Akses Akun",
        1: "Masalah Saldo & Transaksi Transfer",
        2: "Kendala Update, Download & Performa Lambat",
        3: "Kendala Verifikasi Wajah, KTP & Video Call",
        4: "Kendala Perpindahan Akun Syariah/Konvensional & RDN",
        5: "Keluhan Kualitas Pelayanan",
    },
    "Positif": {
        0: "Kemudahan Transaksi & Transfer Gratis",
        1: "Aplikasi Mudah, Cepat, Aman & Pelayanan Baik",
        2: "Pertanyaan & Bantuan Akun / Login",
        3: "Apresiasi Bonus & Saldo Promo",
        4: "Catatan Minor Koneksi & Update",
        5: "Apresiasi & Saran Fitur Pembayaran QRIS",
    },
}

# ==================== KAMUS SLANG ====================
SLANG_DICT = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "g": "tidak",
    "tdk": "tidak", "nggak": "tidak", "ngga": "tidak", "gx": "tidak",
    "ngg": "tidak", "kgk": "tidak", "kagak": "tidak",
    "bgt": "banget", "bngt": "banget", "bngtt": "banget",
    "yg": "yang", "udh": "sudah", "udah": "sudah", "sdh": "sudah",
    "lg": "lagi", "lgi": "lagi", "lgsg": "langsung",
    "blm": "belum", "blum": "belum", "blom": "belum",
    "tp": "tapi", "tpi": "tapi", "cma": "cuma", "cm": "cuma",
    "krn": "karena", "krna": "karena", "karna": "karena",
    "dgn": "dengan", "dg": "dengan", "dri": "dari",
    "bs": "bisa", "bsa": "bisa", "aja": "saja",
    "dlm": "dalam", "smua": "semua", "smpe": "sampai", "smpai": "sampai",
    "msh": "masih", "msih": "masih", "jg": "juga", "jga": "juga",
    "skrg": "sekarang", "skrang": "sekarang",
    "dpt": "dapat", "dpat": "dapat", "hrs": "harus", "hrus": "harus",
    "knp": "kenapa", "knapa": "kenapa", "gmn": "bagaimana", "gimana": "bagaimana",
    "emg": "memang", "emng": "memang", "mmg": "memang",
    "org": "orang", "orng": "orang",
    "trx": "transaksi", "tx": "transaksi",
    "app": "aplikasi", "apl": "aplikasi", "apk": "aplikasi", "apks": "aplikasi",
    "tf": "transfer", "trf": "transfer",
    "rek": "rekening", "no": "nomor",
    "cs": "customer service", "cust": "customer",
    "min": "admin", "kak": "kakak",
    "bgs": "bagus", "bgus": "bagus",
    "jlk": "jelek", "jlek": "jelek",
    "smoga": "semoga", "moga": "semoga",
    "mksh": "makasih", "mkasih": "makasih", "trmksh": "terima kasih",
    "thx": "terima kasih", "thanks": "terima kasih",
    "pke": "pakai", "pkai": "pakai", "pk": "pakai",
    "bnk": "bank", "mslh": "masalah", "mslah": "masalah",
    "sdg": "sedang",
    "bikin": "membuat", "bkin": "membuat",
    "mantap": "mantap", "mantab": "mantap", "mantabs": "mantap",
    "josss": "bagus sekali", "joss": "bagus sekali",
    "top": "bagus", "topp": "bagus", "toppp": "bagus",
    "okee": "oke", "okeee": "oke", "okey": "oke",
}


def clean_text(text):
    """Pipeline cleaning teks yang sama dengan training."""
    if pd.isna(text) or not str(text).strip():
        return ""
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKD", t)
    t = re.sub(r"(?:https?://|www\.)\S+", " ", t, flags=re.I)
    t = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"(?<!\d)(?:\+?62|0)[\s-]?8\d{1,3}[\s-]?\d{3,4}[\s-]?\d{2,5}(?!\d)", " ", t)
    t = re.sub(r"[^\w\s.,!?;:\-'/]", " ", t)
    t = t.lower().strip()
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    tokens = t.split()
    tokens = [SLANG_DICT.get(tok, tok) for tok in tokens]
    t = " ".join(tokens)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ==================== LOAD MODEL (CACHED) ====================
class DefaultBinaryLabelEncoder:
    """Fallback label encoder untuk model biner (0=Negatif, 1=Positif)."""

    def __init__(self):
        self.classes_ = np.array(["Negatif", "Positif"])

    def inverse_transform(self, indices):
        return np.array([self.classes_[int(i)] for i in indices])


@st.cache_resource
def load_classifier():
    """Load IndoBERT classifier + tokenizer + label encoder."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    le_path = MODEL_DIR / "label_encoder.pkl"
    if le_path.exists():
        with open(le_path, "rb") as f:
            le = pickle.load(f)
    else:
        le = DefaultBinaryLabelEncoder()

    return tokenizer, model, le, device


@st.cache_resource
def load_sentence_embedding_model(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    """Load Sentence Transformer yang sama dengan model saat training BERTopic."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource
def load_bertopic_models(_embedding_model=None):
    """Load BERTopic models untuk negatif dan positif."""
    from bertopic import BERTopic

    models = {}

    if BERTOPIC_NEG_DIR.exists():
        models["Negatif"] = BERTopic.load(
            str(BERTOPIC_NEG_DIR), embedding_model=_embedding_model
        )

    if BERTOPIC_POS_DIR.exists():
        models["Positif"] = BERTopic.load(
            str(BERTOPIC_POS_DIR), embedding_model=_embedding_model
        )

    return models


def _extract_binary_probs(logits, le):
    """Normalisasi logits menjadi probabilitas dan selaraskan dengan fallback biner."""
    # Jika memakai fallback biner namun model memiliki >2 output logits, ambil 2 kelas utama (0=Negatif, 1=Positif)
    if isinstance(le, DefaultBinaryLabelEncoder) and logits.shape[1] > 2:
        logits = logits[:, :2]
    return torch.softmax(logits, dim=1)


def predict_sentiments_batch(texts, tokenizer, model, le, device, batch_size=32):
    """Prediksi sentimen secara batch untuk daftar teks."""
    results = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoding = tokenizer(
            batch_texts,
            add_special_tokens=True,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = _extract_binary_probs(outputs.logits, le)
            pred_labels = torch.argmax(probs, dim=1).cpu().tolist()
            probs_np = probs.cpu().numpy()

        sentiments = le.inverse_transform(pred_labels)
        for idx, pred_idx in enumerate(pred_labels):
            results.append(
                (str(sentiments[idx]), float(probs_np[idx][pred_idx]), probs_np[idx])
            )

    return results


def predict_sentiment(text, tokenizer, model, le, device):
    """Prediksi sentimen untuk satu kalimat menggunakan IndoBERT."""
    return predict_sentiments_batch([text], tokenizer, model, le, device, batch_size=1)[0]


def resolve_topic_name(sentimen, topic_id, bt_model):
    """Ambil nama cluster yang deskriptif, dengan fallback ke nama bawaan BERTopic."""
    custom_name = TOPIC_LABELS.get(sentimen, {}).get(int(topic_id))
    if custom_name:
        return custom_name

    if bt_model is not None:
        topic_info_df = bt_model.get_topic_info()
        match = topic_info_df[topic_info_df["Topic"] == topic_id]
        if not match.empty:
            return match.iloc[0].get("Name", f"Topik {topic_id}")

    return f"Topik {topic_id}"


def build_topic_summary_df(sentimen, bt_model):
    """Susun tabel ringkasan seluruh cluster non-outlier untuk ditampilkan di web."""
    columns = ["Cluster", "Nama Topik", "Jumlah Ulasan", "Kata Kunci"]
    if bt_model is None:
        return pd.DataFrame(columns=columns)

    topic_info_df = bt_model.get_topic_info()
    if topic_info_df is None or topic_info_df.empty:
        return pd.DataFrame(columns=columns)

    non_outliers = topic_info_df[topic_info_df["Topic"] != -1].sort_values("Topic")
    rows = []
    for _, row in non_outliers.iterrows():
        tid = int(row["Topic"])
        topic_words = bt_model.get_topic(tid) or []
        keywords = ", ".join(word for word, _ in topic_words[:8])
        rows.append(
            {
                "Cluster": f"#{tid}",
                "Nama Topik": resolve_topic_name(sentimen, tid, bt_model),
                "Jumlah Ulasan": int(row.get("Count", 0)),
                "Kata Kunci": keywords,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def _encode_for_bertopic(texts, bt_model, embedding_model):
    """Encode teks dan selaraskan dimensi embedding dengan model BERTopic target."""
    embeddings = embedding_model.encode(texts)
    topic_embs = getattr(bt_model, "topic_embeddings_", None)
    if topic_embs is not None and len(topic_embs) > 0:
        expected_dim = np.array(topic_embs).shape[1]
        if embeddings.shape[1] != expected_dim:
            if expected_dim == 384:
                embeddings = load_sentence_embedding_model(
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ).encode(texts)
            elif expected_dim == 768:
                embeddings = load_sentence_embedding_model("indobenchmark/indobert-base-p1").encode(texts)
    return embeddings


def predict_topic(text, sentimen, bertopic_models, embedding_model):
    """Cluster ke topik menggunakan BERTopic (transform)."""
    if sentimen not in bertopic_models:
        return None, None, "Model BERTopic tidak tersedia untuk sentimen ini."

    bt_model = bertopic_models[sentimen]
    embedding = _encode_for_bertopic([text], bt_model, embedding_model)

    topics, probs = bt_model.transform([text], embedding)

    topic_id = topics[0]

    if topic_id == -1:
        topic_name = "Outlier (tidak masuk topik spesifik)"
        topic_info = None
    else:
        topic_name = resolve_topic_name(sentimen, topic_id, bt_model)

        # Ambil kata-kata kunci topik
        topic_words = bt_model.get_topic(topic_id)
        topic_info = topic_words[:10] if topic_words else None

    return topic_id, topic_name, topic_info


def load_uploaded_table(file_obj, filename: str) -> pd.DataFrame:
    """Membaca file CSV atau Excel menjadi DataFrame dengan validasi format dan isi."""
    ext = Path(filename).suffix.lower()
    if ext == ".csv":
        last_error = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                df = pd.read_csv(file_obj, encoding=encoding)
                break
            except UnicodeDecodeError as exc:
                last_error = exc
            except pd.errors.EmptyDataError as exc:
                raise ValueError("File CSV kosong dan tidak memiliki data.") from exc
            except Exception as exc:
                raise ValueError(f"Gagal membaca file CSV: {exc}") from exc
        else:
            raise ValueError(f"Gagal membaca encoding CSV: {last_error}")
    elif ext in (".xlsx", ".xls"):
        try:
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            df = pd.read_excel(file_obj)
        except ImportError as exc:
            raise ValueError(
                "Library pembaca Excel belum terpasang (install `openpyxl` atau `xlrd`)."
            ) from exc
        except Exception as exc:
            raise ValueError(f"Gagal membaca file Excel: {exc}") from exc
    else:
        raise ValueError(
            f"Format file `{ext or 'tanpa ekstensi'}` tidak didukung. Gunakan `.csv`, `.xlsx`, atau `.xls`."
        )

    if df.empty or len(df.columns) == 0:
        raise ValueError("File berhasil dibaca tetapi tidak berisi baris data.")
    return df


def detect_text_column(df: pd.DataFrame):
    """Deteksi kolom default yang kemungkinan besar berisi teks ulasan."""
    if df is None or len(df.columns) == 0:
        return None

    preferred_names = ("ulasan", "review", "komentar", "text", "content", "isi")
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for key in preferred_names:
        if key in lowered:
            return lowered[key]
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if any(key in col_lower for key in preferred_names):
            return col

    text_cols = df.select_dtypes(include=["object", "string"]).columns
    if len(text_cols) > 0:
        return text_cols[0]

    return df.columns[0]


def analyze_reviews_batch(
    texts,
    tokenizer,
    model,
    le,
    device,
    bertopic_models,
    embedding_model,
    batch_size=32,
    progress_callback=None,
):
    """Analisis sentimen dan cluster topik secara batch untuk daftar ulasan dari file."""
    raw_texts = ["" if pd.isna(t) else str(t) for t in texts]
    cleaned_texts = [clean_text(t) for t in raw_texts]

    rows = [
        {
            "Ulasan Asli": raw,
            "Teks Bersih": cleaned,
            "Sentimen": "Tidak Valid",
            "Keyakinan (%)": 0.0,
            "Cluster": "-",
            "Nama Topik": "Teks kosong setelah dibersihkan",
        }
        for raw, cleaned in zip(raw_texts, cleaned_texts)
    ]

    valid_indices = [i for i, text in enumerate(cleaned_texts) if text.strip()]
    if not valid_indices:
        if progress_callback:
            progress_callback(1.0)
        return pd.DataFrame(rows)

    valid_texts = [cleaned_texts[i] for i in valid_indices]
    total_valid = len(valid_texts)

    # 1. Batch inference sentimen IndoBERT (rentang progress 0.0 -> 0.6)
    sentiment_results = []
    for start in range(0, total_valid, batch_size):
        chunk = valid_texts[start : start + batch_size]
        sentiment_results.extend(
            predict_sentiments_batch(chunk, tokenizer, model, le, device, batch_size=batch_size)
        )
        if progress_callback:
            progress_callback(0.6 * min(start + len(chunk), total_valid) / total_valid)

    grouped_by_sentiment = {}
    for pos, row_idx in enumerate(valid_indices):
        sentimen, confidence, _ = sentiment_results[pos]
        rows[row_idx]["Sentimen"] = sentimen
        rows[row_idx]["Keyakinan (%)"] = round(confidence * 100, 1)
        rows[row_idx]["Cluster"] = "-"
        rows[row_idx]["Nama Topik"] = (
            "Model BERTopic tidak tersedia"
            if not bertopic_models or sentimen not in bertopic_models
            else "-"
        )
        grouped_by_sentiment.setdefault(sentimen, []).append(row_idx)

    # 2. Batch clustering BERTopic per sentimen (rentang progress 0.6 -> 1.0)
    if bertopic_models and embedding_model is not None:
        processed_topics = 0
        for sentimen, row_indices in grouped_by_sentiment.items():
            if sentimen not in bertopic_models:
                processed_topics += len(row_indices)
                continue

            bt_model = bertopic_models[sentimen]
            group_texts = [cleaned_texts[idx] for idx in row_indices]
            embeddings = _encode_for_bertopic(group_texts, bt_model, embedding_model)
            topics, _ = bt_model.transform(group_texts, embeddings)

            for row_idx, topic_id in zip(row_indices, topics):
                tid = int(topic_id)
                if tid == -1:
                    rows[row_idx]["Cluster"] = "Outlier"
                    rows[row_idx]["Nama Topik"] = "Outlier (tidak masuk topik spesifik)"
                else:
                    rows[row_idx]["Cluster"] = f"#{tid}"
                    rows[row_idx]["Nama Topik"] = resolve_topic_name(sentimen, tid, bt_model)

            processed_topics += len(row_indices)
            if progress_callback:
                progress_callback(0.6 + 0.4 * (processed_topics / total_valid))

    if progress_callback:
        progress_callback(1.0)

    return pd.DataFrame(rows)


def scrape_playstore_reviews(
    start_date,
    end_date,
    max_reviews: int = 100,
    app_id: str = PLAYSTORE_APP_ID,
    progress_callback=None,
) -> pd.DataFrame:
    """Ambil ulasan dari Google Play Store dengan filter rentang tanggal inklusif."""
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    if start_date > end_date:
        raise ValueError("Tanggal mulai tidak boleh lebih besar daripada tanggal akhir.")
    if max_reviews <= 0:
        raise ValueError("Batas maksimal ulasan harus lebih dari 0.")

    from google_play_scraper import Sort, reviews

    columns = ["tanggal", "userName", "score", "ulasan", "thumbsUpCount"]
    collected = []
    continuation_token = None
    max_pages = 50  # Pengaman agar tidak looping tanpa batas pada rentang tanggal lampau

    for _ in range(max_pages):
        try:
            batch, continuation_token = reviews(
                app_id,
                lang="id",
                country="id",
                sort=Sort.NEWEST,
                count=100,
                continuation_token=continuation_token,
            )
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
                if len(collected) >= max_reviews:
                    break

        if progress_callback:
            progress_callback(min(len(collected) / max_reviews, 0.95))

        # Berhenti otomatis jika target tercapai, halaman habis, atau ulasan sudah lebih lama dari start_date
        if (
            len(collected) >= max_reviews
            or continuation_token is None
            or (oldest_in_batch is not None and oldest_in_batch < start_date)
        ):
            break

    if progress_callback:
        progress_callback(1.0)

    if not collected:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(collected, columns=columns)


def _render_batch_analysis_results(result_df: pd.DataFrame, out_stem: str, key_prefix: str) -> None:
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


# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Analisis Ulasan Bank Jago",
        page_icon="🏦",
        layout="centered",
    )

    # Header minimalis & padat
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <h2 style="margin-bottom: 0.2rem;">🏦 Analisis Ulasan Bank Jago</h2>
        <p style="font-size: 0.95rem; color: #666; margin-top: 0;">
          Clustering Topik (BERTopic)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar ringkas
    with st.sidebar:
        st.markdown("### ℹ️ Info Model")
        st.caption("• **Sentimen**: IndoBERT (`indobert-base-p1`)\n• **Topik**: BERTopic (`paraphrase-multilingual-MiniLM-L12-v2`)")
        st.markdown("---")
        st.caption("Kelompok 4 - Kelas A")

    # Cek ketersediaan model
    model_exists = MODEL_DIR.exists() and (MODEL_DIR / "config.json").exists()
    
    if not model_exists:
        st.error(f"⚠️ Model IndoBERT belum ditemukan di `{MODEL_DIR}/`!")
        st.stop()

    # Load models
    tokenizer, classifier, le, device = load_classifier()
    bertopic_available = BERTOPIC_NEG_DIR.exists() or BERTOPIC_POS_DIR.exists()
    bertopic_models = {}
    embedding_model = None
    
    if bertopic_available:
        embedding_model = load_sentence_embedding_model()
        bertopic_models = load_bertopic_models(embedding_model)

    # Tab input: satuan, upload file CSV/Excel, dan ambil data Play Store
    tab_text, tab_file, tab_playstore = st.tabs(
        ["✍️ Input Teks", "📁 Upload CSV / Excel", "📱 Ambil Data Play Store"]
    )

    with tab_text:
        user_input = st.text_area(
            "Tulis ulasan Bank Jago:",
            height=110,
            placeholder="Ketik atau tempel kalimat ulasan di sini...",
        )

        analyze_btn = st.button("🔍 Analisis Ulasan", type="primary", use_container_width=True)

        # Proses analisis satuan
        if analyze_btn:
            if not user_input.strip():
                st.warning("⚠️ Mohon masukkan teks ulasan terlebih dahulu.")
            else:
                cleaned = clean_text(user_input)

                if not cleaned.strip():
                    st.warning("⚠️ Teks ulasan kosong setelah dibersihkan (hanya emoji/karakter khusus).")
                else:
                    # 1. Prediksi Sentimen
                    sentimen, confidence, probs = predict_sentiment(
                        cleaned, tokenizer, classifier, le, device
                    )

                    st.markdown("---")
                    st.markdown("### 📊 Hasil Analisis")

                    # Kartu Sentimen & Keyakinan
                    col_sent, col_conf = st.columns(2)

                    with col_sent:
                        if sentimen == "Positif":
                            st.success(f"**Sentimen:** ✅ **{sentimen}**")
                        else:
                            st.error(f"**Sentimen:** ❌ **{sentimen}**")

                    with col_conf:
                        st.metric("Tingkat Keyakinan", f"{confidence * 100:.1f}%")

                    # Probabilitas bar
                    for i, cls in enumerate(le.classes_):
                        prob_val = float(probs[i])
                        color_dot = "🟢" if cls == "Positif" else "🔴"
                        st.caption(f"{color_dot} Probabilitas {cls}: **{prob_val * 100:.1f}%**")
                        st.progress(prob_val)

                    # 2. Clustering Topik
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
                            st.warning("Ulasan ini dikategorikan sebagai *outlier* (tidak masuk kelompok topik dominan).")
                        else:
                            st.info(str(topic_info))

                    # Detail Preprocessing
                    with st.expander("🔍 Lihat Hasil Preprocessing Teks"):
                        st.write("**Teks Bersih:**", cleaned)
                        st.caption(f"Jumlah kata: {len(cleaned.split())} kata")

    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload file berisi ulasan (CSV atau Excel):",
            type=["csv", "xlsx", "xls"],
        )

        if uploaded_file is not None:
            try:
                uploaded_df = load_uploaded_table(uploaded_file, uploaded_file.name)
            except ValueError as err:
                st.error(f"⚠️ {err}")
            else:
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
                        progress_bar.progress(min(max(val, 0.0), 1.0), text=f"Memproses ulasan... ({pct}%)")

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
                    _render_batch_analysis_results(
                        st.session_state["batch_result_df"],
                        Path(uploaded_file.name).stem,
                        "upload_batch",
                    )

    with tab_playstore:
        st.caption(
            f"Ambil ulasan aplikasi Bank Jago (`{PLAYSTORE_APP_ID}`) dari Google Play Store berdasarkan filter tanggal sebagai input analisis."
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

        if "ps_scraped_df" in st.session_state:
            scraped_df = st.session_state["ps_scraped_df"]
            r_start, r_end = st.session_state.get(
                "ps_scraped_range", (str(start_date), str(end_date))
            )

            if scraped_df.empty:
                st.info(
                    f"Tidak ada ulasan ditemukan pada rentang tanggal **{r_start}** sampai **{r_end}**. Coba perluas rentang tanggal."
                )
            else:
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
                    ps_progress.empty()
                    st.session_state["ps_result_df"] = ps_result_df
                    st.session_state["ps_result_range"] = (r_start, r_end)

                if "ps_result_df" in st.session_state:
                    res_start, res_end = st.session_state.get(
                        "ps_result_range", (r_start, r_end)
                    )
                    _render_batch_analysis_results(
                        st.session_state["ps_result_df"],
                        f"playstore_{res_start}_{res_end}",
                        "ps_batch",
                    )

    # Tampilkan daftar seluruh cluster yang sudah diberi nama di halaman web
    if bertopic_available and bertopic_models:
        st.markdown("---")
        st.markdown("### 📋 Daftar Cluster Topik")
        available_sentiments = [s for s in ("Negatif", "Positif") if s in bertopic_models]
        if available_sentiments:
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
                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True,
                    )

    # Footer minimalis
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8rem;'>"
        "Tugas Akhir — Analisis Ulasan Bank Jago | Kelompok 4"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
