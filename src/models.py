"""Loader model IndoBERT & BERTopic serta fungsi inferensi sentimen dan topik."""

import pickle

import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.config import (
    BERTOPIC_NEG_DIR,
    BERTOPIC_POS_DIR,
    MAX_LEN,
    MODEL_DIR,
    TOPIC_LABELS,
)
from src.preprocessing import clean_text


class DefaultBinaryLabelEncoder:
    """Fallback label encoder untuk model biner (0=Negatif, 1=Positif)."""

    def __init__(self):
        self.classes_ = np.array(["Negatif", "Positif"])

    def inverse_transform(self, indices):
        return np.array([self.classes_[int(i)] for i in indices])


@st.cache_resource
def load_classifier():
    """Load IndoBERT classifier + tokenizer + label encoder."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
                embeddings = load_sentence_embedding_model(
                    "indobenchmark/indobert-base-p1"
                ).encode(texts)
    return embeddings


def predict_topic(text, sentimen, bertopic_models, embedding_model):
    """Cluster ke topik menggunakan BERTopic (transform)."""
    if sentimen not in bertopic_models:
        return None, None, "Model BERTopic tidak tersedia untuk sentimen ini."

    bt_model = bertopic_models[sentimen]
    embedding = _encode_for_bertopic([text], bt_model, embedding_model)
    topics, _ = bt_model.transform([text], embedding)
    topic_id = topics[0]

    if topic_id == -1:
        topic_name = "Outlier (tidak masuk topik spesifik)"
        topic_info = None
    else:
        topic_name = resolve_topic_name(sentimen, topic_id, bt_model)
        topic_words = bt_model.get_topic(topic_id)
        topic_info = topic_words[:10] if topic_words else None

    return topic_id, topic_name, topic_info


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
    """Analisis sentimen dan cluster topik secara batch untuk daftar ulasan."""
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
            predict_sentiments_batch(
                chunk, tokenizer, model, le, device, batch_size=batch_size
            )
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
                    rows[row_idx]["Nama Topik"] = resolve_topic_name(
                        sentimen, tid, bt_model
                    )

            processed_topics += len(row_indices)
            if progress_callback:
                progress_callback(0.6 + 0.4 * (processed_topics / total_valid))

    if progress_callback:
        progress_callback(1.0)

    return pd.DataFrame(rows)
