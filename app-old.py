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
import re
import html
import unicodedata
import pickle
import warnings
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
MAX_LEN = 128

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
@st.cache_resource
def load_classifier():
    """Load IndoBERT classifier + tokenizer + label encoder."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return tokenizer, model, le, device


@st.cache_resource
def load_bertopic_models():
    """Load BERTopic models untuk negatif dan positif."""
    from bertopic import BERTopic

    models = {}
    
    if BERTOPIC_NEG_DIR.exists():
        models["Negatif"] = BERTopic.load(str(BERTOPIC_NEG_DIR))
    
    if BERTOPIC_POS_DIR.exists():
        models["Positif"] = BERTopic.load(str(BERTOPIC_POS_DIR))

    return models


@st.cache_resource
def load_embedding_model():
    """Load Sentence Transformer untuk BERTopic transform."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("indobenchmark/indobert-base-p1")


def predict_sentiment(text, tokenizer, model, le, device):
    """Prediksi sentimen menggunakan IndoBERT."""
    encoding = tokenizer.encode_plus(
        text,
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
        probs = torch.softmax(outputs.logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    sentimen = le.inverse_transform([pred_label])[0]
    return sentimen, confidence, probs[0].cpu().numpy()


def predict_topic(text, sentimen, bertopic_models, embedding_model):
    """Cluster ke topik menggunakan BERTopic (transform)."""
    if sentimen not in bertopic_models:
        return None, None, "Model BERTopic tidak tersedia untuk sentimen ini."

    bt_model = bertopic_models[sentimen]
    embedding = embedding_model.encode([text])
    topics, probs = bt_model.transform([text], embedding)
    
    topic_id = topics[0]
    
    if topic_id == -1:
        topic_name = "Outlier (tidak masuk topik spesifik)"
        topic_info = None
    else:
        topic_info_df = bt_model.get_topic_info()
        match = topic_info_df[topic_info_df["Topic"] == topic_id]
        if not match.empty:
            topic_name = match.iloc[0].get("Name", f"Topik {topic_id}")
        else:
            topic_name = f"Topik {topic_id}"
        
        # Ambil kata-kata kunci topik
        topic_words = bt_model.get_topic(topic_id)
        topic_info = topic_words[:10] if topic_words else None
    
    return topic_id, topic_name, topic_info


# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Analisis Ulasan Bank Jago",
        page_icon="🏦",
        layout="wide",
    )

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>🏦 Analisis Ulasan Bank Jago</h1>
        <p style="font-size: 1.1rem; color: #666;">
            Klasifikasi Sentimen (IndoBERT) + Clustering Topik (BERTopic)
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Tentang Aplikasi")
        st.markdown("""
        **Alur Analisis:**
        1. ✍️ User menginputkan ulasan
        2. 🤖 IndoBERT mengklasifikasi sentimen
        3. 📊 BERTopic meng-cluster ke topik
        
        ---
        
        **Model:**
        - **Klasifikasi**: IndoBERT (`indobert-base-p1`)
        - **Clustering**: BERTopic + HDBSCAN
        
        ---
        
        **Kelompok 4 - Kelas A**
        - Rahmad Dhani
        - Kenneth Mighty Anggracio
        - I Putu Gede Oka Adyuta
        - I Dewa Made Ary Dwisaputra
        """)

    # Cek apakah model tersedia
    model_exists = MODEL_DIR.exists() and (MODEL_DIR / "label_encoder.pkl").exists()
    
    if not model_exists:
        st.error(f"""
        ⚠️ **Model belum ditemukan!**
        
        Jalankan dulu script training:
        ```
        python train_indobert_classifier.py
        ```
        
        Model akan disimpan di folder `{MODEL_DIR}/`
        """)
        st.stop()

    # Load models
    with st.spinner("🔄 Memuat model IndoBERT..."):
        tokenizer, classifier, le, device = load_classifier()

    bertopic_available = BERTOPIC_NEG_DIR.exists() or BERTOPIC_POS_DIR.exists()
    bertopic_models = {}
    embedding_model = None
    
    if bertopic_available:
        with st.spinner("🔄 Memuat model BERTopic..."):
            bertopic_models = load_bertopic_models()
            embedding_model = load_embedding_model()

    # Input section
    st.subheader("✍️ Masukkan Ulasan")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.text_area(
            "Tulis ulasan tentang Bank Jago:",
            height=120,
            placeholder="Contoh: Aplikasinya sering error pas mau transfer, sangat mengecewakan...",
        )
    with col2:
        st.markdown("**Contoh ulasan:**")
        examples = [
            "Aplikasi bagus, transfer cepat dan gratis",
            "Error terus, login gak bisa, bikin kesel",
            "Fitur pocket sangat membantu menabung",
            "CS lambat banget, sudah komplain tidak ditanggapi",
            "Mantap, desain simpel dan mudah dipakai",
        ]
        for ex in examples:
            if st.button(f"📝 {ex[:40]}...", key=ex, use_container_width=True):
                st.session_state["example_text"] = ex

    # Gunakan contoh jika dipilih
    if "example_text" in st.session_state:
        user_input = st.session_state.pop("example_text")
        st.rerun()

    # Proses analisis
    if st.button("🔍 Analisis Ulasan", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("⚠️ Mohon masukkan ulasan terlebih dahulu.")
        else:
            # Cleaning
            cleaned = clean_text(user_input)
            
            if not cleaned.strip():
                st.warning("⚠️ Teks ulasan kosong setelah cleaning (mungkin hanya emoji/simbol).")
            else:
                st.markdown("---")
                
                # === STEP 1: KLASIFIKASI SENTIMEN ===
                st.subheader("📊 Hasil Analisis")
                
                with st.spinner("🤖 Mengklasifikasi sentimen..."):
                    sentimen, confidence, probs = predict_sentiment(
                        cleaned, tokenizer, classifier, le, device
                    )
                
                col_sent, col_conf = st.columns(2)
                
                with col_sent:
                    if sentimen == "Positif":
                        st.success(f"### ✅ Sentimen: **{sentimen}**")
                    else:
                        st.error(f"### ❌ Sentimen: **{sentimen}**")
                
                with col_conf:
                    st.metric("Confidence", f"{confidence * 100:.1f}%")
                    
                    # Progress bar untuk probabilitas tiap kelas
                    for i, cls in enumerate(le.classes_):
                        prob_val = float(probs[i])
                        color = "🟢" if cls == "Positif" else "🔴"
                        st.caption(f"{color} {cls}: {prob_val * 100:.1f}%")
                        st.progress(prob_val)

                # === STEP 2: CLUSTERING TOPIK ===
                if bertopic_available and bertopic_models:
                    st.markdown("---")
                    
                    with st.spinner("📊 Menganalisis topik..."):
                        topic_id, topic_name, topic_info = predict_topic(
                            cleaned, sentimen, bertopic_models, embedding_model
                        )
                    
                    st.subheader("🏷️ Clustering Topik")
                    
                    if topic_id is not None and topic_id != -1:
                        st.info(f"**Topik #{topic_id}**: {topic_name}")
                        
                        if topic_info and isinstance(topic_info, list):
                            st.markdown("**Kata kunci topik:**")
                            keywords = " | ".join(
                                [f"`{word}` ({score:.3f})" for word, score in topic_info]
                            )
                            st.markdown(keywords)
                    elif topic_id == -1:
                        st.warning("Ulasan ini tidak masuk ke topik spesifik (outlier). "
                                   "Mungkin terlalu pendek atau unik.")
                    else:
                        st.info("ℹ️ " + str(topic_info))
                else:
                    st.markdown("---")
                    st.info("""
                    ℹ️ **Model BERTopic belum tersedia.**
                    
                    Untuk mengaktifkan fitur clustering topik, simpan model BERTopic dari 
                    notebook utama menggunakan:
                    ```python
                    # Di notebook Tugas Akhir 5.ipynb, setelah modeling:
                    base_model_neg.save("model_bertopic_negatif")
                    base_model_pos.save("model_bertopic_positif")
                    ```
                    Lalu copy folder model ke direktori yang sama dengan `app.py`.
                    """)
                
                # Detail cleaning
                with st.expander("🔍 Detail Preprocessing"):
                    st.markdown(f"**Teks asli:** {user_input}")
                    st.markdown(f"**Teks setelah cleaning:** {cleaned}")
                    st.markdown(f"**Jumlah kata:** {len(cleaned.split())}")

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #aaa; font-size: 0.85rem;'>"
        "Tugas Akhir — Clustering Ulasan Aplikasi Bank Jago | Kelas A - Kelompok 4"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
