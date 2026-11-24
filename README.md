# Analisis Ulasan Bank Jago — Klasifikasi & Clustering

Aplikasi Streamlit untuk menganalisis ulasan Bank Jago secara real-time:
1. **Klasifikasi Sentimen** menggunakan IndoBERT (`indobert-base-p1`)
2. **Clustering Topik** menggunakan BERTopic

## Struktur Folder

```
streamlit_ulasan_bankjago/
├── app.py                          # Streamlit web app
├── train_indobert_classifier.py    # Script training IndoBERT classifier
├── save_bertopic_models.py         # Script simpan BERTopic dari notebook
├── requirements.txt                # Dependencies
├── README.md                       # Dokumentasi ini
│
├── model_indobert_sentimen/        # (output training) Model IndoBERT
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── label_encoder.pkl
│
├── model_bertopic_negatif/         # (dari notebook) BERTopic Negatif
│   └── ...
└── model_bertopic_positif/         # (dari notebook) BERTopic Positif
    └── ...
```

## Langkah Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training Model IndoBERT (di Google Colab)
```bash
# Pastikan file ulasan_bank_jago_all.csv ada di folder yang sama
python train_indobert_classifier.py
```
> **Catatan:** Disarankan jalankan di Google Colab (GPU T4) untuk mempercepat training.

### 3. Simpan Model BERTopic (di Notebook Colab)
Tambahkan code dari `save_bertopic_models.py` di akhir notebook **Tugas Akhir 5.ipynb**, lalu download hasilnya.

### 4. Jalankan Streamlit
```bash
streamlit run app.py
```

## Alur Aplikasi

```
User Input (Ulasan)
       │
       ▼
  [Text Cleaning]
  - Lowercase, hapus URL/email/emoji
  - Normalisasi slang → kata baku
  - Hapus angka & tanda baca
       │
       ▼
  [IndoBERT Classifier]
  - Tokenisasi (max 128 token)
  - Forward pass → logits
  - Softmax → Probabilitas
  - Output: Positif / Negatif + confidence
       │
       ▼
  [BERTopic Transform]
  - Sentence embedding (IndoBERT)
  - Transform ke model BERTopic yang sesuai sentimen
  - Output: Topik + kata kunci
       │
       ▼
  [Tampilkan Hasil]
  - Sentimen (badge + confidence bar)
  - Topik cluster + kata kunci
```

## Credits
Kelompok 4 — Kelas A
- Rahmad Dhani (2408561064)
- Kenneth Mighty Anggracio (2408561065)
- I Putu Gede Oka Adyuta (2408561067)
- I Dewa Made Ary Dwisaputra (2408561070)
