"""
==============================================================================
SAVE MODEL BERTOPIC DARI NOTEBOOK
==============================================================================
Tambahkan cell ini di akhir notebook "Tugas Akhir 5.ipynb" setelah modeling
BERTopic selesai, untuk menyimpan model agar bisa dipakai di Streamlit app.

PENTING: Jalankan SETELAH cell model negatif dan positif sudah jalan.
==============================================================================
"""

# Pastikan variabel model sudah ada dari notebook utama
# base_model_neg = model BERTopic negatif
# base_model_pos = model BERTopic positif

from pathlib import Path

SAVE_DIR = Path("/content/drive/MyDrive/TRANSFER FILE/streamlit_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Simpan model BERTopic Negatif
print("Menyimpan model BERTopic Negatif...")
base_model_neg.save(
    str(SAVE_DIR / "model_bertopic_negatif"),
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model=False,  # embedding model diload terpisah di Streamlit
)
print("✓ Tersimpan di:", SAVE_DIR / "model_bertopic_negatif")

# Simpan model BERTopic Positif
print("Menyimpan model BERTopic Positif...")
base_model_pos.save(
    str(SAVE_DIR / "model_bertopic_positif"),
    serialization="safetensors",
    save_ctfidf=True,
    save_embedding_model=False,
)
print("✓ Tersimpan di:", SAVE_DIR / "model_bertopic_positif")

# Simpan juga embedding model reference
print("\nSemua model tersimpan!")
print("Copy folder 'model_bertopic_negatif' dan 'model_bertopic_positif'")
print("ke folder yang sama dengan app.py untuk Streamlit.")
