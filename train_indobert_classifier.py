"""
==============================================================================
TRAINING MODEL KLASIFIKASI SENTIMEN MENGGUNAKAN IndoBERT
==============================================================================
Script ini melatih model IndoBERT (indobenchmark/indobert-base-p1) untuk 
klasifikasi sentimen ulasan Bank Jago (Negatif vs Positif).

Sentimen berdasarkan rating:
  - Rating 1-2 → Negatif (label 0)
  - Rating 4-5 → Positif (label 1)
  - Rating 3   → Dibuang (ambigu untuk klasifikasi biner)

Output:
  - model_indobert_sentimen/  → model + tokenizer tersimpan
  - label_encoder.pkl         → LabelEncoder mapping

Jalankan di Google Colab (GPU T4) atau mesin dengan GPU.
==============================================================================
"""

import os
import re
import html
import unicodedata
import warnings
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# ==================== KONFIGURASI ====================
RANDOM_STATE = 42
MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
TEST_SIZE = 0.2

# Deteksi device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ==================== LOKASI FILE ====================
# Sesuaikan path ini dengan lokasi data kamu
import sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE_DIR = Path("/content/drive/MyDrive/TRANSFER FILE")
else:
    BASE_DIR = Path(".")

DATA_FILE = "ulasan_bank_jago_all.csv"
CANDIDATES = [BASE_DIR / DATA_FILE, Path(DATA_FILE)]
DATA_PATH = next((p for p in CANDIDATES if p.exists()), None)

if DATA_PATH is None:
    print(f"[ERROR] File '{DATA_FILE}' tidak ditemukan.")
    print(f"Lokasi yang dicek: {[str(p) for p in CANDIDATES]}")
    print("Letakkan file CSV di folder yang sama dengan script ini.")
    sys.exit(1)

OUTPUT_DIR = Path("model_indobert_sentimen")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Data    : {DATA_PATH}")
print(f"Output  : {OUTPUT_DIR}")


# ==================== 1. LOAD & PREP DATA ====================
print("\n" + "=" * 60)
print("1. Loading dan Menyiapkan Data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"Total data mentah: {len(df):,}")

# Filter hanya rating 1-2 (Negatif) dan 4-5 (Positif)
df = df[df["score"].isin([1, 2, 4, 5])].copy()
df["sentimen"] = df["score"].apply(lambda x: "Negatif" if x <= 2 else "Positif")
print(f"Data setelah filter (tanpa rating 3): {len(df):,}")
print(f"\nDistribusi sentimen:")
print(df["sentimen"].value_counts().to_string())


# ==================== 2. CLEANING TEKS ====================
print("\n" + "=" * 60)
print("2. Cleaning Teks")
print("=" * 60)

# Kamus slang sederhana (sama seperti notebook utama)
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
    "sdg": "sedang", "lg": "lagi",
    "bikin": "membuat", "bkin": "membuat",
    "mantap": "mantap", "mantab": "mantap", "mantabs": "mantap",
    "josss": "bagus sekali", "joss": "bagus sekali",
    "top": "bagus", "topp": "bagus", "toppp": "bagus",
    "okee": "oke", "okeee": "oke", "okey": "oke",
}

def clean_text(text):
    """Cleaning pipeline mirip notebook utama (text_light)."""
    if pd.isna(text) or not str(text).strip():
        return ""
    
    t = html.unescape(str(text))
    t = unicodedata.normalize("NFKD", t)
    
    # Hapus URL, email, mention, nomor HP
    t = re.sub(r"(?:https?://|www\.)\S+", " ", t, flags=re.I)
    t = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"(?<!\d)(?:\+?62|0)[\s-]?8\d{1,3}[\s-]?\d{3,4}[\s-]?\d{2,5}(?!\d)", " ", t)
    
    # Hapus emoji & karakter non-latin
    t = re.sub(r"[^\w\s.,!?;:\-'/]", " ", t)
    
    # Lowercase
    t = t.lower().strip()
    
    # Normalisasi huruf berulang: bagusss -> bagus, mantapppp -> mantap
    t = re.sub(r"(.)\1{2,}", r"\1", t)
    
    # Normalisasi slang
    tokens = t.split()
    tokens = [SLANG_DICT.get(tok, tok) for tok in tokens]
    t = " ".join(tokens)
    
    # Hapus angka dan tanda baca berlebih, biarkan huruf
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    return t


tqdm.pandas(desc="Cleaning teks")
df["text_clean"] = df["content"].progress_apply(clean_text)

# Buang teks kosong
n_before = len(df)
df = df[df["text_clean"].str.strip() != ""].copy()
print(f"Teks kosong dibuang: {n_before - len(df):,}")
print(f"Data akhir: {len(df):,}")

# Balance check
print(f"\nDistribusi akhir:")
print(df["sentimen"].value_counts().to_string())


# ==================== 3. ENCODING LABEL ====================
print("\n" + "=" * 60)
print("3. Encoding Label & Split Data")
print("=" * 60)

le = LabelEncoder()
df["label"] = le.fit_transform(df["sentimen"])
print(f"Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Simpan label encoder
with open(OUTPUT_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df["text_clean"].values,
    df["label"].values,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["label"].values,
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")


# ==================== 4. DATASET & DATALOADER ====================
print("\n" + "=" * 60)
print("4. Menyiapkan Tokenizer & DataLoader")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SentimenDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


train_dataset = SentimenDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset = SentimenDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Tokenizer: {MODEL_NAME}")
print(f"Max length: {MAX_LEN}")
print(f"Batch size: {BATCH_SIZE}")


# ==================== 5. MODEL ====================
print("\n" + "=" * 60)
print("5. Loading Model IndoBERT")
print("=" * 60)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(le.classes_),
)
model = model.to(device)
print(f"Model loaded: {MODEL_NAME}")
print(f"Num labels: {len(le.classes_)} ({list(le.classes_)})")


# ==================== 6. TRAINING ====================
print("\n" + "=" * 60)
print("6. Training Model")
print("=" * 60)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

best_acc = 0.0

for epoch in range(EPOCHS):
    # --- Training ---
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{total_loss / (pbar.n + 1):.4f}",
            acc=f"{correct / total:.4f}",
        )

    train_acc = correct / total
    train_loss = total_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]  "):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total

    print(f"\n  Epoch {epoch + 1}: Train Loss={train_loss:.4f}, "
          f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    # Simpan model terbaik
    if val_acc > best_acc:
        best_acc = val_acc
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"  ✓ Model terbaik disimpan (Val Acc={val_acc:.4f})")


# ==================== 7. EVALUASI AKHIR ====================
print("\n" + "=" * 60)
print("7. Evaluasi Akhir (Model Terbaik)")
print("=" * 60)

# Load model terbaik
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluasi final"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

print("Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))

print(f"\n{'=' * 60}")
print(f"TRAINING SELESAI!")
print(f"Model tersimpan di: {OUTPUT_DIR.resolve()}")
print(f"Best Validation Accuracy: {best_acc:.4f}")
print(f"{'=' * 60}")
