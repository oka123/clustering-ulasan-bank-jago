"""Fungsi pembersihan teks dan pembacaan file CSV/Excel."""

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import SLANG_DICT


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
