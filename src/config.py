"""Konfigurasi path, kamus slang, label topik, dan konstanta Dashboard."""

from pathlib import Path

MODEL_DIR = Path("model_indobert_sentimen")
BERTOPIC_NEG_DIR = Path("model_bertopic_negatif")
BERTOPIC_POS_DIR = Path("model_bertopic_positif")
PLAYSTORE_APP_ID = "com.jago.digitalBanking"
DASHBOARD_CACHE_PATH = Path("data/playstore_reviews_cache.csv")
MAX_LEN = 128

MONTH_NAMES_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}
MONTH_SHORT_ID = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu",
    9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
}
DASHBOARD_COLUMNS = [
    "tanggal", "tahun", "bulan", "userName", "score",
    "ulasan", "thumbsUpCount", "Teks Bersih", "Sentimen",
    "Keyakinan (%)", "Cluster", "Nama Topik",
]

# Contoh ulasan cepat untuk pengujian di tab Analisis Ulasan
SAMPLE_REVIEWS = [
    (
        "Contoh Positif: Transfer",
        "Aplikasinya sangat membantu, transfer gratis ke mana saja dan cepat masuk.",
    ),
    (
        "Contoh Positif: QRIS",
        "Fitur QRIS nya mantap, bayar di kasir tinggal scan langsung beres.",
    ),
    (
        "Contoh Negatif: Login",
        "Susah banget login di hp baru, verifikasi email tidak masuk padahal data sudah benar.",
    ),
]

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
