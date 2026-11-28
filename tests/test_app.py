"""Unit test untuk fungsi preprocessing dan prediksi topik di app.py."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

# Pastikan root project ada di sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402


class TestApp(unittest.TestCase):
    """Kumpulan unit test untuk logika bisnis app.py."""

    def test_environment_tf_disabled(self):
        """Memastikan env var penonaktifan TensorFlow terset sebelum load model."""
        self.assertEqual(os.environ.get("USE_TF"), "0")
        self.assertEqual(os.environ.get("TF_ENABLE_ONEDNN_OPTS"), "0")
        self.assertEqual(os.environ.get("TF_CPP_MIN_LOG_LEVEL"), "3")

    def test_clean_text_positive(self):
        """Positive case: normalisasi slang, hapus URL, email, dan tanda baca."""
        raw = "Apk bank jago gk bisa tf cek https://jago.com atau cs@jago.com!"
        cleaned = app.clean_text(raw)
        self.assertIn("aplikasi", cleaned)
        self.assertIn("tidak", cleaned)
        self.assertIn("transfer", cleaned)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("cs@jago.com", cleaned)

    def test_clean_text_negative_empty_and_nan(self):
        """Negative case: input kosong, spasi, dan NaN mengembalikan string kosong."""
        self.assertEqual(app.clean_text(""), "")
        self.assertEqual(app.clean_text("   "), "")
        self.assertEqual(app.clean_text(np.nan), "")
        self.assertEqual(app.clean_text("😊🔥🎉"), "")

    def test_predict_topic_missing_sentiment(self):
        """Negative case: sentimen tidak ada di dictionary model BERTopic."""
        mock_emb = MagicMock()
        topic_id, topic_name, topic_info = app.predict_topic(
            "aplikasi bagus", "Netral", {}, mock_emb
        )
        self.assertIsNone(topic_id)
        self.assertIsNone(topic_name)
        self.assertIsInstance(topic_info, str)
        self.assertIn("tidak tersedia", str(topic_info))

    def test_predict_topic_outlier(self):
        """Negative/Edge case: BERTopic mengembalikan topic_id -1 (outlier)."""
        mock_emb = MagicMock()
        mock_emb.encode.return_value = np.zeros((1, 384))

        mock_bt = MagicMock()
        mock_bt.topic_embeddings_ = np.zeros((3, 384))
        mock_bt.transform.return_value = ([-1], [0.1])

        topic_id, topic_name, topic_info = app.predict_topic(
            "ulasan acak", "Positif", {"Positif": mock_bt}, mock_emb
        )
        self.assertEqual(topic_id, -1)
        self.assertIsInstance(topic_name, str)
        self.assertIn("Outlier", str(topic_name))
        self.assertIsNone(topic_info)

    def test_predict_topic_positive_cluster(self):
        """Positive case: BERTopic mengembalikan cluster topik valid beserta kata kunci."""
        mock_emb = MagicMock()
        mock_emb.encode.return_value = np.zeros((1, 384))

        mock_bt = MagicMock()
        mock_bt.topic_embeddings_ = np.zeros((3, 384))
        mock_bt.transform.return_value = ([0], [0.92])
        mock_bt.get_topic_info.return_value = pd.DataFrame(
            [{"Topic": 0, "Name": "0_transfer_cepat_mudah"}]
        )
        mock_bt.get_topic.return_value = [("transfer", 0.5), ("cepat", 0.4)]

        topic_id, topic_name, topic_info = app.predict_topic(
            "transfer cepat mudah", "Positif", {"Positif": mock_bt}, mock_emb
        )
        self.assertEqual(topic_id, 0)
        self.assertEqual(topic_name, "Kemudahan Transaksi & Transfer Gratis")
        self.assertEqual(topic_info, [("transfer", 0.5), ("cepat", 0.4)])

    def test_resolve_topic_name_mapped_and_fallback(self):
        """Positive & fallback case: nama cluster diambil dari TOPIC_LABELS atau fallback ke BERTopic."""
        mock_bt = MagicMock()
        mock_bt.get_topic_info.return_value = pd.DataFrame(
            [{"Topic": 99, "Name": "99_custom_topik"}]
        )

        # Mapped cluster
        self.assertEqual(
            app.resolve_topic_name("Negatif", 0, mock_bt),
            "Kendala Login & Akses Akun",
        )
        # Fallback ke Name BERTopic saat ID belum ada di TOPIC_LABELS
        self.assertEqual(
            app.resolve_topic_name("Negatif", 99, mock_bt),
            "99_custom_topik",
        )
        # Fallback saat bt_model None
        self.assertEqual(
            app.resolve_topic_name("Negatif", 99, None),
            "Topik 99",
        )

    def test_build_topic_summary_df_positive_and_negative(self):
        """Positive & negative case: tabel ringkasan memfilter outlier dan aman jika model kosong."""
        mock_bt = MagicMock()
        mock_bt.get_topic_info.return_value = pd.DataFrame(
            [
                {"Topic": -1, "Count": 500, "Name": "-1_outlier"},
                {"Topic": 1, "Count": 120, "Name": "1_saldo"},
                {"Topic": 0, "Count": 300, "Name": "0_login"},
            ]
        )
        mock_bt.get_topic.side_effect = lambda tid: (
            [("login", 0.4), ("akun", 0.3)] if tid == 0 else [("saldo", 0.5), ("uang", 0.2)]
        )

        df = app.build_topic_summary_df("Negatif", mock_bt)
        self.assertEqual(list(df["Cluster"]), ["#0", "#1"])
        self.assertEqual(
            list(df["Nama Topik"]),
            ["Kendala Login & Akses Akun", "Masalah Saldo & Transaksi Transfer"],
        )
        self.assertEqual(list(df["Jumlah Ulasan"]), [300, 120])
        self.assertEqual(df.iloc[0]["Kata Kunci"], "login, akun")

        # Negative case: model None atau kosong
        empty_df = app.build_topic_summary_df("Negatif", None)
        self.assertTrue(empty_df.empty)
        self.assertEqual(
            list(empty_df.columns),
            ["Cluster", "Nama Topik", "Jumlah Ulasan", "Kata Kunci"],
        )

    def test_load_bertopic_models_passes_embedding_model(self):
        """Memastikan BERTopic.load dipanggil dengan parameter embedding_model."""
        app.load_bertopic_models.clear()
        mock_emb = MagicMock()

        with patch("bertopic.BERTopic.load") as mock_load:
            mock_load.return_value = MagicMock()
            models = app.load_bertopic_models(mock_emb)

        self.assertTrue(mock_load.called)
        for call in mock_load.call_args_list:
            self.assertIs(call.kwargs.get("embedding_model"), mock_emb)
        self.assertTrue(len(models) > 0)

    def test_default_binary_label_encoder(self):
        """Positive case: fallback encoder memetakan 0 -> Negatif dan 1 -> Positif."""
        le = app.DefaultBinaryLabelEncoder()
        self.assertEqual(list(le.classes_), ["Negatif", "Positif"])
        self.assertEqual(le.inverse_transform([0])[0], "Negatif")
        self.assertEqual(le.inverse_transform([1])[0], "Positif")

    def test_predict_sentiment_binary_and_5_logits_fallback(self):
        """Positive case: predict_sentiment menangani output 2 logits maupun 5 logits."""
        le = app.DefaultBinaryLabelEncoder()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 200, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

        # Simulasi model dengan 5 logits (seperti _num_labels=5 pada config)
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(
            logits=torch.tensor([[0.2, 2.5, 9.9, 9.9, 9.9]])
        )

        sentimen, confidence, probs = app.predict_sentiment(
            "sangat puas", mock_tokenizer, mock_model, le, torch.device("cpu")
        )
        self.assertEqual(sentimen, "Positif")
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=5)
        self.assertGreater(confidence, 0.8)

    def test_load_classifier_without_label_encoder_pkl(self):
        """Positive case: load_classifier memakai DefaultBinaryLabelEncoder jika pkl tidak ada."""
        app.load_classifier.clear()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with (
            patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
            patch(
                "transformers.AutoModelForSequenceClassification.from_pretrained",
                return_value=mock_model,
            ),
        ):
            _, _, le, _ = app.load_classifier()

        self.assertIsInstance(le, app.DefaultBinaryLabelEncoder)

    def test_load_uploaded_table_csv_and_excel_positive(self):
        """Positive case: membaca file CSV UTF-8 dan Excel .xlsx valid."""
        csv_buf = io.BytesIO("ulasan,rating\nAplikasi bagus,5\n".encode("utf-8"))
        df_csv = app.load_uploaded_table(csv_buf, "reviews.csv")
        self.assertEqual(list(df_csv.columns), ["ulasan", "rating"])
        self.assertEqual(df_csv.iloc[0]["ulasan"], "Aplikasi bagus")

        xlsx_buf = io.BytesIO()
        pd.DataFrame({"komentar": ["Transfer cepat"], "skor": [5]}).to_excel(
            xlsx_buf, index=False
        )
        xlsx_buf.seek(0)
        df_xlsx = app.load_uploaded_table(xlsx_buf, "reviews.xlsx")
        self.assertEqual(list(df_xlsx.columns), ["komentar", "skor"])
        self.assertEqual(df_xlsx.iloc[0]["komentar"], "Transfer cepat")

    def test_load_uploaded_table_negative(self):
        """Negative case: format tidak didukung atau file CSV kosong melempar ValueError."""
        bad_buf = io.BytesIO(b"isi teks")
        with self.assertRaises(ValueError):
            app.load_uploaded_table(bad_buf, "notes.txt")

        empty_csv = io.BytesIO(b"")
        with self.assertRaises(ValueError):
            app.load_uploaded_table(empty_csv, "kosong.csv")

        header_only_csv = io.BytesIO(b"ulasan,rating\n")
        with self.assertRaises(ValueError):
            app.load_uploaded_table(header_only_csv, "header_only.csv")

    def test_detect_text_column_positive_and_negative(self):
        """Positive & negative case: deteksi kolom ulasan atau fallback ke kolom teks."""
        df_pref = pd.DataFrame({"id": [1], "Ulasan Pengguna": ["bagus"], "skor": [5]})
        self.assertEqual(app.detect_text_column(df_pref), "Ulasan Pengguna")

        df_fallback = pd.DataFrame({"angka": [10], "catatan": ["oke"]})
        self.assertEqual(app.detect_text_column(df_fallback), "catatan")

        self.assertIsNone(app.detect_text_column(pd.DataFrame()))

    def test_analyze_reviews_batch_positive_and_negative(self):
        """Positive & negative case: analisis batch campuran teks valid dan baris kosong/emoji."""
        le = app.DefaultBinaryLabelEncoder()
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda texts, **kwargs: {
            "input_ids": torch.ones((len(texts), 3), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 3), dtype=torch.long),
        }

        # Baris valid 0 -> Positif (logit index 1 tinggi), baris valid 1 -> Negatif (logit index 0 tinggi)
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(
            logits=torch.tensor([[0.1, 3.0], [2.8, 0.2]])
        )

        mock_emb = MagicMock()
        mock_emb.encode.side_effect = lambda texts: np.zeros((len(texts), 384))

        mock_bt_pos = MagicMock()
        mock_bt_pos.topic_embeddings_ = np.zeros((2, 384))
        mock_bt_pos.transform.return_value = ([0], [0.9])

        mock_bt_neg = MagicMock()
        mock_bt_neg.topic_embeddings_ = np.zeros((2, 384))
        mock_bt_neg.transform.return_value = ([-1], [0.2])

        progress_values = []
        result_df = app.analyze_reviews_batch(
            ["Aplikasi mantap gratis transfer", "😊🔥", "Tidak bisa login"],
            mock_tokenizer,
            mock_model,
            le,
            torch.device("cpu"),
            {"Positif": mock_bt_pos, "Negatif": mock_bt_neg},
            mock_emb,
            progress_callback=progress_values.append,
        )

        self.assertEqual(len(result_df), 3)
        self.assertEqual(result_df.iloc[0]["Sentimen"], "Positif")
        self.assertEqual(result_df.iloc[0]["Cluster"], "#0")
        self.assertEqual(
            result_df.iloc[0]["Nama Topik"],
            "Kemudahan Transaksi & Transfer Gratis",
        )
        self.assertEqual(result_df.iloc[1]["Sentimen"], "Tidak Valid")
        self.assertEqual(result_df.iloc[1]["Cluster"], "-")
        self.assertEqual(result_df.iloc[2]["Sentimen"], "Negatif")
        self.assertEqual(result_df.iloc[2]["Cluster"], "Outlier")
        self.assertEqual(progress_values[-1], 1.0)

    def test_scrape_playstore_reviews_date_filter_and_early_stop(self):
        """Positive case: pagination difilter tanggal dan berhenti saat melewati start_date."""
        from datetime import date, datetime

        page1 = [
            {
                "at": datetime(2026, 11, 25, 10, 0),
                "userName": "User Baru",
                "score": 5,
                "content": "Melewati batas akhir",
                "thumbsUpCount": 1,
            },
            {
                "at": datetime(2026, 11, 20, 9, 30),
                "userName": "User Valid",
                "score": 4,
                "content": "Transfer cepat dan mudah",
                "thumbsUpCount": 3,
            },
        ]
        page2 = [
            {
                "at": datetime(2026, 11, 10, 8, 0),
                "userName": "User Lama",
                "score": 2,
                "content": "Sudah di luar rentang mulai",
                "thumbsUpCount": 0,
            }
        ]

        with patch(
            "google_play_scraper.reviews",
            side_effect=[(page1, "tok_2"), (page2, "tok_3")],
        ) as mock_reviews:
            df = app.scrape_playstore_reviews(
                start_date=date(2026, 11, 15),
                end_date=date(2026, 11, 22),
                max_reviews=50,
            )

        self.assertEqual(mock_reviews.call_count, 2)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["userName"], "User Valid")
        self.assertEqual(df.iloc[0]["ulasan"], "Transfer cepat dan mudah")
        self.assertEqual(df.iloc[0]["score"], 4)

    def test_scrape_playstore_reviews_negative_cases(self):
        """Negative case: tanggal terbalik, respons kosong, dan error koneksi."""
        from datetime import date

        with self.assertRaises(ValueError):
            app.scrape_playstore_reviews(
                start_date=date(2026, 11, 25),
                end_date=date(2026, 11, 20),
            )

        with patch("google_play_scraper.reviews", return_value=([], None)):
            empty_df = app.scrape_playstore_reviews(
                start_date=date(2026, 11, 1),
                end_date=date(2026, 11, 25),
            )
        self.assertTrue(empty_df.empty)
        self.assertEqual(
            list(empty_df.columns),
            ["tanggal", "userName", "score", "ulasan", "thumbsUpCount"],
        )

        with patch("google_play_scraper.reviews", side_effect=OSError("timeout")):
            with self.assertRaises(RuntimeError):
                app.scrape_playstore_reviews(
                    start_date=date(2026, 11, 1),
                    end_date=date(2026, 11, 25),
                )

        with self.assertRaises(ValueError):
            app.scrape_playstore_reviews(
                start_date=date(2026, 11, 1),
                end_date=date(2026, 11, 25),
                max_reviews=-1,
            )

    def test_scrape_playstore_reviews_unlimited_mode(self):
        """Positive case: max_reviews=0 atau None mengambil semua halaman tanpa batas 50 halaman."""
        from datetime import date, datetime

        pages = [
            (
                [
                    {
                        "at": datetime(2026, 11, 20, 10, 0),
                        "userName": f"User {idx}",
                        "score": 5,
                        "content": f"Bagus {idx}",
                        "thumbsUpCount": 0,
                    }
                    for _ in range(12)
                ],
                f"tok_{idx + 1}" if idx < 54 else None,
            )
            for idx in range(55)
        ]

        with patch("google_play_scraper.reviews", side_effect=pages) as mock_reviews:
            df = app.scrape_playstore_reviews(
                start_date=date(2026, 11, 15),
                end_date=date(2026, 11, 22),
                max_reviews=0,
            )

        self.assertEqual(mock_reviews.call_count, 55)
        self.assertEqual(len(df), 55 * 12)


if __name__ == "__main__":
    unittest.main()
