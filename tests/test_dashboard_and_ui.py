"""Unit test untuk logika dashboard, standar UI, dan batas panjang file Python."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Pastikan root project ada di sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import app  # noqa: E402
from src import ui_sections  # noqa: E402


class TestDashboardAndUI(unittest.TestCase):
    """Kumpulan unit test untuk dashboard, facade UI, dan kualitas kode."""

    def test_dashboard_cache_save_load_and_dedup(self):
        """Positive & negative case: simpan/muat cache Dashboard dan deduplikasi data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "cache.csv"

            # Negative case: file cache belum ada -> DataFrame kosong berkolom standar
            empty_loaded = app.load_dashboard_cache(cache_file)
            self.assertTrue(empty_loaded.empty)
            self.assertEqual(list(empty_loaded.columns), app.DASHBOARD_COLUMNS)

            df_old = pd.DataFrame(
                [
                    {
                        "tanggal": "2025-12-10 08:00",
                        "userName": "Ani",
                        "score": 5,
                        "ulasan": "Bagus banget",
                        "thumbsUpCount": 1,
                        "Teks Bersih": "bagus banget",
                        "Sentimen": "Positif",
                        "Keyakinan (%)": 98.5,
                        "Cluster": "#0",
                        "Nama Topik": "Kemudahan Transaksi & Transfer Gratis",
                    }
                ]
            )
            df_new = pd.DataFrame(
                [
                    {
                        "tanggal": "2025-12-10 08:00",
                        "userName": "Ani",
                        "score": 5,
                        "ulasan": "Bagus banget",
                        "thumbsUpCount": 2,
                        "Teks Bersih": "bagus banget",
                        "Sentimen": "Positif",
                        "Keyakinan (%)": 99.0,
                        "Cluster": "#0",
                        "Nama Topik": "Kemudahan Transaksi & Transfer Gratis",
                    },
                    {
                        "tanggal": "2026-01-15 11:30",
                        "userName": "Budi",
                        "score": 1,
                        "ulasan": "Gagal login terus",
                        "thumbsUpCount": 0,
                        "Teks Bersih": "gagal login terus",
                        "Sentimen": "Negatif",
                        "Keyakinan (%)": 96.0,
                        "Cluster": "#0",
                        "Nama Topik": "Kendala Login & Akses Akun",
                    },
                ]
            )

            merged = app.merge_dashboard_cache(df_old, df_new)
            self.assertEqual(len(merged), 2)
            self.assertEqual(list(merged["tahun"]), [2026, 2025])
            self.assertEqual(list(merged["bulan"]), [1, 12])

            app.save_dashboard_cache(merged, cache_file)
            reloaded = app.load_dashboard_cache(cache_file)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(reloaded.iloc[0]["userName"], "Budi")

    def test_dashboard_filter_metrics_and_trends(self):
        """Positive & negative case: filter Tahun->Bulan->Sentimen, KPI, dan tren bulanan."""
        raw_df = pd.DataFrame(
            [
                {
                    "tanggal": "2026-01-10 09:00",
                    "userName": "A",
                    "score": 5,
                    "ulasan": "transfer cepat",
                    "thumbsUpCount": 0,
                    "Teks Bersih": "transfer cepat",
                    "Sentimen": "Positif",
                    "Keyakinan (%)": 99.0,
                    "Cluster": "#0",
                    "Nama Topik": "Kemudahan Transaksi & Transfer Gratis",
                },
                {
                    "tanggal": "2026-02-12 10:00",
                    "userName": "B",
                    "score": 1,
                    "ulasan": "susah login",
                    "thumbsUpCount": 0,
                    "Teks Bersih": "susah login",
                    "Sentimen": "Negatif",
                    "Keyakinan (%)": 95.0,
                    "Cluster": "#0",
                    "Nama Topik": "Kendala Login & Akses Akun",
                },
                {
                    "tanggal": "2026-02-20 14:00",
                    "userName": "C",
                    "score": 2,
                    "ulasan": "tidak bisa login",
                    "thumbsUpCount": 0,
                    "Teks Bersih": "tidak bisa login",
                    "Sentimen": "Negatif",
                    "Keyakinan (%)": 97.0,
                    "Cluster": "#0",
                    "Nama Topik": "Kendala Login & Akses Akun",
                },
            ]
        )
        dash_df = app.merge_dashboard_cache(pd.DataFrame(), raw_df)

        # Semua Bulan di tahun 2026
        yr_2026 = app.filter_dashboard_data(dash_df, 2026, "Semua Bulan", "Semua")
        self.assertEqual(len(yr_2026), 3)

        # Bulan Februari saja
        feb_2026 = app.filter_dashboard_data(dash_df, 2026, 2, "Semua")
        self.assertEqual(len(feb_2026), 2)

        # Negative case: tahun tidak ada
        empty_yr = app.filter_dashboard_data(dash_df, 2023, "Semua Bulan", "Semua")
        self.assertTrue(empty_yr.empty)

        metrics = app.summarize_dashboard_metrics(yr_2026)
        self.assertEqual(metrics["total"], 3)
        self.assertEqual(metrics["positif_count"], 1)
        self.assertEqual(metrics["negatif_count"], 2)
        self.assertIn("Kendala Login & Akses Akun", metrics["top_topic"])

        trend = app.build_monthly_trend_df(yr_2026)
        self.assertEqual(list(trend["bulan"]), [1, 2])
        self.assertEqual(list(trend["bulan_label"]), ["Jan", "Feb"])

        topic_trend = app.build_topic_monthly_trend_df(yr_2026, top_n=5)
        self.assertFalse(topic_trend.empty)

        empty_metrics = app.summarize_dashboard_metrics(empty_yr)
        self.assertEqual(empty_metrics["total"], 0)
        self.assertEqual(empty_metrics["top_topic"], "-")

    def test_build_batch_distribution_and_filter_results(self):
        """Positive & negative case: agregasi distribusi batch dan filter Sentimen/Cluster."""
        sample_df = pd.DataFrame(
            [
                {"Ulasan Asli": "a", "Sentimen": "Positif", "Cluster": "#0", "Nama Topik": "Kemudahan Transaksi"},
                {"Ulasan Asli": "b", "Sentimen": "Positif", "Cluster": "#0", "Nama Topik": "Kemudahan Transaksi"},
                {"Ulasan Asli": "c", "Sentimen": "Negatif", "Cluster": "#1", "Nama Topik": "Masalah Saldo"},
                {"Ulasan Asli": "d", "Sentimen": "Tidak Valid", "Cluster": "-", "Nama Topik": "Kosong"},
            ]
        )
        dist = app.build_batch_distribution_df(sample_df)
        self.assertEqual(len(dist), 2)
        self.assertEqual(dist.iloc[0]["Jumlah Ulasan"], 2)
        self.assertEqual(dist.iloc[0]["Label Topik"], "#0 - Kemudahan Transaksi")

        filtered_neg = app.filter_batch_results(sample_df, "Negatif", "#1")
        self.assertEqual(len(filtered_neg), 1)
        self.assertEqual(filtered_neg.iloc[0]["Ulasan Asli"], "c")

        self.assertTrue(app.build_batch_distribution_df(pd.DataFrame()).empty)
        self.assertTrue(app.filter_batch_results(sample_df, "Positif", "#1").empty)

    def test_ui_standards_no_em_dash_or_deprecated_width(self):
        """Memastikan UI bebas karakter em dash (R-02) dan bebas API deprecated use_container_width."""
        ui_files = (
            "app.py",
            "src/ui_sections.py",
            "src/ui_styles.py",
            "src/ui_dashboard.py",
            "src/ui_tabs.py",
            "src/ui_playstore.py",
        )
        for rel_path in ui_files:
            content = (ROOT_DIR / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("—", content, f"Em dash ditemukan di {rel_path}")
            self.assertNotIn("use_container_width", content, f"use_container_width ditemukan di {rel_path}")

    def test_dark_mode_adaptive_tokens_and_no_chart_title_overlap(self):
        """Memastikan CSS tidak memakai fallback gelap --text-color/#0F172A dan judul grafik tidak bertumpuk."""
        styles_src = (ROOT_DIR / "src/ui_styles.py").read_text(encoding="utf-8")
        self.assertNotIn("var(--text-color, #0F172A)", styles_src)
        self.assertIn("currentColor", styles_src)

        dashboard_src = (ROOT_DIR / "src/ui_dashboard.py").read_text(encoding="utf-8")
        tabs_src = (ROOT_DIR / "src/ui_tabs.py").read_text(encoding="utf-8")
        self.assertNotIn("title=f\"", dashboard_src)
        self.assertNotIn("title=\"Distribusi Topik Cluster\"", tabs_src)
        self.assertIn("chart-title", dashboard_src)
        self.assertIn("chart-title", tabs_src)

    def test_ui_sections_reexports(self):
        """Memastikan facade src.ui_sections tetap mengekspor seluruh fungsi UI publik."""
        expected_symbols = (
            "apply_custom_styles",
            "render_dashboard_charts",
            "render_dashboard_tab",
            "render_batch_analysis_results",
            "render_text_tab",
            "render_upload_tab",
            "render_playstore_tab",
            "render_topic_reference_tab",
        )
        for name in expected_symbols:
            self.assertTrue(hasattr(ui_sections, name), f"{name} tidak tersedia di src.ui_sections")

    def test_no_python_file_exceeds_500_lines(self):
        """Memastikan setiap file Python di repo memiliki maksimal 500 baris."""
        ignored_parts = {"__pycache__", ".venv", ".git"}
        for py_file in ROOT_DIR.rglob("*.py"):
            if any(part in ignored_parts for part in py_file.parts):
                continue
            line_count = len(py_file.read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(
                line_count,
                500,
                f"{py_file.relative_to(ROOT_DIR)} memiliki {line_count} baris (> 500)",
            )


if __name__ == "__main__":
    unittest.main()
