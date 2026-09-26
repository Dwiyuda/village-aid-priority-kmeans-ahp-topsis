"""
utils.py — Helper functions untuk TA Clustering DTSEN Kabupaten Lahat
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Direktori ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

# ── Nama kolom 20 (sesuai header baris 1-2 file Excel: tiap kategori desil
#    punya 2 kolom berpasangan KELUARGA lalu INDIVIDU) ─────────────────────────
COLS_RAW = [
    "KECAMATAN",
    "KELURAHAN",
    "JUMLAH_KELUARGA",
    "JUMLAH_INDIVIDU",
    "DESIL1_KLG",
    "DESIL1_IND",
    "DESIL2_KLG",
    "DESIL2_IND",
    "DESIL3_KLG",
    "DESIL3_IND",
    "DESIL4_KLG",
    "DESIL4_IND",
    "DESIL5_KLG",
    "DESIL5_IND",
    "DESIL6_10_KLG",
    "DESIL6_10_IND",
    "BELUM_PRMK_KLG",
    "BELUM_PRMK_IND",
    "NONAKTIF_KLG",
    "NONAKTIF_IND",
]

# ── Fitur proporsi hasil rekayasa ──────────────────────────────────────────────
FEATURE_COLS = [
    "pct_desil1",
    "pct_desil2",
    "pct_desil3",
    "pct_desil4",
    "pct_desil5",
    "pct_desil6_10",
    "pct_belum_prmk",
    "pct_nonaktif",
]


def load_dtsen(filepath) -> pd.DataFrame:
    """
    Baca file Excel DTSEN, bersihkan, validasi, dan kembalikan DataFrame bersih.

    Langkah:
    1. Baca dengan engine='openpyxl' — file .xls sebenarnya format xlsx
    2. Skip 2 baris header (row 0-1), data mulai row 2
    3. Assign COLS_RAW (20 kolom)
    4. Hapus baris subtotal kecamatan (KELURAHAN == KECAMATAN)
    5. Hapus baris total kabupaten (KECAMATAN == 'Kabupaten Lahat')
    6. Konversi kolom numerik
    7. Assert 372 baris & 24 kecamatan sebelum filter KK=0
    8. Exclude desa dengan JUMLAH_KELUARGA == 0 (Talang Jawa) → 371 baris
    9. Print summary

    Parameters
    ----------
    filepath : str | Path
        Path ke file Excel DTSEN.

    Returns
    -------
    pd.DataFrame  371 baris × 20 kolom, terfilter dan siap analisis.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {filepath.resolve()}")

    # Baca seluruh sheet, tanpa header otomatis
    raw = pd.read_excel(filepath, header=None, engine="openpyxl")

    # Data dimulai dari baris index 2 (skip 2 baris header)
    df = raw.iloc[2:].copy()
    df = df.iloc[:, :20]          # ambil 20 kolom pertama
    df.columns = COLS_RAW
    df = df.reset_index(drop=True)

    # Hapus baris total kabupaten
    df = df[df["KECAMATAN"].astype(str).str.strip() != "Kabupaten Lahat"]

    # Hapus baris subtotal kecamatan (KELURAHAN == KECAMATAN)
    df = df[
        df["KELURAHAN"].astype(str).str.strip() != df["KECAMATAN"].astype(str).str.strip()
    ]

    # Bersihkan whitespace pada string
    df["KECAMATAN"] = df["KECAMATAN"].astype(str).str.strip()
    df["KELURAHAN"]  = df["KELURAHAN"].astype(str).str.strip()

    # Hapus baris TOTAL
    df = df[df["KECAMATAN"].str.upper() != "TOTAL"]

    # Konversi kolom numerik
    numeric_cols = COLS_RAW[2:]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.reset_index(drop=True)

    # ── Validasi sebelum exclude KK=0 ─────────────────────────────────────────
    n_before = len(df)
    n_kec = df["KECAMATAN"].nunique()

    assert n_before == 372, (
    f"Jumlah baris seharusnya 372, didapat {n_before}. "
    "Periksa file Excel dan logika filter."
    )

    assert n_kec == 24, (
    f"Jumlah kecamatan seharusnya 24, didapat {n_kec}."
    )

    # ── Exclude desa tanpa profil kesejahteraan yang valid ───────────────────
    #    Kriteria: JUMLAH_KELUARGA = 0, ATAU tidak ada satu pun keluarga yang
    #    terperingkat di desil mana pun (Desil 1-6/10 semuanya 0). Desa seperti
    #    ini tidak memiliki profil kemiskinan yang dapat dianalisis — misalnya
    #    Talang Jawa (Kec. Lahat) pada data 20 Juli: 1 KK, 100% belum diperingkat.
    DESIL_KLG = ["DESIL1_KLG", "DESIL2_KLG", "DESIL3_KLG",
                 "DESIL4_KLG", "DESIL5_KLG", "DESIL6_10_KLG"]
    profil_kosong = df[
        (df["JUMLAH_KELUARGA"] == 0) | (df[DESIL_KLG].sum(axis=1) == 0)
    ][["KECAMATAN", "KELURAHAN", "JUMLAH_KELUARGA"]]
    if len(profil_kosong) > 0:
        print("ℹ  Desa tanpa profil desil valid (di-exclude):")
        print(profil_kosong.to_string(index=False))
    df = df[
        (df["JUMLAH_KELUARGA"] > 0) & (df[DESIL_KLG].sum(axis=1) > 0)
    ].reset_index(drop=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  DTSEN Kabupaten Lahat — Data Loaded")
    print(f"{'='*55}")
    print(f"  Baris (desa/kel)  : {len(df)}")
    print(f"  Kolom             : {df.shape[1]}")
    print(f"  Kecamatan         : {df['KECAMATAN'].nunique()}")
    print(f"  Total KK          : {df['JUMLAH_KELUARGA'].sum():,.0f}")
    print(f"  Total Individu    : {df['JUMLAH_INDIVIDU'].sum():,.0f}")
    print(f"{'='*55}\n")

    return df


def buat_fitur_proporsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hitung 8 fitur proporsi dari kolom KK absolut.

    Denominatornya adalah JUMLAH_KELUARGA untuk pct_desil* dan pct_belum_prmk.
    pct_nonaktif  = NONAKTIF_KLG / JUMLAH_KELUARGA × 100
    (NONAKTIF berada di luar total KK, bukan bagian dari penjumlahan desil)

    Returns
    -------
    pd.DataFrame  kolom identitas (KECAMATAN, KELURAHAN, JUMLAH_KELUARGA)
                  + 8 kolom pct_ (float, satuan %)
    """
    df = df.copy()
    denom = df["JUMLAH_KELUARGA"]

    df["pct_desil1"]     = df["DESIL1_KLG"]     / denom * 100
    df["pct_desil2"]     = df["DESIL2_KLG"]     / denom * 100
    df["pct_desil3"]     = df["DESIL3_KLG"]     / denom * 100
    df["pct_desil4"]     = df["DESIL4_KLG"]     / denom * 100
    df["pct_desil5"]     = df["DESIL5_KLG"]     / denom * 100
    df["pct_desil6_10"]  = df["DESIL6_10_KLG"]  / denom * 100
    df["pct_belum_prmk"] = df["BELUM_PRMK_KLG"] / denom * 100
    df["pct_nonaktif"]   = df["NONAKTIF_KLG"]   / denom * 100

    id_cols = ["KECAMATAN", "KELURAHAN", "JUMLAH_KELUARGA"]
    return df[id_cols + FEATURE_COLS].reset_index(drop=True)
