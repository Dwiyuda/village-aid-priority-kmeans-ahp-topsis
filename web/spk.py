"""
Mesin SPK — AHP, TOPSIS, SAW.

Algoritma diangkat apa adanya dari NB06_SPK_TOPSIS_SAW dan NB07_AHP_Pembobotan
supaya hasil web identik dengan yang tertulis di laporan.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

# ── Tujuh kriteria SPK. pct_nonaktif sengaja tidak diikutkan ───────────────
KRITERIA = [
    "pct_desil1", "pct_desil2", "pct_desil3", "pct_desil4",
    "pct_desil5", "pct_desil6_10", "pct_belum_prmk",
]
LABEL = ["Desil 1", "Desil 2", "Desil 3", "Desil 4",
         "Desil 5", "Desil 6-10", "Belum diperingkat"]
# pct_desil6_10 = cost (makin sejahtera, makin rendah prioritas)
IS_BENEFIT = np.array([c != "pct_desil6_10" for c in KRITERIA])

RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}
EPS = 1e-9

# ── Matriks perbandingan berpasangan hasil penelitian (CR = 0,0100) ────────
MATRIKS_RISET = [
    [1,   2,   3,   4,   5,   3,   2  ],
    [1/2, 1,   2,   3,   4,   2,   1  ],
    [1/3, 1/2, 1,   2,   3,   1,   1/2],
    [1/4, 1/3, 1/2, 1,   2,   1/2, 1/3],
    [1/5, 1/4, 1/3, 1/2, 1,   1/3, 1/4],
    [1/3, 1/2, 1,   2,   3,   1,   1/2],
    [1/2, 1,   2,   3,   4,   2,   1  ],
]

# Sebelas pola pembobotan, ditulis sebagai 21 nilai skala Saaty (segitiga atas,
# urut baris). Semua sudah diuji konsisten (CR < 0,10). Nilai preset baru
# diturunkan dari vektor prioritas kebijakan lalu dibulatkan ke skala Saaty.
PRESET = {
    "riset":    None,  # dipakai MATRIKS_RISET langsung
    "merata":   [0] * 21,
    "ekstrem":  [1, 4, 5, 6, 5, 3,  3, 4, 5, 4, 2,  2, 3, 2, 1,  1, 1, -1,  1, -2,  -2],
    "cakupan":  [2, 3, 4, 5, 3, -1,  2, 3, 4, 2, -2,  2, 3, 1, -3,  2, 1, -4,  1, -4,  -5],
    # ── Pola tambahan (dirancang policy-first, diverifikasi CR) ─────────────
    "rentan":   [0, -1, -1, 0, 1, 0, 0, 0, 1, 2, 1, 0, 2, 4, 2, 2, 3, 2, 0, 0, 0],       # CR 0,0152
    "darurat":  [1, 4, 8, 8, 8, 8, 1, 3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],         # CR 0,0004
    "gradasi":  [0, 0, 1, 1, 2, 2, 0, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 0, 0, 0],         # CR 0,0122
    "ganda":    [0, 2, 3, 3, 3, 0, 1, 2, 2, 2, 0, 0, 0, 0, -2, 0, 0, -3, 0, -3, -3],     # CR 0,0047
    "anti":     [0, 0, 1, 2, 0, 1, 0, 1, 2, -1, 1, 0, 1, -1, 0, 0, -2, 0, -3, 0, 2],     # CR 0,0158
    "menengah": [-1, -1, 0, 1, 1, 0, 0, 1, 3, 3, 2, 1, 3, 3, 2, 1, 1, 0, 0, 0, 0],       # CR 0,0076
    "data":     [0, 0, 0, 0, 0, -2, 0, 0, 0, 0, -2, 0, 0, 0, -4, 0, 0, -4, 0, -4, -4],   # CR 0,0054
}

# Metadata tampilan preset: (kunci, nama, deskripsi singkat), dikelompokkan per tema.
PRESET_GRUP = [
    ("Resmi & netral", [
        ("riset",    "Bobot penelitian",        "Nilai resmi di laporan (logika desil)"),
        ("merata",   "Bobot merata",            "Semua kriteria dianggap sama penting"),
    ]),
    ("Fokus kemiskinan", [
        ("ekstrem",  "Fokus kemiskinan ekstrem", "Tekankan Desil 1 & 2"),
        ("darurat",  "Darurat termiskin mutlak", "Hampir semua bobot ke Desil 1"),
        ("rentan",   "Fokus keluarga rentan",    "Tekankan Desil 3 & 4 (hampir miskin)"),
        ("menengah", "Fokus menengah-bawah",     "Tekankan Desil 2 & 3"),
    ]),
    ("Fokus data", [
        ("cakupan",  "Fokus cakupan data",       "Prioritaskan desa berdata bolong"),
        ("data",     "Data super dominan",       "Cakupan data paling dominan"),
    ]),
    ("Lanjutan", [
        ("gradasi",  "Gradasi murni (BPS)",      "Bobot menurun rapi ikut desil"),
        ("ganda",    "Ganda: ekstrem + data",    "Desil 1 & belum-diperingkat setara"),
        ("anti",     "Anti salah-sasaran",       "Tekan kuat desa yang mampu (cost)"),
    ]),
]


def nilai_ke_rasio(v):
    """Skala slider (-8..8) ke rasio Saaty. 0 -> 1, 1 -> 2, ..., -1 -> 1/2."""
    v = int(v)
    return v + 1 if v >= 0 else 1 / (1 - v)


def rasio_ke_nilai(r):
    """Kebalikan nilai_ke_rasio, untuk mengisi posisi slider dari matriks."""
    return int(round(r - 1)) if r >= 1 else -int(round(1 / r - 1))


def matriks_dari_nilai(nilai):
    """Susun matriks 7x7 dari 21 nilai segitiga atas."""
    n = len(KRITERIA)
    A = np.ones((n, n))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            r = nilai_ke_rasio(nilai[k])
            A[i][j] = r
            A[j][i] = 1 / r
            k += 1
    return A


def nilai_dari_matriks(A):
    """Ambil 21 nilai segitiga atas dari matriks, untuk posisi awal slider."""
    n = len(KRITERIA)
    return [rasio_ke_nilai(A[i][j]) for i in range(n) for j in range(i + 1, n)]


def hitung_ahp(A):
    """Vektor prioritas + uji konsistensi. Sama persis dengan NB07."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    w = (A / A.sum(axis=0)).mean(axis=1)
    lamda_max = float(((A @ w) / w).mean())
    ci = (lamda_max - n) / (n - 1)
    cr = ci / RI[n]
    return w, lamda_max, ci, cr


def matriks_preset(key):
    """Matriks 7x7 untuk sebuah preset (riset memakai MATRIKS_RISET langsung)."""
    if key == "riset":
        return np.array(MATRIKS_RISET, dtype=float)
    return matriks_dari_nilai(PRESET[key])


def cr_semua_preset():
    """CR tiap preset — untuk ditampilkan pada tombolnya. Semua < 0,10."""
    return {k: hitung_ahp(matriks_preset(k))[3] for k in PRESET}


def topsis(X, w):
    """Skor kedekatan TOPSIS. Sama persis dengan NB06."""
    R = X / (np.sqrt((X ** 2).sum(axis=0)) + EPS)
    V = R * w
    a_pos = np.where(IS_BENEFIT, V.max(axis=0), V.min(axis=0))
    a_neg = np.where(IS_BENEFIT, V.min(axis=0), V.max(axis=0))
    d_pos = np.sqrt(((V - a_pos) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - a_neg) ** 2).sum(axis=1))
    return d_neg / (d_pos + d_neg + EPS)


def saw(X, w):
    """Skor SAW dengan normalisasi rasio. Sama persis dengan NB06."""
    R = np.zeros_like(X)
    for j in range(X.shape[1]):
        kol = X[:, j]
        R[:, j] = kol / (kol.max() + EPS) if IS_BENEFIT[j] else (kol.min() + EPS) / (kol + EPS)
    return (R * w).sum(axis=1)


def muat_data():
    """Gabungkan fitur proporsi dengan label klaster, lalu beri nama zona."""
    df = pd.read_csv(DATA_DIR / "dtsen_clean.csv")
    km = pd.read_csv(DATA_DIR / "hasil_kmeans.csv")[
        ["KECAMATAN", "KELURAHAN", "cluster_kmeans"]]
    df = df.merge(km, on=["KECAMATAN", "KELURAHAN"], how="left")

    # Zona diberi nama dari rata-rata Desil 1+2 tiap klaster, bukan nomor klaster
    urut = (df.assign(_m=df["pct_desil1"] + df["pct_desil2"])
              .groupby("cluster_kmeans")["_m"].mean()
              .sort_values(ascending=False).index.tolist())
    nama = ["Kemiskinan Tinggi", "Kemiskinan Sedang", "Kemiskinan Rendah"]
    df["zona"] = df["cluster_kmeans"].map(dict(zip(urut, nama)))
    return df


def hitung_semua(df, A):
    """Jalankan AHP lalu perankingan TOPSIS & SAW, kembalikan tabel siap tampil."""
    w, lamda_max, ci, cr = hitung_ahp(A)
    X = df[KRITERIA].values.astype(float)

    out = df.copy()
    out["skor_topsis"] = topsis(X, w)
    out["skor_saw"] = saw(X, w)
    out["rank_topsis"] = out["skor_topsis"].rank(ascending=False, method="min").astype(int)
    out["rank_saw"] = out["skor_saw"].rank(ascending=False, method="min").astype(int)

    # Penanda masalah administratif: proporsi belum diperingkat di 25% teratas
    ambang = out["pct_belum_prmk"].quantile(0.75)
    out["jenis_masalah"] = np.where(
        out["pct_belum_prmk"] >= ambang, "Administratif", "Ekonomi")
    out["rekomendasi"] = [rekomendasi(z, j)
                          for z, j in zip(out["zona"], out["jenis_masalah"])]

    out = out.sort_values("rank_topsis").reset_index(drop=True)
    info = {
        "bobot": w, "lambda_max": lamda_max, "ci": ci, "cr": cr,
        "konsisten": bool(cr < 0.10), "ambang_belum": float(ambang),
        # Kedua kolom sudah berupa peringkat, jadi Pearson di atasnya
        # secara matematis sama dengan Spearman — tanpa perlu scipy.
        "korelasi": float(out["rank_topsis"].corr(out["rank_saw"])),
    }
    return out, info


def rekomendasi(zona, jenis):
    if jenis == "Administratif":
        return "Prioritas survei dan pemutakhiran data"
    if zona == "Kemiskinan Tinggi":
        return "Prioritas bantuan tunai (PKH/BLT)"
    if zona == "Kemiskinan Sedang":
        return "Bantuan pangan dan pemberdayaan ekonomi mikro"
    return "Bukan prioritas, pemberdayaan preventif"
