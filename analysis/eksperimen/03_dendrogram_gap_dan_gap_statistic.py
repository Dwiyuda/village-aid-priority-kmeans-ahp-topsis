"""
Eksperimen: Analisis Gap Dendrogram (nyata) + Gap Statistic

NB04 mengklaim k ditentukan dari "celah terpanjang pada dendrogram", tapi
kodenya cuma hardcode K_OPTIMAL=3 tanpa analisis apa pun -- klaim itu tidak
pernah benar-benar diuji. Script ini menjalankan dua metode yang belum
pernah dicoba di proyek ini:

1. Analisis gap dendrogram nyata -- menghitung jarak merge dari linkage
   matrix Ward, mencari celah (gap) terbesar antar merge, dan menentukan k
   dari situ (bukan visual/tebakan).
2. Gap Statistic (Tibshirani, Walther & Hastie, 2001) -- membandingkan
   within-cluster dispersion data asli terhadap data referensi acak seragam,
   untuk k=1..10.

Output disimpan di output/eksperimen_dendrogram_gap/ (terpisah dari hasil
resmi di output/results/).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import load_dtsen, buat_fitur_proporsi, FEATURE_COLS, DATA_DIR

OUT_DIR = ROOT / "output" / "eksperimen_dendrogram_gap"
OUT_DIR.mkdir(parents=True, exist_ok=True)
matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 10})

df = load_dtsen(DATA_DIR / "20_JULI_2026-_Rekap_DTSEN_Kabupaten_Lahat.xls")
df_feat = buat_fitur_proporsi(df)
X = StandardScaler().fit_transform(df_feat[FEATURE_COLS].values)
n_samples = X.shape[0]
rng = np.random.RandomState(42)

# ══════════════════════════════════════════════════════════════════════════
# 1. ANALISIS GAP DENDROGRAM (NYATA)
# ══════════════════════════════════════════════════════════════════════════
Z = linkage(X, method="ward")

TOP_N = 12  # tinjau k=2 s.d. 13 (mencakup rentang k yang sama dgn eksperimen 01)
last_merges_asc = Z[-TOP_N:, 2]          # jarak merge terakhir, terurut naik
last_merges_desc = last_merges_asc[::-1]  # index 0 = merge termahal (2 klaster -> 1)

# gaps[i] = selisih jarak antara merge ke-(i+2->i+1 klaster) dan (i+3->i+2 klaster)
gaps = -np.diff(last_merges_desc)
k_values = np.arange(2, 2 + len(gaps))

df_dendro = pd.DataFrame({
    "k": k_values,
    "jarak_merge_untuk_capai_k": last_merges_desc[:-1],
    "gap_ke_merge_berikutnya": gaps,
})
df_dendro.to_csv(OUT_DIR / "dendrogram_gap_per_k.csv", index=False)

best_idx = int(np.argmax(gaps))
k_dendrogram = int(k_values[best_idx])
assert 2 <= k_dendrogram <= TOP_N + 1, "k dari gap dendrogram di luar rentang wajar"

cut_height = float((last_merges_desc[best_idx] + last_merges_desc[best_idx + 1]) / 2)

print("Tabel gap dendrogram (k vs celah ke merge berikutnya):")
print(df_dendro.to_string(index=False))
print(f"\n>>> k disarankan dari GAP DENDROGRAM TERBESAR: k = {k_dendrogram}")
print(f">>> Tinggi potong (cut height) yang sesuai      : {cut_height:.4f}")

# Plot dendrogram dengan garis potong pada tinggi yang benar-benar dihitung
fig, ax = plt.subplots(figsize=(11, 5))
dendrogram(Z, truncate_mode="lastp", p=30, color_threshold=cut_height, ax=ax)
ax.axhline(cut_height, color="red", linestyle="--", linewidth=1.5,
           label=f"Garis potong (k={k_dendrogram}, tinggi={cut_height:.2f})")
ax.set_title("Dendrogram dengan Garis Potong Berbasis Gap Terbesar (Nyata)", fontweight="bold")
ax.set_xlabel("Sampel (atau ukuran klaster gabungan)")
ax.set_ylabel("Jarak Merge (Ward)")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "dendrogram_dengan_garis_potong.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# 2. GAP STATISTIC (Tibshirani, Walther & Hastie, 2001)
# ══════════════════════════════════════════════════════════════════════════
def hitung_log_wk(data, labels, k):
    """Log within-cluster dispersion (pooled) untuk k klaster."""
    wk = 0.0
    for c in range(k):
        pts = data[labels == c]
        if len(pts) > 1:
            centroid = pts.mean(axis=0)
            wk += ((pts - centroid) ** 2).sum()
    return np.log(wk) if wk > 0 else 0.0


K_RANGE_GAP = range(1, 11)
B = 30  # jumlah dataset referensi acak per k
mins, maxs = X.min(axis=0), X.max(axis=0)

gap_rows = []
for k in K_RANGE_GAP:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    log_wk = hitung_log_wk(X, km.labels_, k)

    log_wkb_list = []
    for b in range(B):
        X_ref = rng.uniform(mins, maxs, size=X.shape)
        km_ref = KMeans(n_clusters=k, random_state=b, n_init=10).fit(X_ref)
        log_wkb_list.append(hitung_log_wk(X_ref, km_ref.labels_, k))

    log_wkb_mean = float(np.mean(log_wkb_list))
    sdk = float(np.std(log_wkb_list))
    sk = sdk * np.sqrt(1 + 1 / B)

    gap_rows.append({
        "k": k, "log_Wk": round(log_wk, 4),
        "log_Wk_referensi_rata2": round(log_wkb_mean, 4),
        "gap": round(log_wkb_mean - log_wk, 4),
        "sk": round(float(sk), 4),
    })
    print(f"  Gap Statistic k={k}: Gap={gap_rows[-1]['gap']:.4f}  sk={sk:.4f}")

df_gapstat = pd.DataFrame(gap_rows)
df_gapstat.to_csv(OUT_DIR / "gap_statistic_per_k.csv", index=False)

# Aturan Tibshirani: pilih k terkecil dgn Gap(k) >= Gap(k+1) - s_{k+1}
k_gap_statistic = None
for i in range(len(df_gapstat) - 1):
    gap_k = df_gapstat.loc[i, "gap"]
    gap_k1 = df_gapstat.loc[i + 1, "gap"]
    sk1 = df_gapstat.loc[i + 1, "sk"]
    if gap_k >= gap_k1 - sk1:
        k_gap_statistic = int(df_gapstat.loc[i, "k"])
        break
if k_gap_statistic is None:
    k_gap_statistic = int(df_gapstat.loc[df_gapstat["gap"].idxmax(), "k"])

print(f"\n>>> k disarankan dari GAP STATISTIC: k = {k_gap_statistic}")

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(df_gapstat["k"], df_gapstat["gap"], yerr=df_gapstat["sk"],
            fmt="o-", capsize=4, color="darkorange")
ax.axvline(k_gap_statistic, color="red", linestyle="--",
           label=f"k terpilih = {k_gap_statistic}")
ax.set_xlabel("k")
ax.set_ylabel("Gap Statistic")
ax.set_title("Gap Statistic per k (Tibshirani et al., 2001)", fontweight="bold")
ax.set_xticks(list(K_RANGE_GAP))
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "gap_statistic_per_k.png", dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════════════
# KESIMPULAN GABUNGAN
# ══════════════════════════════════════════════════════════════════════════
kesimpulan = {
    "catatan": (
        "Dua metode penentuan k yang DIKLAIM/disarankan tapi belum pernah "
        "benar-benar dijalankan di NB03/NB04 atau eksperimen sebelumnya."
    ),
    "k_dari_gap_dendrogram_nyata": k_dendrogram,
    "k_dari_gap_statistic": k_gap_statistic,
    "k_dari_silhouette_kmeans_sebelumnya": 2,   # dari eksperimen/01_pemilihan_k_optimal.py
    "k_dari_silhouette_hierarchical_sebelumnya": 2,
    "k_yang_dipakai_di_notebook_resmi": 3,
    "ringkasan": (
        f"Gap dendrogram nyata -> k={k_dendrogram}. "
        f"Gap Statistic -> k={k_gap_statistic}. "
        "Bandingkan dengan k=2 (Silhouette, eksperimen 01) dan k=3 (notebook resmi, hardcoded)."
    ),
}
with open(OUT_DIR / "kesimpulan_dendrogram_gap.json", "w", encoding="utf-8") as f:
    json.dump(kesimpulan, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("KESIMPULAN GABUNGAN:")
print("=" * 60)
print(json.dumps(kesimpulan, indent=2, ensure_ascii=False))
print(f"\nSemua output disimpan di: {OUT_DIR}")
