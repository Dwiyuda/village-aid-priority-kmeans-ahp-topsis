"""
Eksperimen: Pengaruh PCA sebagai Reduksi Dimensi Sebelum Clustering

Membandingkan performa K-Means & Hierarchical (Ward) TANPA PCA (8 fitur
scaled penuh) vs DENGAN PCA (fitur direduksi ke n komponen yang menjelaskan
>=90% variance), memakai k yang sama untuk perbandingan adil.

Merujuk pada Izzuddin & Wijayanto (2024) yang membuktikan Ward+PCA
mengungguli Ward tanpa PCA (silhouette 0.48 vs tanpa PCA) pada data
kemiskinan level provinsi -- di sini diuji apakah pola serupa berlaku
pada level desa/kelurahan (data DTSEN Kabupaten Lahat, sudah dibetulkan).

Output disimpan terpisah di output/eksperimen_pca/.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import load_dtsen, buat_fitur_proporsi, FEATURE_COLS, DATA_DIR

OUT_DIR = ROOT / "output" / "eksperimen_pca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── Load & preprocessing (identik dengan NB01/NB02, data sudah benar) ─────────
df = load_dtsen(DATA_DIR / "20_JULI_2026-_Rekap_DTSEN_Kabupaten_Lahat.xls")
df_feat = buat_fitur_proporsi(df)
X = StandardScaler().fit_transform(df_feat[FEATURE_COLS].values)

# ── PCA: cari jumlah komponen yang menjelaskan >=90% variance ─────────────────
pca_full = PCA(random_state=42).fit(X)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.argmax(cum_var >= 0.90) + 1)

print(f"Variance dijelaskan per komponen: {np.round(pca_full.explained_variance_ratio_, 4)}")
print(f"Kumulatif variance: {np.round(cum_var, 4)}")
print(f"Jumlah komponen dipilih (>=90% variance): {n_components}")

pca = PCA(n_components=n_components, random_state=42)
X_pca = pca.fit_transform(X)

# Plot scree (variance explained)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(range(1, len(cum_var) + 1), pca_full.explained_variance_ratio_, alpha=0.6, label="Per komponen")
ax.plot(range(1, len(cum_var) + 1), cum_var, "r-o", markersize=5, label="Kumulatif")
ax.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="Threshold 90%")
ax.axvline(n_components, color="green", linestyle=":", linewidth=1.5)
ax.set_xlabel("Komponen ke-")
ax.set_ylabel("Proporsi Variance Dijelaskan")
ax.set_title("Scree Plot PCA — 8 Fitur Proporsi DTSEN", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "scree_plot_pca.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Bandingkan TANPA PCA vs DENGAN PCA, untuk k=2 s.d. 5 ──────────────────────
rows = []
for k in range(2, 6):
    for label, data in [("Tanpa PCA", X), ("Dengan PCA", X_pca)]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = km.fit_predict(data)
        rows.append({
            "algoritma": "K-Means", "fitur": label, "k": k,
            "silhouette": round(float(silhouette_score(data, km_labels)), 6),
            "davies_bouldin": round(float(davies_bouldin_score(data, km_labels)), 6),
            "calinski_harabasz": round(float(calinski_harabasz_score(data, km_labels)), 6),
        })

        hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hc_labels = hc.fit_predict(data)
        rows.append({
            "algoritma": "Hierarchical", "fitur": label, "k": k,
            "silhouette": round(float(silhouette_score(data, hc_labels)), 6),
            "davies_bouldin": round(float(davies_bouldin_score(data, hc_labels)), 6),
            "calinski_harabasz": round(float(calinski_harabasz_score(data, hc_labels)), 6),
        })

hasil = pd.DataFrame(rows)
hasil.to_csv(OUT_DIR / "perbandingan_tanpa_vs_dengan_pca.csv", index=False)
print("\nTabel perbandingan Tanpa PCA vs Dengan PCA:")
print(hasil.to_string(index=False))

# ── Plot perbandingan silhouette per k, tanpa vs dengan PCA ───────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, algo in zip(axes, ["K-Means", "Hierarchical"]):
    sub = hasil[hasil["algoritma"] == algo]
    for label, marker, color in [("Tanpa PCA", "o-", "tab:blue"), ("Dengan PCA", "s--", "tab:orange")]:
        s = sub[sub["fitur"] == label]
        ax.plot(s["k"], s["silhouette"], marker, color=color, label=label, markersize=6)
    ax.set_title(f"Silhouette Score — {algo}", fontweight="bold")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "perbandingan_silhouette_pca.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Kesimpulan otomatis ────────────────────────────────────────────────────────
best_row = hasil.loc[hasil["silhouette"].idxmax()]
tanpa_pca_best = hasil[hasil["fitur"] == "Tanpa PCA"]["silhouette"].max()
dengan_pca_best = hasil[hasil["fitur"] == "Dengan PCA"]["silhouette"].max()

kesimpulan = {
    "n_components_pca": n_components,
    "variance_dijelaskan_pca": round(float(cum_var[n_components - 1]), 4),
    "silhouette_terbaik_tanpa_pca": round(float(tanpa_pca_best), 6),
    "silhouette_terbaik_dengan_pca": round(float(dengan_pca_best), 6),
    "pca_meningkatkan_hasil": bool(dengan_pca_best > tanpa_pca_best),
    "kombinasi_terbaik_keseluruhan": {
        "algoritma": best_row["algoritma"],
        "fitur": best_row["fitur"],
        "k": int(best_row["k"]),
        "silhouette": float(best_row["silhouette"]),
    },
}
with open(OUT_DIR / "kesimpulan_pca.json", "w", encoding="utf-8") as f:
    json.dump(kesimpulan, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("KESIMPULAN EKSPERIMEN PCA:")
print("=" * 60)
print(json.dumps(kesimpulan, indent=2, ensure_ascii=False))
print(f"\nSemua output disimpan di: {OUT_DIR}")
