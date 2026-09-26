"""
Eksperimen: Pemilihan k Optimal Berbasis Data (bukan visual/manual)

Menguji K-Means dan Hierarchical (Ward) untuk k=2 s.d. 10, menghitung Elbow
(inertia), Silhouette, Davies-Bouldin, dan Calinski-Harabasz untuk tiap k,
lalu MEMILIH k secara otomatis dari Silhouette Score tertinggi -- bukan
dipilih manual seperti di NB03/NB04 (K_OPTIMAL = 3 hardcoded).

Output disimpan terpisah di output/eksperimen_pemilihan_k/ supaya tidak
menimpa hasil resmi di output/results/.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import load_dtsen, buat_fitur_proporsi, FEATURE_COLS, DATA_DIR

OUT_DIR = ROOT / "output" / "eksperimen_pemilihan_k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

matplotlib.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── Load & preprocessing (identik dengan NB01/NB02, data sudah benar) ─────────
df = load_dtsen(DATA_DIR / "20_JULI_2026-_Rekap_DTSEN_Kabupaten_Lahat.xls")
df_feat = buat_fitur_proporsi(df)
X = StandardScaler().fit_transform(df_feat[FEATURE_COLS].values)

K_RANGE = range(2, 11)
rows = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    rows.append({
        "algoritma": "K-Means", "k": k,
        "inertia": round(float(km.inertia_), 4),
        "silhouette": round(float(silhouette_score(X, km_labels)), 6),
        "davies_bouldin": round(float(davies_bouldin_score(X, km_labels)), 6),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, km_labels)), 6),
    })

    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    hc_labels = hc.fit_predict(X)
    rows.append({
        "algoritma": "Hierarchical", "k": k,
        "inertia": None,
        "silhouette": round(float(silhouette_score(X, hc_labels)), 6),
        "davies_bouldin": round(float(davies_bouldin_score(X, hc_labels)), 6),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, hc_labels)), 6),
    })

hasil = pd.DataFrame(rows)
hasil.to_csv(OUT_DIR / "metrik_per_k.csv", index=False)
print("Tabel metrik per k:")
print(hasil.to_string(index=False))

# ── Plot Elbow (K-Means) & Silhouette per k (kedua algoritma) ─────────────────
km_rows = hasil[hasil["algoritma"] == "K-Means"]
hc_rows = hasil[hasil["algoritma"] == "Hierarchical"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(km_rows["k"], km_rows["inertia"], "b-o", markersize=6)
axes[0].set_title("Elbow Method (K-Means)", fontweight="bold")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")

axes[1].plot(km_rows["k"], km_rows["silhouette"], "g-s", markersize=6, label="K-Means")
axes[1].plot(hc_rows["k"], hc_rows["silhouette"], "m-^", markersize=6, label="Hierarchical")
axes[1].set_title("Silhouette Score per k", fontweight="bold")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "elbow_silhouette_per_k.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Keputusan OTOMATIS: k dengan Silhouette tertinggi per algoritma ───────────
best_km = km_rows.loc[km_rows["silhouette"].idxmax()]
best_hc = hc_rows.loc[hc_rows["silhouette"].idxmax()]

kesimpulan = {
    "catatan": "k dipilih otomatis dari Silhouette Score tertinggi (bukan manual/visual)",
    "k_optimal_kmeans": int(best_km["k"]),
    "silhouette_kmeans_terbaik": float(best_km["silhouette"]),
    "k_optimal_hierarchical": int(best_hc["k"]),
    "silhouette_hierarchical_terbaik": float(best_hc["silhouette"]),
    "k_yang_dipakai_di_notebook_resmi": 3,
    "catatan_perbandingan": (
        "Bandingkan k_optimal di atas dengan k=3 yang dipakai di NB03/NB04. "
        "Jika berbeda, k=3 di notebook resmi adalah keputusan manual "
        "(pertimbangan interpretability), bukan hasil murni data."
    ),
}
with open(OUT_DIR / "kesimpulan_k_optimal.json", "w", encoding="utf-8") as f:
    json.dump(kesimpulan, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("KESIMPULAN OTOMATIS (berdasarkan Silhouette tertinggi):")
print("=" * 60)
print(json.dumps(kesimpulan, indent=2, ensure_ascii=False))
print(f"\nSemua output disimpan di: {OUT_DIR}")
