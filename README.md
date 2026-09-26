# Village Aid Priority — K-Means + AHP-TOPSIS

Decision support system that ranks villages for social-assistance targeting.
It turns the national socio-economic registry (DTSEN) of a regency in South Sumatra,
Indonesia — **371 villages, 24 districts, ~142,800 households** — into a village-level
priority map in two stages:

1. **Clustering** groups villages by poverty profile (K-Means, compared with Hierarchical Ward).
2. **Multi-criteria ranking** orders villages inside the decision process: AHP derives the
   criteria weights, TOPSIS ranks the villages, and SAW is used as a cross-check.

Government targeting tools already work at household and regency level. Aid, however, is
decided in village meetings, and there was no village-level analysis. This project fills that gap.

## Results

| Item | Value |
|---|---|
| Villages analysed | 371 (1 excluded: no decile profile) |
| K-Means (k = 3) | Silhouette 0.1816 · Davies-Bouldin 1.8145 · Calinski-Harabasz 101.60 |
| Priority zones | High 136 · Medium 109 · Low 126 |
| AHP consistency ratio | **0.0100** (threshold 0.10) |
| Criteria | 7 (share of households in deciles 1–5, unranked households, etc.) |

## Web app

Flask app for the decision-making step. Weights can be adjusted live with 21 pairwise
comparison sliders; the consistency ratio is recomputed instantly, and the system refuses to
save weights once CR exceeds 0.10.

| Page | Content |
|---|---|
| Dashboard | Summary of 371 villages, zone distribution, top five villages, active weights |
| Clustering | K-Means vs Hierarchical Ward, cluster profiles |
| AHP weighting | 21 sliders, live consistency check |
| Priority ranking | Ranked table with district/zone filters and search |
| TOPSIS vs SAW | Rank correlation, top-20 agreement |
| Export | CSV and PDF reports following the active weights |

```bash
pip install -r requirements.txt
cd web
python app.py        # http://localhost:5000
```

Tested on Python 3.12: all web pages, CSV/PDF export, the three experiment scripts and the
seven notebooks run end to end and reproduce the numbers above.

## Analysis pipeline

Notebooks in `analysis/notebooks/` run in order:

| Notebook | Step |
|---|---|
| `NB01_EDA` | Exploratory analysis of the registry |
| `NB02_Preprocessing` | Cleaning, exclusion rules, 8 proportion features, Z-score |
| `NB03_KMeans` | Elbow, silhouette, final k = 3 |
| `NB04_Hierarchical` | Ward linkage, dendrogram, comparison with K-Means |
| `NB06_SPK_TOPSIS_SAW` | TOPSIS ranking, SAW cross-check |
| `NB07_AHP_Pembobotan` | Pairwise comparison matrix and consistency ratio |
| `NB08_Peta_Prioritas_Final` | Final village priority map |

`analysis/eksperimen/` holds the supporting experiments (optimal k, PCA, gap statistic).
Figures and result tables are in `analysis/output/`.

## Data

The registry recap is **aggregated per village** (household counts per welfare decile).
It contains no individual or household-level records.

## Stack

Python · pandas · scikit-learn · SciPy · Matplotlib · Flask · fpdf2

---

Built by **Dwi Yuda** · [dwiyuda.is-a.dev](https://dwiyuda.is-a.dev)
