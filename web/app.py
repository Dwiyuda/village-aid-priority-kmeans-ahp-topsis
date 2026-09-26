"""
SPK Penentuan Prioritas Desa Penerima Bantuan Sosial — Kabupaten Lahat.

Clustering dijalankan terpisah di notebook; web ini menangani tahap
pendukung keputusan: pembobotan AHP, perankingan TOPSIS/SAW, dan pelaporan.
"""
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from flask import (Flask, Response, flash, redirect, render_template,
                   request, url_for)

import spk

app = Flask(__name__)
app.secret_key = "spk-lahat"

BASE = Path(__file__).resolve().parent

# Satu pengguna, jadi status bobot cukup disimpan di memori proses.
DF = spk.muat_data()
STATE = {
    "matriks": np.array(spk.MATRIKS_RISET, dtype=float),
    "preset": "riset",
}

# CR & nama tiap preset, dipakai untuk melabeli tombol di halaman AHP.
PRESET_CR = spk.cr_semua_preset()
NAMA_PRESET = {k: nm for _, items in spk.PRESET_GRUP for k, nm, _ in items}


def hasil():
    return spk.hitung_semua(DF, STATE["matriks"])


def metrik_clustering():
    out = {}
    for nama, berkas in [("kmeans", "metrics_kmeans.json"),
                         ("hier", "metrics_hierarchical.json")]:
        with open(BASE / "data" / berkas, encoding="utf-8") as f:
            out[nama] = json.load(f)
    return out


@app.context_processor
def nav():
    return {"halaman": request.endpoint}


@app.route("/")
def dashboard():
    out, info = hasil()
    zona = out["zona"].value_counts()
    return render_template(
        "dashboard.html", info=info, out=out,
        n_desa=len(out), n_kec=out["KECAMATAN"].nunique(),
        zona={k: int(zona.get(k, 0)) for k in
              ["Kemiskinan Tinggi", "Kemiskinan Sedang", "Kemiskinan Rendah"]},
        masalah=out["jenis_masalah"].value_counts().to_dict(),
        top=out.head(5).to_dict("records"),
        per_kec=(out.groupby("KECAMATAN")["skor_topsis"].mean()
                 .sort_values(ascending=False).head(8).round(4).to_dict()),
        labels=spk.LABEL,
    )


@app.route("/ahp", methods=["GET", "POST"])
def ahp():
    if request.method == "POST":
        aksi = request.form.get("aksi", "simpan")
        if aksi in spk.PRESET:
            STATE["matriks"] = spk.matriks_preset(aksi)
            STATE["preset"] = aksi
            flash(f"Pola '{NAMA_PRESET.get(aksi, aksi)}' diterapkan. "
                  f"CR = {PRESET_CR[aksi]:.4f}, peringkat sudah dihitung ulang.",
                  "success")
        else:
            nilai = [int(request.form.get(f"n{i}", 0)) for i in range(21)]
            A = spk.matriks_dari_nilai(nilai)
            _, _, _, cr = spk.hitung_ahp(A)
            if cr >= 0.10:
                flash(f"Penilaian belum konsisten (CR = {cr:.4f}). "
                      "Bobot tidak disimpan, perbaiki dulu perbandingannya.", "danger")
                return redirect(url_for("ahp"))
            STATE["matriks"] = A
            STATE["preset"] = "manual"
            flash(f"Bobot tersimpan. CR = {cr:.4f}, peringkat sudah dihitung ulang.",
                  "success")
        return redirect(url_for("ahp"))

    _, info = hasil()
    pasangan = []
    k = 0
    n = len(spk.KRITERIA)
    for i in range(n):
        for j in range(i + 1, n):
            pasangan.append({"idx": k, "i": i, "j": j,
                             "kiri": spk.LABEL[i], "kanan": spk.LABEL[j]})
            k += 1
    return render_template(
        "ahp.html", info=info, labels=spk.LABEL,
        nilai=spk.nilai_dari_matriks(STATE["matriks"]),
        pasangan=pasangan, preset=STATE["preset"],
        preset_grup=spk.PRESET_GRUP, preset_cr=PRESET_CR,
        nama_preset=NAMA_PRESET.get(STATE["preset"], "Penilaian manual"),
        grup=[{"induk": spk.LABEL[i],
               "items": [p for p in pasangan if p["i"] == i]} for i in range(n - 1)],
    )


@app.route("/ranking")
def ranking():
    out, info = hasil()
    kec = request.args.get("kec", "")
    zona = request.args.get("zona", "")
    cari = request.args.get("cari", "").strip().lower()
    d = out
    if kec:
        d = d[d["KECAMATAN"] == kec]
    if zona:
        d = d[d["zona"] == zona]
    if cari:
        d = d[d["KELURAHAN"].str.lower().str.contains(cari)]
    return render_template(
        "ranking.html", info=info, rows=d.to_dict("records"), total=len(d),
        kecamatan=sorted(out["KECAMATAN"].unique()),
        f={"kec": kec, "zona": zona, "cari": request.args.get("cari", "")},
    )


@app.route("/desa/<kec>/<desa>")
def detail(kec, desa):
    out, info = hasil()
    baris = out[(out["KECAMATAN"] == kec) & (out["KELURAHAN"] == desa)]
    if baris.empty:
        flash("Desa tidak ditemukan.", "danger")
        return redirect(url_for("ranking"))
    r = baris.iloc[0].to_dict()
    return render_template("detail.html", r=r, info=info, total=len(out),
                           kriteria=spk.KRITERIA, labels=spk.LABEL)


@app.route("/clustering")
def clustering():
    out, _ = hasil()
    profil = (out.groupby("zona")[spk.KRITERIA + ["pct_nonaktif"]]
              .mean().round(2).reset_index().to_dict("records"))
    return render_template("clustering.html", m=metrik_clustering(),
                           profil=profil, labels=spk.LABEL,
                           jumlah=out["zona"].value_counts().to_dict())


@app.route("/perbandingan")
def perbandingan():
    out, info = hasil()
    beda = out.assign(selisih=(out["rank_topsis"] - out["rank_saw"]).abs())
    return render_template(
        "perbandingan.html", info=info,
        top20_sama=len(set(out.nsmallest(20, "rank_topsis")["KELURAHAN"]) &
                       set(out.nsmallest(20, "rank_saw")["KELURAHAN"])),
        cv_topsis=round(out["skor_topsis"].std() / out["skor_topsis"].mean(), 4),
        cv_saw=round(out["skor_saw"].std() / out["skor_saw"].mean(), 4),
        rows=out.head(15).to_dict("records"),
        beda=beda.nlargest(5, "selisih").to_dict("records"),
    )


@app.route("/ekspor/csv")
def ekspor_csv():
    out, _ = hasil()
    kec, zona = request.args.get("kec", ""), request.args.get("zona", "")
    if kec:
        out = out[out["KECAMATAN"] == kec]
    if zona:
        out = out[out["zona"] == zona]
    kolom = ["rank_topsis", "KECAMATAN", "KELURAHAN", "zona"] + spk.KRITERIA + \
            ["skor_topsis", "rank_saw", "skor_saw", "jenis_masalah", "rekomendasi"]
    buf = io.StringIO()
    out[kolom].to_csv(buf, index=False)
    return Response(
        buf.getvalue(), mimetype="text/csv",
        headers={"Content-Disposition":
                 "attachment; filename=prioritas_desa_lahat.csv"})


@app.route("/ekspor/pdf")
def ekspor_pdf():
    from laporan_pdf import buat_pdf
    out, info = hasil()
    return Response(
        buat_pdf(out, info), mimetype="application/pdf",
        headers={"Content-Disposition":
                 "attachment; filename=laporan_prioritas_desa.pdf"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
