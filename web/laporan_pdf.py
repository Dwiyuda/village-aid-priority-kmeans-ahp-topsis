"""Penyusun laporan PDF. Grafik dibuat ulang memakai bobot yang sedang aktif."""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

import spk

BIRU = "#1E3A8A"
WARNA_ZONA = {"Kemiskinan Tinggi": "#EF4444",
              "Kemiskinan Sedang": "#F59E0B",
              "Kemiskinan Rendah": "#10B981"}


def _grafik_top15(out):
    d = out.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(d["KECAMATAN"] + " - " + d["KELURAHAN"], d["skor_topsis"],
            color=[WARNA_ZONA.get(z, BIRU) for z in d["zona"]])
    ax.set_xlabel("Skor prioritas (TOPSIS)")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _simpan(fig)


def _grafik_kecamatan(out):
    k = out.groupby("KECAMATAN")["skor_topsis"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.barh(k.index, k.values, color=BIRU)
    ax.set_xlabel("Rata-rata skor prioritas desa")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _simpan(fig)


def _simpan(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf


class Laporan(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120)
        self.cell(0, 6, "SPK Prioritas Desa Penerima Bantuan Sosial - Kabupaten Lahat",
                  align="L")
        self.ln(8)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120)
        self.cell(0, 6, f"Halaman {self.page_no()}", align="C")

    def judul(self, teks):
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(15, 23, 42)
        self.cell(0, 8, teks)
        self.ln(9)


def buat_pdf(out, info):
    pdf = Laporan()
    pdf.set_auto_page_break(auto=True, margin=16)

    # ── Sampul ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.ln(38)
    pdf.set_font("Helvetica", "B", 17)
    pdf.set_text_color(30, 58, 138)
    pdf.multi_cell(0, 9, "Laporan Prioritas Desa\nPenerima Bantuan Sosial", align="C")
    pdf.ln(4)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(100)
    pdf.multi_cell(0, 7, "Kabupaten Lahat, Sumatera Selatan", align="C")
    pdf.ln(18)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(15, 23, 42)
    zona = out["zona"].value_counts()
    ringkas = [
        ("Jumlah desa dianalisis", f"{len(out)} desa"),
        ("Jumlah kecamatan", f"{out['KECAMATAN'].nunique()} kecamatan"),
        ("Zona kemiskinan tinggi", f"{zona.get('Kemiskinan Tinggi', 0)} desa"),
        ("Zona kemiskinan sedang", f"{zona.get('Kemiskinan Sedang', 0)} desa"),
        ("Zona kemiskinan rendah", f"{zona.get('Kemiskinan Rendah', 0)} desa"),
        ("Metode perankingan", "TOPSIS, pembobotan AHP"),
        ("Rasio konsistensi (CR)", f"{info['cr']:.4f}"),
        ("Tanggal cetak", datetime.now().strftime("%d %B %Y, %H:%M")),
    ]
    for k, v in ringkas:
        pdf.cell(85, 8, k, border="B")
        pdf.cell(0, 8, v, border="B", align="R")
        pdf.ln(8)

    # ── Bobot kriteria ────────────────────────────────────────────────────
    pdf.add_page()
    pdf.judul("Bobot kriteria yang dipakai")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100)
    pdf.multi_cell(0, 5,
                   "Bobot dihitung dengan Analytic Hierarchy Process dari matriks perbandingan "
                   "berpasangan skala Saaty. Penilaian dinyatakan konsisten apabila rasio "
                   "konsistensi berada di bawah 0,10.")
    pdf.ln(4)
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(90, 7, "Kriteria", border="B")
    pdf.cell(35, 7, "Jenis", border="B")
    pdf.cell(0, 7, "Bobot", border="B", align="R")
    pdf.ln(7)
    pdf.set_font("Helvetica", "", 9)
    for i, label in enumerate(spk.LABEL):
        jenis = "Cost" if not spk.IS_BENEFIT[i] else "Benefit"
        pdf.cell(90, 7, label, border="B")
        pdf.cell(35, 7, jenis, border="B")
        pdf.cell(0, 7, f"{info['bobot'][i] * 100:.2f}%", border="B", align="R")
        pdf.ln(7)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100)
    pdf.multi_cell(0, 5,
                   f"Lambda maks {info['lambda_max']:.4f}  |  CI {info['ci']:.4f}  |  "
                   f"CR {info['cr']:.4f}  |  "
                   f"{'Konsisten' if info['konsisten'] else 'Tidak konsisten'}\n"
                   f"Korelasi peringkat TOPSIS dan SAW {info['korelasi']:.4f}")

    # ── Grafik ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.judul("Lima belas desa prioritas teratas")
    pdf.image(_grafik_top15(out), w=178)

    pdf.add_page()
    pdf.judul("Prioritas rata-rata per kecamatan")
    pdf.image(_grafik_kecamatan(out), w=178)

    # ── Tabel peringkat ───────────────────────────────────────────────────
    pdf.add_page()
    pdf.judul("Peringkat prioritas desa")
    lebar = [12, 44, 36, 26, 18, 40]
    kepala = ["#", "Desa", "Kecamatan", "Zona", "Skor", "Rekomendasi"]

    def baris_kepala():
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(15, 23, 42)
        for w, t in zip(lebar, kepala):
            pdf.cell(w, 7, t, border="B")
        pdf.ln(7)
        pdf.set_font("Helvetica", "", 7.5)

    baris_kepala()
    singkat = {"Kemiskinan Tinggi": "Tinggi", "Kemiskinan Sedang": "Sedang",
               "Kemiskinan Rendah": "Rendah"}
    for _, r in out.iterrows():
        if pdf.get_y() > 262:
            pdf.add_page()
            pdf.judul("Peringkat prioritas desa (lanjutan)")
            baris_kepala()
        rek = r["rekomendasi"]
        rek = rek[:34] + "..." if len(rek) > 37 else rek
        nilai = [str(r["rank_topsis"]), str(r["KELURAHAN"])[:24],
                 str(r["KECAMATAN"])[:19], singkat.get(r["zona"], ""),
                 f"{r['skor_topsis']:.4f}", rek]
        for w, t in zip(lebar, nilai):
            pdf.cell(w, 6, t, border="B")
        pdf.ln(6)

    return bytes(pdf.output())
