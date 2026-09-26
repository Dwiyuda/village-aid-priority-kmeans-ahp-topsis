# SPK Prioritas Desa Penerima Bantuan Sosial — Kabupaten Lahat

Sistem pendukung keputusan untuk menentukan urutan prioritas desa penerima bantuan sosial
dari data DTSEN, dengan bobot kriteria yang dapat disesuaikan pengambil keputusan.

## Menjalankan

```bash
cd "web"
python app.py
```

Buka `http://localhost:5000` di browser. Hentikan dengan `Ctrl + C`.

Kalau port 5000 dipakai aplikasi lain, ubah baris terakhir `app.py` jadi `port=5001`.

## Sekali saja, bila di komputer baru

```bash
python -m pip install flask fpdf2 pandas numpy matplotlib
```

## Halaman

| Menu | Isi |
|---|---|
| Dashboard | Ringkasan 371 desa, distribusi zona, lima desa teratas, bobot aktif |
| Hasil Clustering | Perbandingan K-Means dan Hierarchical Ward, profil tiap zona |
| Pembobotan AHP | 21 slider perbandingan berpasangan, uji konsistensi langsung |
| Peringkat Prioritas | Tabel 371 desa, penyaring kecamatan dan zona, pencarian nama |
| TOPSIS vs SAW | Korelasi peringkat, kesepakatan 20 teratas, daya pemisah |
| Unduh CSV / PDF | Ekspor hasil, mengikuti bobot yang sedang aktif |

## Urutan demo yang disarankan

1. Buka **Dashboard** — tunjukkan 371 desa, tiga zona, dan CR 0,0100
2. Masuk **Pembobotan AHP** — geser satu slider, tunjukkan CR berubah seketika
3. Geser sampai CR melewati 0,10 — badge berubah merah, sistem menolak menyimpan
4. Kembalikan ke preset **Bobot penelitian**, tekan **Simpan dan hitung ulang peringkat**
5. Buka **Peringkat Prioritas** — tunjukkan urutan 371 desa ikut berubah
6. Klik satu desa untuk melihat profil dan rekomendasinya
7. Unduh PDF sebagai penutup

Langkah 2 dan 3 adalah bukti bahwa ini sistem pendukung keputusan, bukan penampil hasil.

## Catatan

Tahap clustering dijalankan terpisah di notebook (`analysis`), hasilnya disalin ke
folder `data/`. Web ini menangani tahap pendukung keputusan: pembobotan AHP, perankingan
TOPSIS dan SAW, serta pelaporan. Bobot awal memakai matriks hasil penelitian dengan
CR 0,0100, sehingga saat pertama dibuka sistem langsung menampilkan hasil yang sama
dengan yang tertulis di laporan.

Data yang dipakai: DTSEN Kabupaten Lahat versi 20 Juli 2026, 371 desa di 24 kecamatan.

## Berkas

```
app.py           rute Flask
spk.py           mesin AHP, TOPSIS, SAW
laporan_pdf.py   penyusun laporan PDF dan grafiknya
data/            hasil clustering dari notebook
templates/       enam halaman
static/css/      tema
```
