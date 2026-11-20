# Evaluasi Penjelasan Gemini

Dibuat oleh `python -m scripts.eval_gemini --n 30` pada 20-11-2026 23:04. Model `gemini-3.5-flash-lite`, temperature 0,2. Data per kasus: [evaluasi_gemini_hasil.jsonl](evaluasi_gemini_hasil.jsonl).

Yang diukur adalah **output mentah Gemini sebelum fallback**: di aplikasi, penjelasan yang tidak lolos validasi otomatis diganti template, sehingga pengguna tidak pernah melihat angka yang salah.

| Ukuran | Hasil | Target |
|---|---|---|
| Akurasi angka (cocok / semua angka yang ditulis) | **100.0%** (233/233) | ≥ 95% |
| Penjelasan valid penuh (semua angka cocok + menyebut peringkat 1) | 21/21 (100%) | – |
| Menyebut vendor peringkat 1 | 21/21 | – |
| Panggilan gagal (kuota/jaringan/JSON) | 7/28 | – |
| Kasus waktu rawan (hari besar/musim ramai) yang valid | 13/13 | – |
| Waktu respons median | 10.7 detik | – |

**Sampel.** Kebutuhan diambil acak (seed 7) dari transaksi periode test (Apr–Sep 2026), ditambah beberapa tanggal waktu rawan 2027; preset diputar Seimbang/Termurah/Tercepat.

**Angka yang tidak cocok:** tidak ada.
