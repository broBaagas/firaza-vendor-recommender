# Asumsi Data dan Kamus Data

Firaza tidak menyediakan data vendor maupun histori procurement (lihat dokumen use case). Karena itu seluruh data dibuat **sintetis** oleh [`generator/generate.py`](../generator/generate.py) dengan numpy/pandas dan seed tetap (`SEED = 42`). Data tidak ditulis oleh LLM. Menjalankan `python -m generator.generate` selalu menghasilkan file yang sama persis (diuji di `tests/test_generator.py`).

## Ringkasan

| Item | Nilai |
|---|---|
| Periode transaksi | 1 Okt 2024 – 30 Sep 2026 (24 bulan) |
| Vendor | 30 (tekstil 9, packaging 10, jasa 6, aksesoris 5) |
| Material | 15 |
| Penawaran (vendor × material) | 109 |
| Transaksi | 6.075 |
| Tingkat berhasil | 78,0% |
| Split evaluasi | train bulan 1–18 (s.d. Mar 2026), test bulan 19–24 (Apr–Sep 2026) |

## Kamus data

### `vendors.csv`
| Kolom | Arti |
|---|---|
| `vendor_id` | ID vendor (V01–V30) |
| `nama_vendor` | Nama fiktif, `<Kategori> <Huruf>` (mis. "Tekstil C"), sengaja tidak menyerupai usaha nyata |
| `kategori` | tekstil / packaging / jasa / aksesoris |
| `kota` | Kota asal (fiktif tetapi masuk akal untuk kategorinya, mis. Pekalongan untuk tekstil) |
| `tanggal_bergabung` | Sejak kapan vendor bisa dipilih. 4 vendor bergabung belakangan (bulan ke-8, 14, 20, 23) untuk memunculkan kasus "data terbatas" |

### `materials.csv`
| Kolom | Arti |
|---|---|
| `material_id`, `nama_material`, `kategori`, `satuan` | Identitas material (pcs/roll) |
| `harga_ref_min`, `harga_ref_max` | Rentang harga referensi per satuan (Rp). Lihat bagian **Kalibrasi harga** |
| `lead_time_min`, `lead_time_max` | Rentang lead time wajar (hari) |
| `qty_median` | Ukuran pesanan tipikal |

### `offers.csv`
Daftar harga vendor saat ini (akhir periode data).
| Kolom | Arti |
|---|---|
| `offer_id` | ID penawaran |
| `vendor_id`, `material_id` | Pasangan vendor–material |
| `harga_satuan` | Harga per satuan (Rp) |
| `lead_time_janji` | Lead time yang dijanjikan vendor (hari) |
| `moq` | Minimum order quantity |

### `transactions.csv`
| Kolom | Arti |
|---|---|
| `transaksi_id` | ID transaksi |
| `tanggal_pesan` | Tanggal pesanan dibuat |
| `vendor_id`, `material_id`, `offer_id` | Vendor dan penawaran yang dipilih |
| `qty`, `harga_satuan`, `total_harga` | Jumlah dan harga pada saat transaksi |
| `lead_time_janji` | Janji lead time saat itu |
| `hari_tersedia` | Waktu yang dimiliki Firaza sampai barang harus ada (hari) |
| `deadline` | `tanggal_pesan + hari_tersedia` |
| `tanggal_datang` | Tanggal barang benar-benar datang |
| `qty_cacat` | Jumlah unit cacat saat diterima |

### `vendors_hidden.csv` (tidak di-commit, dibuat ulang oleh generator)
Sifat "sebenarnya" vendor yang dipakai generator. **Hanya untuk validasi generator dan evaluasi**, tidak pernah dibaca oleh `core/`, model, atau aplikasi.
| Kolom | Arti |
|---|---|
| `true_reliability` | Peluang dasar tepat janji |
| `true_defect_rate` | Rata-rata tingkat cacat |
| `price_factor` | Faktor harga vendor terhadap harga referensi |
| `peak_sensitivity` | Seberapa besar kinerja turun saat musim ramai (0–1) |
| `capacity_factor` | Kapasitas pesanan sebelum kinerja turun (× qty tipikal) |
| `drift_logit` | Perubahan keandalan selama 24 bulan (membaik/memburuk) |

## Mekanisme generator

1. **Sifat vendor.** Setiap vendor punya "kompetensi" laten. Vendor yang lebih kompeten cenderung lebih andal, lebih sedikit cacat, dan **sedikit lebih mahal**. Akibatnya pilihan termurah tidak selalu terbaik, sesuai problem di use case. Hubungan ini diberi derau supaya tidak sempurna.
2. **Penawaran.** Setiap material ditawarkan 5–9 vendor dari kategori yang sesuai. Material aksesoris juga ditawarkan sebagian vendor packaging. Harga penawaran = harga referensi × `price_factor` × derau, dengan premi ±2% per hari untuk vendor yang menjanjikan lebih cepat. MOQ = 10–100% dari qty tipikal.
3. **Permintaan.** Jumlah pesanan per hari mengikuti distribusi Poisson dengan pertumbuhan bisnis 30% selama 24 bulan, pengali ×1,8 di musim ramai, dan ×0,5 pada hari Minggu. Material dipilih berdasarkan popularitas, qty mengikuti distribusi lognormal (×1,4 saat musim ramai), dan `hari_tersedia` = lead time maksimum material −3 s.d. +11 hari (2 hari lebih ketat saat musim ramai).
4. **Pemilihan vendor historis (kebijakan owner).** Dari penawaran yang memenuhi syarat (vendor sudah bergabung, lead time ≤ hari tersedia, MOQ ≤ qty), owner memilih dengan utilitas `−1,2·harga_z + 0,6·persepsi_keandalan + derau Gumbel`. Owner cenderung memilih yang murah dan punya kesan kasar tentang keandalan vendor. Jika tidak ada yang memenuhi syarat, owner mengambil vendor tercepat dan menaikkan qty ke MOQ.
5. **Hasil pengiriman.** Peluang tepat janji = sigmoid(logit keandalan + drift − penalti kelebihan kapasitas − penalti musim ramai × `peak_sensitivity`). Jika tepat, barang datang 0–2 hari lebih cepat dari janji. Jika telat, barang datang 1 + Poisson(λ) hari setelah janji, dengan λ lebih besar untuk vendor kurang andal dan saat musim ramai.
6. **Cacat.** `qty_cacat ~ Binomial(qty, p)`, dengan p = tingkat cacat vendor × (1 + 0,5·`peak_sensitivity` saat musim ramai) × derau lognormal per batch. Untuk satuan roll p dikali 0,25, karena qty roll kecil (±10) sehingga 1 roll cacat sudah berarti 10%.
7. **Harga transaksi.** Harga daftar saat ini disesuaikan mundur dengan inflasi 0,4% per bulan, naik 3% saat musim ramai, ditambah derau kecil.

## Musim ramai

Didefinisikan di [`core/seasons.py`](../core/seasons.py) dan dipakai generator maupun fitur. Tanggal hijriah adalah perkiraan.

| Periode | Rentang |
|---|---|
| Ramadan–Lebaran 2025 (order hampers mulai ±3 minggu sebelum Ramadan) | 8 Feb – 31 Mar 2025 |
| Musim nikah Syawal 2025 | 1 – 29 Apr 2025 |
| Musim nikah Dzulhijjah 2025 | 28 Mei – 25 Jun 2025 |
| Ramadan–Lebaran 2026 | 28 Jan – 20 Mar 2026 |
| Musim nikah Syawal 2026 | 21 Mar – 18 Apr 2026 |
| Musim nikah Dzulhijjah 2026 | 18 Mei – 15 Jun 2026 |

## Label dan fitur

- **Label** `berhasil = 1` jika `tanggal_datang ≤ deadline` **dan** `qty_cacat / qty ≤ 3%`.
- **Anti-leakage.** Rekam jejak vendor untuk pesanan tanggal *t* hanya memakai transaksi vendor itu yang **sudah datang sebelum t** (`tanggal_datang < t`). Pesanan yang sudah dibuat tetapi barangnya belum datang hasilnya belum diketahui, jadi tidak ikut dihitung. Diuji di `tests/test_features.py`.
- **Penghalusan.** Rasio rekam jejak dihaluskan dengan prior tetap (berhasil 0,7, telat 1 hari, cacat 2%, kekuatan prior 2 transaksi) supaya vendor dengan sedikit transaksi tidak menghasilkan nilai ekstrem. Prior berupa konstanta, tidak dihitung dari data.
- **Harga sebagai fitur** dipakai dalam bentuk `harga_relatif` (harga ÷ titik tengah harga referensi material), karena harga absolut antarmaterial tidak sebanding (Rp300 s.d. Rp160.000). Harga rupiah tetap dipakai untuk saringan budget dan skor ranking.

## Kalibrasi harga

Rentang `harga_ref_min`/`harga_ref_max` di `MATERIALS` (generator) saat ini adalah **perkiraan awal** harga grosir/marketplace untuk produk sejenis dan **belum diverifikasi**. Sebelum project brief, setiap rentang perlu dicek di Shopee/Tokopedia, lalu tautan dan tanggal aksesnya dicatat di tabel ini:

| Material | Rentang (Rp) | Sumber | Diakses |
|---|---|---|---|
| Handuk mandi 70x140 cm | 35.000 – 65.000 | _belum_ | |
| Handuk muka 30x30 cm | 6.000 – 12.000 | _belum_ | |
| Sajadah polos dewasa | 40.000 – 80.000 | _belum_ | |
| Sajadah travel lipat | 20.000 – 40.000 | _belum_ | |
| Mukena katun dewasa | 85.000 – 160.000 | _belum_ | |
| Kotak hampers rigid box | 9.000 – 22.000 | _belum_ | |
| Kardus pengiriman | 2.000 – 5.000 | _belum_ | |
| Paper bag | 3.000 – 8.000 | _belum_ | |
| Tas spunbond | 2.500 – 6.000 | _belum_ | |
| Bubble wrap 125 cm x 50 m | 45.000 – 95.000 | _belum_ | |
| Plastik OPP | 300 – 1.000 | _belum_ | |
| Pita satin (roll) | 5.000 – 15.000 | _belum_ | |
| Kartu ucapan cetak | 500 – 2.000 | _belum_ | |
| Jasa bordir nama | 5.000 – 15.000 | _belum_ | |
| Jasa printing sajadah | 10.000 – 25.000 | _belum_ | |

Jika ada rentang yang diubah, jalankan ulang generator dan notebook EDA.

## Keterbatasan

- **Sintetis.** Pola di data (termasuk trade-off harga vs keandalan) adalah asumsi generator, bukan hasil observasi Firaza. Model yang bagus di data ini belum tentu sama bagusnya di data nyata. Nilainya ada pada pipeline yang siap dilatih ulang dengan data asli.
- **Bias seleksi.** Hasil pengiriman hanya teramati untuk vendor yang dipilih owner. Vendor yang jarang dipilih punya rekam jejak yang lebih tidak pasti.
- **Penyederhanaan.** Satu material per pesanan, tanpa pembatalan, retur, diskon volume, atau ongkos kirim. Kota vendor tidak memengaruhi lead time.
- **Pesanan di akhir periode.** Beberapa pesanan September 2026 punya `tanggal_datang` setelah 30 Sep 2026. Pesanan ini tetap dicatat. Fitur rekam jejak tidak terpengaruh karena hanya memakai barang yang sudah datang.
