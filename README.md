# E-Commerce Product Intelligence — Hybrid Recommendation System

Sistem rekomendasi produk hybrid yang dibangun di atas **E-Commerce Product Intelligence Dataset**, sebuah dataset relasional sintetis yang mensimulasikan 3,5 tahun aktivitas pelanggan retailer online skala menengah. Proyek ini menggabungkan tiga pendekatan rekomendasi — Collaborative Filtering, Content-Based Filtering, dan Session-Based Popularity — ke dalam satu pipeline hybrid yang siap dijalankan di Google Colab.

## Daftar isi

- [Tentang dataset](#tentang-dataset)
- [Arsitektur sistem](#arsitektur-sistem)
- [Struktur project](#struktur-project)
- [Cara menjalankan](#cara-menjalankan)
- [Hasil & evaluasi](#hasil--evaluasi)
- [Keterbatasan](#keterbatasan)
- [Roadmap](#roadmap)
- [Kontribusi](#kontribusi)
- [Lisensi](#lisensi)
- [Kolaborator](#kolaborator)

## Tentang dataset

Dataset terdiri dari 6 tabel relasional dengan total sekitar 133.000 baris dan 51 kolom, mencakup periode 1 Januari 2023 hingga 1 Juni 2026.

| Tabel | Jumlah baris | Deskripsi |
|---|---|---|
| `users.csv` | 10.000 | Profil dan demografi pelanggan |
| `products.csv` | 1.000 | Katalog produk dengan hierarki kategori |
| `sessions.csv` | 19.315 | Sesi browsing pengguna |
| `interactions.csv` | 100.000 | Event interaksi user-produk (6 tipe) |
| `purchases.csv` | 1.737 | Baris order pembelian |
| `reviews.csv` | 1.253 | Ulasan produk dengan teks NLP |

Karakteristik kunci yang relevan untuk sistem rekomendasi:

- Densitas matriks user-item hanya **1,0%**
- **30,6%** pengguna dan **54,8%** produk bersifat cold-start
- Konsentrasi Pareto: 20% produk menerima **84,8%** interaksi
- Conversion rate sesi sebesar **7,5%**

Dokumentasi lengkap skema, relasi foreign key, dan metodologi generasi data tersedia di file dokumentasi yang disertakan bersama dataset.

## Arsitektur sistem

Pipeline menggabungkan tiga modul rekomendasi dengan skema pembobotan adaptif:

| Komponen | Bobot dasar | Peran utama | Library |
|---|---|---|---|
| Collaborative Filtering (ALS) | 50% | Personalisasi untuk pengguna dengan riwayat | `implicit` |
| Content-Based (TF-IDF + cosine) | 30% | Mengatasi cold-start produk | `scikit-learn` |
| Session-Based (popularitas + konversi) | 20% | Mengatasi cold-start pengguna & konteks sesi | `pandas`, `numpy` |

Logika fallback otomatis menyesuaikan bobot saat salah satu komponen tidak tersedia:

- Pengguna cold-start → bobot CF dinolkan, dialihkan ke Content-Based dan Session
- Pengguna tanpa histori interaksi sama sekali → murni Session-Based
- Bobot selalu dinormalisasi ulang agar total tetap 1

Skor akhir setiap produk dihasilkan dari fusi tertimbang (`hybrid_score`), dengan kontribusi tiap komponen tetap dapat ditelusuri (`score_cf`, `score_cb`, `score_session`) untuk keperluan debugging dan analisis bisnis.

## Struktur project

```
.
├── hybrid_recommender_colab.py   # Pipeline utama: preprocessing, model, evaluasi
├── data/
│   ├── users.csv
│   ├── products.csv
│   ├── sessions.csv
│   ├── interactions.csv
│   ├── purchases.csv
│   └── reviews.csv
├── recommendations_sample.csv    # Contoh output rekomendasi
└── README.md
```

## Cara menjalankan

### 1. Persiapan di Google Colab

Buka notebook baru di [Google Colab](https://colab.research.google.com), lalu upload keenam file CSV dataset ke sesi runtime.

### 2. Install dependency

```python
!pip install implicit scikit-learn pandas numpy scipy tqdm --quiet
```

### 3. Jalankan pipeline

Copy seluruh isi `hybrid_recommender_colab.py` ke dalam cell-cell notebook secara berurutan, atau import sebagai modul:

```python
from hybrid_recommender_colab import hybrid_recommend

rekomendasi = hybrid_recommend(
    user_id="0000780a-2126-4e84-9622-42ce0ea9b17a",
    session_context={"device": "mobile", "referrer": "organic_search"},
    top_n=10,
)
print(rekomendasi)
```

### 4. Evaluasi model

```python
eval_results = evaluate_model(interactions, n_users_eval=300, top_k=10)
```

Metrik yang dihasilkan: **Hit Rate@K** dan **NDCG@K**, dipecah berdasarkan tipe pengguna (known vs cold-start).

## Hasil & evaluasi

Pengujian awal pada sampel pengguna menunjukkan:

- Pengguna dengan riwayat interaksi (`known user`) mendapat rekomendasi yang didominasi komponen Content-Based, mencerminkan preferensi kategori produk yang konsisten dengan histori mereka.
- Pengguna cold-start sepenuhnya mengandalkan komponen Session-Based, menghasilkan daftar rekomendasi berbasis popularitas global yang identik antar pengguna baru.
- Skor Collaborative Filtering belum berkontribusi secara optimal pada pengujian awal, mengindikasikan perlunya tuning lebih lanjut pada parameter ALS (jumlah faktor, iterasi, dan confidence scaling).

## Keterbatasan

- Model ALS sensitif terhadap sparsity matriks; performa CF dapat menurun signifikan jika histori interaksi pengguna terlalu sedikit.
- Rekomendasi cold-start belum memanfaatkan fitur demografis pengguna (income level, lokasi) secara langsung — saat ini hanya konteks sesi (device, referrer).
- Evaluasi offline menggunakan split temporal sederhana (80/20 per pengguna), belum mencerminkan skenario produksi sepenuhnya.
- Dataset bersifat sintetis sehingga pola perilaku mungkin tidak menangkap seluruh nuansa perilaku belanja dunia nyata.

## Roadmap

- [ ] Tuning hyperparameter ALS (factors, regularization, confidence scaling)
- [ ] Menambahkan fitur demografis ke modul session-based untuk cold-start yang lebih personal
- [ ] Implementasi Graph Neural Network (LightGCN) sebagai komponen tambahan
- [ ] Implementasi Sequential Recommendation (SASRec) berbasis urutan sesi
- [ ] Dashboard evaluasi interaktif untuk perbandingan antar model
- [ ] Eksperimen A/B simulasi untuk validasi bobot hybrid

## Kontribusi

Kontribusi dalam bentuk issue, pull request, maupun diskusi sangat terbuka. Sebelum mengirimkan PR, mohon:

1. Fork repository ini
2. Buat branch baru (`git checkout -b fitur/nama-fitur`)
3. Commit perubahan dengan pesan yang jelas
4. Ajukan pull request beserta deskripsi perubahan

## Lisensi

Project ini menggunakan dataset sintetis untuk tujuan riset dan pembelajaran. Sesuaikan lisensi (misalnya MIT License) sesuai kebutuhan sebelum publikasi resmi.

## Kolaborator

<!-- Tambahkan kolaborator project di bawah ini -->

| Nama | Peran | GitHub |
|---|---|---|
| | | |
