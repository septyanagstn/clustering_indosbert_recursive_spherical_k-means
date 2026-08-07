# Cluster analysis untuk membangun taksonomi hierarkis pada kasus tiket Helpdesk

Proyek ini berisi alur kerja clustering tiket berbahasa Indonesia menggunakan IndoSBERT dan spherical k-means bertingkat. Fokus utama repo ada pada dua notebook: preprocessing data dan eksperimen clustering.

## Ringkasan Alur

1. `data_preparation.ipynb` memuat data mentah, membersihkan teks, menormalkan isi tiket, membuat embedding, lalu menyimpan hasil turunan ke file CSV dan pickle.
2. `clustering.ipynb` memuat embedding hasil preprocessing, menjalankan spherical k-means bertingkat, membuat visualisasi, dan mengekspor hasil cluster.

## Struktur Data

Dataset awal di repo ini tidak disertakan penuh. Sebelum menjalankan notebook, pastikan file `raw_tickets.csv` sudah diunduh dan diletakkan di folder `datasets/` dengan nama:

`datasets/raw_tickets.csv`

Notebook preprocessing akan membaca file tersebut dan menghasilkan file turunan seperti `cleaned_tickets.csv`, `preprocessed_output.csv`, dan `tickets.pkl`.

## Panduan Penggunaan

1. Pastikan environment Python aktif dan dependensi notebook tersedia.
2. Unduh atau salin `raw_tickets.csv` ke `datasets/raw_tickets.csv`.
3. Jalankan `data_preparation.ipynb` untuk menghasilkan data bersih dan embedding.
4. Jalankan `clustering.ipynb` untuk melakukan eksperimen spherical k-means, membuat word cloud, menyimpan label cluster, dan mengekspor centroid serta tiket representatif.
5. Lihat hasil di folder `results/`.

## Output Utama

- `datasets/cleaned_tickets.csv`: hasil cleansing dan normalisasi.
- `datasets/preprocessed_output.csv`: data akhir yang dipakai untuk clustering.
- `datasets/tickets.pkl`: data lengkap termasuk embedding.
- `results/experiments/results.csv`: ringkasan hasil eksperimen.
- `results/labels/`: label cluster per skenario.
- `results/centroids/`: centroid dan tiket terdekat/terjauh.
- `results/wordclouds/` dan `results/tsne/`: visualisasi pendukung.

## Catatan

- Folder `utils/` tidak menjadi bagian dari alur utama notebook ini.
- Jika notebook dijalankan ulang dari awal, beberapa file di `datasets/` dan `results/` akan ditimpa dengan keluaran baru.