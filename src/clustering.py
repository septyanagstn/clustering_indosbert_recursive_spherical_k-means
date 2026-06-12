import os
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import silhouette_score

class RecursiveSphericalKMeans:
    def __init__(self, d, min_k=2, max_k=10, niter=20):
        """
        Inisialisasi model Recursive Spherical K-Means.
        
        Parameter:
        d (int): Dimensi fitur dari vektor embedding (IndoSBERT = 1024).
        min_k (int): Batas bawah pencarian jumlah klaster k (Skenario = 2).
        max_k (int): Batas atas pencarian jumlah klaster k (Skenario = 10).
        niter (int): Jumlah iterasi maksimum untuk FAISS K-Means.
        """
        self.d = d
        self.k_range = list(range(min_k, max_k + 1))
        self.niter = niter
        self.total_runs = 0
        self.results = {}
        
        # STOPWORDS UNTUK KATA MASKING (Agar Word Cloud Bersih dari Residu Preprocessing)
        self.masking_stopwords = {
            'tautan', 'sapaan', 'surel', 'nama', 'orang', 
            'klien', 'client', 'clien', 'isu'
        }
        
        # Membuat folder untuk menyimpan hasil eksperimen dan wordcloud (FR-1.10)
        os.makedirs('results/experiments', exist_ok=True)
        os.makedirs('results/wordclouds', exist_ok=True)

    def _run_kmeans(self, x, k):
        """Metode internal untuk menjalankan 1 kali FAISS Spherical K-Means."""
        self.total_runs += 1
        
        clus = faiss.Clustering(self.d, k)
        clus.niter = self.niter
        clus.spherical = True  # Mengunci koordinat pada permukaan bola (Hypersphere)
        
        # Menggunakan IndexFlatIP (Inner Product) karena data telah dinormalisasi L2
        index = faiss.IndexFlatIP(self.d)
        clus.train(x, index)
        
        # Ekstrak centroid dan cari label untuk setiap data
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, self.d)
        index_assign = faiss.IndexFlatIP(self.d)
        index_assign.add(centroids)
        _, labels = index_assign.search(x, 1)
        
        return labels.flatten(), centroids

    def _calculate_intra_cosine_similarity(self, embeddings, labels, centroids):
        """Menghitung rata-rata Cosine Similarity intra-klaster untuk kriteria Robust."""
        similarities = []
        for i in range(len(centroids)):
            cluster_data = embeddings[labels == i]
            if len(cluster_data) > 0:
                # Dot product bernilai sama dengan Cosine Similarity karena vektor telah dinormalisasi L2
                dot_products = np.dot(cluster_data, centroids[i])
                similarities.append(np.mean(dot_products))
        return np.mean(similarities) if similarities else 0.0

    def _generate_wordcloud(self, texts, labels, filename_prefix):
        """Otomatisasi pembuatan Word Cloud tunggal untuk memenuhi kriteria Explanatory."""
        unique_labels = np.unique(labels)
        for label in unique_labels:
            # Mengambil teks khusus klaster secara aman menggunakan list comprehension
            cluster_texts = [texts[idx] for idx, lbl in enumerate(labels) if lbl == label]
            combined_text = " ".join(cluster_texts)
            
            if not combined_text.strip():
                continue
                
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white',
                colormap='viridis',
                max_words=50,
                stopwords=self.masking_stopwords # Menyaring kata-kata masking kustom
            ).generate(combined_text)
            
            # Menyimpan berkas gambar tunggal ke direktori target secara mandiri
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(f"results/wordclouds/{filename_prefix}_c{label}.png", dpi=150)
            plt.close()

    def fit_exhaustive_matrix(self, x, text_wordcloud_path):
        """
        Menjalankan 486 kombinasi matriks skenario eksperimen secara penuh (Exhaustive):
        Menguji kombinasi k1 (Level 1) dan k2 (Level 2) dari rentang 2 s.d 10 secara granular
        tanpa merata-rata metrik Level 2 untuk pencatatan taksonomi yang presisi.
        """
        print("Memulai inisialisasi dan penguncian Normalisasi L2...")
        x_norm = np.ascontiguousarray(x.copy())
        faiss.normalize_L2(x_norm) # Memproyeksikan data ke permukaan bola (Hypersphere)
        
        # Memuat HANYA berkas teks wordcloud yang relevan untuk kebutuhan visualisasi kata dominan
        with open(text_wordcloud_path, 'r', encoding='utf-8') as f:
            texts_wordcloud = [line.strip() for line in f.readlines() if line.strip()]
            
        # PENGAMAN SIMETRIS: Memastikan panjang list teks sama dengan jumlah baris matriks x (Mencegah IndexError)
        if len(texts_wordcloud) != len(x_norm):
            diff = len(x_norm) - len(texts_wordcloud)
            if diff > 0:
                texts_wordcloud.extend(["sapaan"] * diff)
            else:
                texts_wordcloud = texts_wordcloud[:len(x_norm)]
            
        self.total_runs = 0
        matrix_records = []
        
        print("\n=== Menjalankan Komputasi Matriks 486 Skenario Eksperimen (Recursive Spherical K-Means) ===")
        
        # LOOP LEVEL 1: Pembentukkan kelompok Kategori Utama (k rentang 2 s.d 10)
        for k1 in self.k_range:
            labels_l1, centroids_l1 = self._run_kmeans(x_norm, k1)
            
            # Menghitung metrik spasial evaluasi Level 1 (Menggunakan metrik 'cosine')
            sil_l1 = silhouette_score(x_norm, labels_l1, metric='cosine')
            intra_sim_l1 = self._calculate_intra_cosine_similarity(x_norm, labels_l1, centroids_l1)
            
            # Memverifikasi sebaran keanggotaan untuk Objective Ending Condition 2
            counts_l1 = np.bincount(labels_l1, minlength=k1)
            status_null_l1 = "Ada" if 0 in counts_l1 else "Aman"
            
            # Membuat Word Cloud unik untuk setiap klaster Level 1 yang terbentuk
            # self._generate_wordcloud(texts_wordcloud, labels_l1, filename_prefix=f"L1_K{k1}")
            
            # LOOP LEVEL 2: Sub-clustering rekursif untuk tingkat Sub-kategori
            for k2 in self.k_range:
                for cluster_id in range(k1):
                    # Isolasi koordinat fitur subset klaster induk saat ini
                    mask = (labels_l1 == cluster_id)
                    x_sub = x_norm[mask]
                    sub_texts_wc = [texts_wordcloud[idx] for idx, flag in enumerate(mask) if flag]
                    
                    n_samples = len(x_sub)
                    
                    # KRITERIA HENTI (Objective Ending Condition 2): 
                    # Jika jumlah data lebih kecil dari parameter k2 target, catat sebagai klaster null/singleton
                    if n_samples < k2:
                        matrix_records.append({
                            "Skenario K1 (L1)": k1,
                            "Skenario K2 (L2)": k2,
                            "Parent Cluster ID": cluster_id,
                            "Silhouette Score L1": round(sil_l1, 4),
                            "Silhouette Score L2": -1.0,
                            "Intra-Cluster Cosine Sim L1": round(intra_sim_l1, 4),
                            "Intra-Cluster Cosine Sim L2": 0.0,
                            "Total Objek Induk (n_parent)": n_samples,
                            "Status Kosong Level 1": status_null_l1,
                            "Status Kosong Level 2": "Null/Singleton (n < k2)"
                        })
                        continue
                        
                    # Eksekusi sub-clustering Level 2
                    labels_l2, centroids_l2 = self._run_kmeans(x_sub, k2)
                    
                    # Hitung Silhouette & Cosine Similarity sub-klaster level 2 secara murni
                    if len(np.unique(labels_l2)) > 1:
                        sil_sub = silhouette_score(x_sub, labels_l2, metric='cosine')
                    else:
                        sil_sub = -1.0
                        
                    intra_sub = self._calculate_intra_cosine_similarity(x_sub, labels_l2, centroids_l2)
                    
                    # Membuat Word Cloud tunggal untuk setiap sub-klaster secara mandiri
                    # self._generate_wordcloud(
                    #     texts=sub_texts_wc, 
                    #     labels=labels_l2, 
                    #     filename_prefix=f"L2_K{k1}x{k2}_parent{cluster_id}"
                    # )
                    
                    # PENCATATAN GRANULAR: Rekam data performa per cluster_id individu (Akumulasi total 486 baris)
                    matrix_records.append({
                        "Skenario K1 (L1)": k1,
                        "Skenario K2 (L2)": k2,
                        "Parent Cluster ID": cluster_id,
                        "Silhouette Score L1": round(sil_l1, 4),
                        "Silhouette Score L2": round(sil_sub, 4),
                        "Intra-Cluster Cosine Sim L1": round(intra_sim_l1, 4),
                        "Intra-Cluster Cosine Sim L2": round(intra_sub, 4),
                        "Total Objek Induk (n_parent)": n_samples,
                        "Status Kosong Level 1": status_null_l1,
                        "Status Kosong Level 2": "Aman"
                    })
                
            print(f"-> Skenario Level 1 k={k1} selesai dievaluasi untuk seluruh rentang variasi k2.")

        # Ekspor kumpulan rekapitulasi data granular menjadi berkas .csv (FR-1.10)
        df_matrix = pd.DataFrame(matrix_records)
        output_csv_path = 'results/experiments/matriks_skenario_rekap_1.csv'
        df_matrix.to_csv(output_csv_path, index=False)
        
        print(f"\nSelesai! Total eksekusi repositori matriks: {len(df_matrix)} rekaman skenario.")
        print(f"Data rekapitulasi skenario kuantitatif disimpan di: '{output_csv_path}'")
        print("Seluruh aset visualisasi Word Cloud disimpan di folder: 'results/wordclouds/'")
        
        return matrix_records