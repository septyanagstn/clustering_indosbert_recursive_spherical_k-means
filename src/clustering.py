import math
import os
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import silhouette_score

class RecursiveSphericalKMeans:
    def __init__(self, d, min_k=2, max_k=10, niter=20, generate_wc_l1=True, generate_wc_l2=False):
        """
        Parameter:
        d (int): Dimensi fitur embedding — 256 untuk denaya/indoSBERT-large.
        min_k (int): Batas bawah jumlah klaster (default 2).
        max_k (int): Batas atas jumlah klaster (default 10).
        niter (int): Iterasi maksimum FAISS K-Means.
        generate_wc_l1 (bool): Apakah ingin menghasilkan Word Cloud untuk level 1 (default True).
        generate_wc_l2 (bool): Apakah ingin menghasilkan Word Cloud untuk level 2 (default False).
        """
        self.d = d
        self.k_range = list(range(min_k, max_k + 1))
        self.niter = niter
        self.total_runs = 0
        self.results = {}
        self.generate_wc_l1 = generate_wc_l1
        self.generate_wc_l2 = generate_wc_l2
        
        self.masking_stopwords = {
            'tautan', 'sapaan', 'surel', 'nama', 'orang',
            'klien', 'client', 'clien', 'isu'
        }
        
        os.makedirs('results/experiments', exist_ok=True)
        os.makedirs('results/wordclouds', exist_ok=True)

    def _run_kmeans(self, x, k):
        """Jalankan 1 kali FAISS Spherical K-Means."""
        self.total_runs += 1
        
        clus = faiss.Clustering(self.d, k)
        clus.niter = self.niter
        clus.spherical = True
        
        index = faiss.IndexFlatIP(self.d)
        clus.train(x, index)
        
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, self.d)
        index_assign = faiss.IndexFlatIP(self.d)
        index_assign.add(centroids)
        _, labels = index_assign.search(x, 1)
        
        return labels.flatten(), centroids

    def _calculate_intra_cosine_similarity(self, embeddings, labels, centroids):
        """Rata-rata Cosine Similarity intra-klaster."""
        similarities = []
        for i in range(len(centroids)):
            cluster_data = embeddings[labels == i]
            if len(cluster_data) > 0:
                dot_products = np.dot(cluster_data, centroids[i])
                similarities.append(np.mean(dot_products))
        return np.mean(similarities) if similarities else 0.0

    def _generate_wordcloud(self, texts, labels, filename_prefix):
        """Buat Word Cloud semua klaster dalam satu gambar dengan subplot."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        # Tentukan layout grid subplot
        # Jika klaster <= 3, susun horizontal. Lebih dari itu, pakai grid 2 kolom.
        if n_clusters <= 3:
            n_cols = n_clusters
            n_rows = 1
        else:
            n_cols = 2
            n_rows = math.ceil(n_clusters / n_cols)
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(8 * n_cols, 4 * n_rows)
        )
        
        # Normalisasi axes menjadi array 1D agar indexing konsisten
        # (plt.subplots mengembalikan objek tunggal jika hanya 1 subplot)
        if n_clusters == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()
        
        for idx, label in enumerate(unique_labels):
            ax = axes[idx]
            
            cluster_indices = np.where(labels == label)[0]
            cluster_texts = [texts[i] for i in cluster_indices]
            combined_text = " ".join(cluster_texts)
            
            if not combined_text.strip():
                ax.set_visible(False)
                continue
            
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                colormap='viridis',
                max_words=50,
                stopwords=self.masking_stopwords
            ).generate(combined_text)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(
                f"Klaster {label}  •  {len(cluster_indices)} tiket",
                fontsize=13,
                fontweight='bold',
                pad=10
            )
        
        # Sembunyikan subplot sisa jika jumlah klaster ganjil di grid 2 kolom
        for idx in range(n_clusters, len(axes)):
            axes[idx].set_visible(False)
        
        # Judul keseluruhan gambar
        fig.suptitle(filename_prefix.replace('_', ' '), fontsize=15, fontweight='bold', y=1.01)
        
        plt.tight_layout()
        plt.savefig(
            f"results/wordclouds/{filename_prefix}.png",
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()

    def fit_exhaustive_matrix(self, x, text_wordcloud_path):
        """
        Menjalankan 486 kombinasi skenario eksperimen (k1 × k2 = 9 × 9 × k1 rekaman granular).
        
        Alur yang benar:
          untuk setiap k1 (2–10):
            bentuk k1 klaster dari seluruh data
            untuk setiap cluster_id (0 s.d k1-1) yang terbentuk:
              untuk setiap k2 (2–10):
                sub-clustering subset klaster tersebut dengan k2
        """
        print("Memulai inisialisasi dan penguncian Normalisasi L2...")
        x_norm = np.ascontiguousarray(x.copy())
        faiss.normalize_L2(x_norm)
        
        with open(text_wordcloud_path, 'r', encoding='utf-8') as f:
            texts_wordcloud = [line.strip() for line in f.readlines() if line.strip()]
        
        # Sinkronisasi panjang teks dengan jumlah data
        assert len(texts_wordcloud) == len(x_norm), (
            f"Jumlah teks wordcloud ({len(texts_wordcloud)}) "
            f"≠ jumlah embedding ({len(x_norm)}). "
            f"Pastikan kedua file dibuat dari DataFrame yang sama."
        )
        
        self.total_runs = 0
        matrix_records = []
        
        print("\n=== Menjalankan Komputasi Matriks Skenario Eksperimen (Recursive Spherical K-Means) ===")
        
        # ── LOOP LEVEL 1: bentuk k1 klaster dari seluruh data ──────────────────
        for k1 in self.k_range:
            # KMeans L1 hanya dijalankan SEKALI per nilai k1
            labels_l1, centroids_l1 = self._run_kmeans(x_norm, k1)

            if self.generate_wc_l1:
                self._generate_wordcloud(texts=texts_wordcloud, labels=labels_l1, filename_prefix=f"L1_k1-{k1}")
            
            sil_l1 = silhouette_score(x_norm, labels_l1, metric='cosine')
            intra_sim_l1 = self._calculate_intra_cosine_similarity(
                x_norm, labels_l1, centroids_l1
            )
            
            counts_l1 = np.bincount(labels_l1.astype(int), minlength=k1)
            status_null_l1 = "Ada" if np.any(counts_l1 == 0) else "Aman"
            
            # ── LOOP KLASTER: iterasi setiap klaster hasil L1 ──────────────────
            for cluster_id in range(k1):
                # Isolasi subset data milik klaster ini
                indices_mask = np.where(labels_l1 == cluster_id)[0]
                x_sub = x_norm[indices_mask]
                sub_texts_wc = [texts_wordcloud[idx] for idx in indices_mask]
                n_samples = len(x_sub)
                
                # ── LOOP LEVEL 2: uji semua nilai k2 pada subset klaster ini ───
                for k2 in self.k_range:
                    # Kondisi henti: data lebih sedikit dari target k2
                    if n_samples < k2:
                        matrix_records.append({
                            "Skenario K1 (L1)": k1,
                            "Skenario K2 (L2)": k2,
                            "Parent Cluster ID": cluster_id,
                            "N Parent Cluster": n_samples,
                            "Silhouette Score L1": round(sil_l1, 4),
                            "Silhouette Score L2": -1.0,
                            "Intra-Cluster Cosine Sim L1": round(intra_sim_l1, 4),
                            "Intra-Cluster Cosine Sim L2": 0.0,
                            "Status Kosong Level 1": status_null_l1,
                            "Status Kosong Level 2": "Null/Singleton (n < k2)"
                        })
                        continue
                    
                    # Sub-clustering Level 2 pada subset klaster ini
                    labels_l2, centroids_l2 = self._run_kmeans(x_sub, k2)

                    if self.generate_wc_l2:
                        self._generate_wordcloud(texts=sub_texts_wc, labels=labels_l2, filename_prefix=f"L2_k1-{k1}_parent-{cluster_id}_k2-{k2}")
                    
                    unique_l2 = np.unique(labels_l2)
                    if n_samples >= 2 and len(unique_l2) > 1:
                        sil_sub = silhouette_score(x_sub, labels_l2, metric='cosine')
                    else:
                        sil_sub = -1.0
                    
                    intra_sub = self._calculate_intra_cosine_similarity(
                        x_sub, labels_l2, centroids_l2
                    )
                    
                    matrix_records.append({
                        "Skenario K1 (L1)": k1,
                        "Skenario K2 (L2)": k2,
                        "Parent Cluster ID": cluster_id,
                        "N Parent Cluster": n_samples,
                        "Silhouette Score L1": round(sil_l1, 4),
                        "Silhouette Score L2": round(sil_sub, 4),
                        "Intra-Cluster Cosine Sim L1": round(intra_sim_l1, 4),
                        "Intra-Cluster Cosine Sim L2": round(intra_sub, 4),
                        "Status Kosong Level 1": status_null_l1,
                        "Status Kosong Level 2": "Aman"
                    })
            
            print(f"-> k1={k1}: {k1} klaster × 9 variasi k2 selesai dievaluasi.")
        
        df_matrix = pd.DataFrame(matrix_records)
        output_csv_path = 'results/experiments/matriks_skenario_rekap_1.csv'
        df_matrix.to_csv(output_csv_path, index=False)
        
        print(f"\nSelesai! Total eksekusi KMeans: {self.total_runs} kali.")
        print(f"Total rekaman skenario: {len(df_matrix)} baris.")
        print(f"Tersimpan di: '{output_csv_path}'")
        
        return matrix_records