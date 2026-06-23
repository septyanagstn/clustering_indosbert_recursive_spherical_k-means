import math
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import silhouette_score


class RecursiveSphericalKMeans:
    def __init__(self, d, min_k=2, max_k=10, niter=20, generate_wc_l1=True, generate_wc_l2=False,
                 duplicate_similarity_threshold=0.9, robust_intra_sim_threshold=0.7,
                 top_n_keywords=5):
        self.d = d
        self.k_range = list(range(min_k, max_k + 1))
        self.niter = niter
        self.total_runs = 0
        self.results = {}
        self.generate_wc_l1 = generate_wc_l1
        self.generate_wc_l2 = generate_wc_l2

        self.duplicate_similarity_threshold = duplicate_similarity_threshold
        self.robust_intra_sim_threshold = robust_intra_sim_threshold
        self.top_n_keywords = top_n_keywords

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

    def _max_inter_centroid_similarity(self, centroids):
        """
        Hitung nilai cosine similarity tertinggi antar SEMUA pasangan centroid
        (bukan dengan dirinya sendiri). Dipakai untuk mendeteksi indikasi klaster
        yang terlalu mirip / redundan satu sama lain (Objective 2: No Duplicates).
        """
        n = len(centroids)
        if n < 2:
            return 0.0
        sims = centroids @ centroids.T
        np.fill_diagonal(sims, -1.0)  # abaikan similarity centroid dengan dirinya sendiri (=1.0)
        return float(np.max(sims))

    def _top_keywords(self, texts, labels, top_n=5):
        """
        Ekstrak top-N kata berfrekuensi tertinggi per klaster (di luar stopword masking),
        dipakai sebagai bukti kuantitatif untuk kondisi Explanatory.
        Mengembalikan dict {label: [kata1, kata2, ...]}.
        """
        result = {}
        for label in np.unique(labels):
            idx = np.where(labels == label)[0]
            combined = " ".join([texts[i] for i in idx]).lower()
            words = re.findall(r"[a-zA-Z]{3,}", combined)
            words = [w for w in words if w not in self.masking_stopwords]
            counts = Counter(words)
            result[int(label)] = [w for w, _ in counts.most_common(top_n)]
        return result

    def _evaluate_row_ending_conditions(self, k1, k2, n_samples,
                                         centroids_l1, intra_sim_l1,
                                         centroids_l2, intra_sub,
                                         sub_texts_wc, labels_l2,
                                         n_classified_this_row, n_total_data):
        k_min, k_max = self.k_range[0], self.k_range[-1]

        concise_status = (k_min <= k1 <= k_max) and (k_min <= k2 <= k_max)

        max_sim_l1 = self._max_inter_centroid_similarity(centroids_l1)
        no_dup_l1 = max_sim_l1 < self.duplicate_similarity_threshold

        if centroids_l2 is not None:
            max_sim_l2 = self._max_inter_centroid_similarity(centroids_l2)
            no_dup_l2 = max_sim_l2 < self.duplicate_similarity_threshold
        else:
            max_sim_l2 = None
            no_dup_l2 = None

        robust_l1 = intra_sim_l1 >= self.robust_intra_sim_threshold
        robust_l2 = (intra_sub >= self.robust_intra_sim_threshold) if intra_sub is not None else None

        comprehensive_status = (n_classified_this_row == n_samples)

        if labels_l2 is not None and sub_texts_wc is not None:
            top_keywords = self._top_keywords(sub_texts_wc, labels_l2, top_n=self.top_n_keywords)
            explanatory_status = all(len(v) > 0 for v in top_keywords.values())
        else:
            top_keywords = None
            explanatory_status = False

        return {
            "Concise (k dlm rentang)": concise_status,
            "Max Inter-Centroid Sim L1": round(max_sim_l1, 4),
            "No Duplicate Cluster L1": no_dup_l1,
            "Max Inter-Centroid Sim L2": round(max_sim_l2, 4) if max_sim_l2 is not None else None,
            "No Duplicate Cluster L2": no_dup_l2,
            "Robust (Intra-Sim L1 >= thr)": robust_l1,
            "Robust (Intra-Sim L2 >= thr)": robust_l2,
            "Comprehensive (Row Covered)": comprehensive_status,
            "Explanatory Top Keywords L2": top_keywords
        }

    def _generate_wordcloud(self, texts, labels, filename_prefix):
        unique_labels = np.unique(labels)  # cek unique di numpy
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
            combined_text = " ".join(cluster_texts) # cek

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

    def evaluate_extendibility(self, dummy_texts, dummy_embeddings, reference_centroids,
                               outlier_similarity_threshold=0.5):
        dummy_norm = np.ascontiguousarray(dummy_embeddings.astype("float32").copy())
        faiss.normalize_L2(dummy_norm)

        ref_norm = np.ascontiguousarray(reference_centroids.astype("float32").copy())
        faiss.normalize_L2(ref_norm)

        sims = dummy_norm @ ref_norm.T
        max_sim_per_dummy = sims.max(axis=1)
        is_outlier = max_sim_per_dummy < outlier_similarity_threshold

        detail_per_dummy = []
        for i, (sim, outlier) in enumerate(zip(max_sim_per_dummy, is_outlier)):
            detail_per_dummy.append({
                "dummy_id": i + 1,
                "max_similarity": round(float(sim), 4),
                "status": "Rendah (Outlier)" if outlier else "Cocok dengan klaster lama"
            })

        n_outlier = int(is_outlier.sum())
        
        # Logika APE: rata-ratakan vektor outlier untuk centroid baru (k+1)
        if n_outlier > 0:
            new_centroid_candidate = dummy_norm[is_outlier].mean(axis=0, keepdims=True)
            faiss.normalize_L2(new_centroid_candidate)
        else:
            new_centroid_candidate = None

        return {
            "status_extendible": n_outlier > 0, # Jika ada outlier, berarti terekspansi
            "n_outlier": n_outlier,
            "new_centroid_candidate": new_centroid_candidate,
            "detail_per_dummy": detail_per_dummy
        }

    def fit_exhaustive_matrix(self, x, text_wordcloud_path, dummy_tickets, dummy_embeddings):
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

        n_total_data = len(x_norm)
        self.total_runs = 0
        matrix_records = []

        global_l2_centroids = {(k1, k2): [] for k1 in self.k_range for k2 in self.k_range}

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

            # Objective 1 (All Classified) untuk skenario k1 ini: konstan untuk seluruh baris k1.
            objective_1_status = (len(labels_l1) == n_total_data)

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
                        ending_conditions = self._evaluate_row_ending_conditions(
                            k1=k1, k2=k2, n_samples=n_samples,
                            centroids_l1=centroids_l1, intra_sim_l1=intra_sim_l1,
                            centroids_l2=None, intra_sub=None,
                            sub_texts_wc=None, labels_l2=None,
                            n_classified_this_row=0, n_total_data=n_total_data,
                            dummy_texts=dummy_tickets, dummy_embeddings=dummy_embeddings
                        )
                        record = {
                            "Skenario K1 (L1)": k1,
                            "Skenario K2 (L2)": k2,
                            "Parent Cluster ID": cluster_id,
                            "N Parent Cluster": n_samples,
                            "Silhouette Score L1": round(sil_l1, 4),
                            "Silhouette Score L2": -1.0,
                            "Intra-Cluster Cosine Sim L1": round(intra_sim_l1, 4),
                            "Intra-Cluster Cosine Sim L2": 0.0,
                            "Status Kosong Level 1": status_null_l1,
                            "Status Kosong Level 2": "Null/Singleton (n < k2)",
                            "Objective 1 (All Classified)": objective_1_status,
                        }
                        record.update(ending_conditions)
                        matrix_records.append(record)
                        continue

                    # Sub-clustering Level 2 pada subset klaster ini
                    labels_l2, centroids_l2 = self._run_kmeans(x_sub, k2)

                    global_l2_centroids[(k1, k2)].extend(centroids_l2)

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

                    ending_conditions = self._evaluate_row_ending_conditions(
                        k1=k1, k2=k2, n_samples=n_samples,
                        centroids_l1=centroids_l1, intra_sim_l1=intra_sim_l1,
                        centroids_l2=centroids_l2, intra_sub=intra_sub,
                        sub_texts_wc=sub_texts_wc, labels_l2=labels_l2,
                        n_classified_this_row=len(labels_l2), n_total_data=n_total_data
                    )

                    record = {
                        "Skenario K1 (L1)": k1,
                        "Skenario K2 (L2)": k2,
                        "Parent Cluster ID": cluster_id,
                        "N Parent Cluster": n_samples,
                        "Silhouette Score L1": round(sil_l1, 4),
                        "Silhouette Score L2": round(sil_sub, 4),
                        "Intra-Cluster Cosine Sim L1": round(intra_sim_l1, 4),
                        "Intra-Cluster Cosine Sim L2": round(intra_sub, 4),
                        "Status Kosong Level 1": status_null_l1,
                        "Status Kosong Level 2": "Aman",
                        "Objective 1 (All Classified)": objective_1_status,
                    }
                    record.update(ending_conditions)
                    matrix_records.append(record)

            print(f"-> k1={k1}: {k1} klaster × 9 variasi k2 selesai dievaluasi.")

        if dummy_tickets is not None and dummy_embeddings is not None:
            print("\n=== Mengeksekusi Kondisi Extendible untuk Semua Skenario ===")
            
            # Siapkan list untuk menyimpan hasil matriks extendibility terpisah
            extendibility_records = []
            
            for (k1, k2), list_centroids in global_l2_centroids.items():
                if len(list_centroids) > 0:
                    # Ubah list centroid gabungan menjadi numpy array
                    ref_centroids = np.array(list_centroids)
                    
                    ext_result = self.evaluate_extendibility(
                        dummy_texts=dummy_tickets,
                        dummy_embeddings=dummy_embeddings,
                        reference_centroids=ref_centroids,
                        outlier_similarity_threshold=0.5
                    )
                    
                    extendibility_records.append({
                        "Skenario K1 (L1)": k1,
                        "Skenario K2 (L2)": k2,
                        "Extendible": "Ya" if ext_result["status_extendible"] else "Tidak",
                        "Jumlah Outlier (dari 10)": ext_result["n_outlier"]
                    })
                    
                    # Opsional: Cetak ke terminal meniru gaya APE
                    print(f"Evaluating K1={k1}, K2={k2} ...")
                    print(f"- Extendible (Ekspansi) : {ext_result['n_outlier']}/10 tiket dummy jadi Outlier")

            # Simpan hasil extendibility ke CSV terpisah agar rapi
            df_ext = pd.DataFrame(extendibility_records)
            df_ext.to_csv('results/experiments/extendibility_semua_skenario.csv', index=False)
            print("\nHasil evaluasi Extendible tersimpan di 'extendibility_semua_skenario.csv'")

        # [KODE PENYIMPANAN df_matrix CSV LAMA TETAP SAMA] ...
        return matrix_records
    
        df_matrix = pd.DataFrame(matrix_records)
        output_csv_path = 'results/experiments/matriks_skenario_rekap_1.csv'
        df_matrix.to_csv(output_csv_path, index=False)

        print(f"\nSelesai! Total eksekusi KMeans: {self.total_runs} kali.")
        print(f"Total rekaman skenario: {len(df_matrix)} baris.")
        print(f"Tersimpan di: '{output_csv_path}'")

        return matrix_records