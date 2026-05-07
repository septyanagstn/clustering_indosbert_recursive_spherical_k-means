import numpy as np
import faiss
from sklearn.metrics import silhouette_score

class RecursiveSphericalKMeans:
    def __init__(self, d, min_k=2, max_k=10, niter=20):
        """
        Inisialisasi model Recursive Spherical K-Means.
        
        Parameter:
        d (int): Dimensi vektor.
        min_k (int): Batas bawah pencarian k.
        max_k (int): Batas atas pencarian k.
        niter (int): Jumlah iterasi maksimum untuk FAISS K-Means.
        """
        self.d = d
        self.k_range = list(range(min_k, max_k + 1))
        self.niter = niter
        self.total_runs = 0  # Counter untuk melacak berapa kali K-Means dijalankan
        self.results = {}

    def _run_kmeans(self, x, k):
        """Metode internal untuk menjalankan 1 kali FAISS Spherical K-Means."""
        self.total_runs += 1
        
        clus = faiss.Clustering(self.d, k)
        clus.niter = self.niter
        clus.spherical = True
        
        index = faiss.IndexFlatIP(self.d)
        clus.train(x, index)
        
        # Ekstrak centroid dan cari label untuk setiap data
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, self.d)
        index_assign = faiss.IndexFlatIP(self.d)
        index_assign.add(centroids)
        _, labels = index_assign.search(x, 1)
        
        return labels.flatten()

    def _find_optimal_k(self, x_sub):
        """Mencari nilai k optimal untuk sebuah subset data (Level 2)."""
        best_k = None
        best_score = -1.0
        
        # Safeguard: Pastikan jumlah data cukup untuk di-cluster menjadi max_k
        max_possible_k = min(max(self.k_range), len(x_sub) - 1)
        
        if max_possible_k < min(self.k_range):
            return None, -1.0 # Data terlalu sedikit untuk di-cluster
            
        valid_k_range = range(min(self.k_range), max_possible_k + 1)
        
        for k in valid_k_range:
            labels = self._run_kmeans(x_sub, k)
            
            # Silhouette Score butuh > 1 klaster unik yang terisi
            if len(np.unique(labels)) > 1:
                score = silhouette_score(x_sub, labels, metric='cosine')
                if score > best_score:
                    best_score = score
                    best_k = k
                    
        return best_k, best_score

    def fit_exhaustive(self, x):
        """
        Menjalankan evaluasi Exhaustive:
        Menguji semua k di Level 1, dan mencari k optimal untuk SETIAP klaster yang terbentuk.
        """
        # Copy dan normalisasi data di dalam fungsi agar data asli tidak termodifikasi
        print("Memulai inisialisasi dan normalisasi L2...")
        x_norm = np.ascontiguousarray(x.copy())
        faiss.normalize_L2(x_norm)
        
        self.total_runs = 0
        self.results = {}
        
        print("\nMemulai proses Exhaustive Recursive K-Means...")
        
        # LEVEL 1
        for k1 in self.k_range:
            labels_l1 = self._run_kmeans(x_norm, k1)
            score_l1 = silhouette_score(x_norm, labels_l1, metric='cosine')
            
            print(f"\n>> Level 1: k={k1} (Silhouette: {score_l1:.4f})")
            
            subclusters_info = {}
            
            # LEVEL 2 (Untuk setiap klaster dari k1)
            for cluster_id in range(k1):
                # Ambil subset data untuk klaster ini
                mask = (labels_l1 == cluster_id)
                x_sub = x_norm[mask]
                
                # Cari k optimal untuk sub-klaster ini
                best_k2, best_score_l2 = self._find_optimal_k(x_sub)
                
                subclusters_info[cluster_id] = {
                    'size': len(x_sub),
                    'optimal_k': best_k2,
                    'score': best_score_l2
                }
                
                # Print progress bar sederhana
                print(f"   L2 -> Klaster Induk {cluster_id} (n={len(x_sub)}): Optimal sub-k = {best_k2} (Score: {best_score_l2:.4f})")
            
            # Simpan hasil untuk k1 ini
            self.results[k1] = {
                'level_1_score': score_l1,
                'subclusters': subclusters_info
            }
            
        print(f"\nSelesai! Total eksekusi FAISS K-Means: {self.total_runs} kali.")
        return self.results