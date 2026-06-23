import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import math
import faiss
import csv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from urllib.parse import urlparse
from collections import Counter
from sklearn.metrics import silhouette_score


def rm_placeholder(teks):
    if not isinstance(teks, str):
        return teks

    # 1a. Tiket berformat label: buang blok klien sampai sebelum label isu/kendala
    pattern_klien_berlabel = r'\b(?:(?:klien|client|clien|lien)[\s:;,\.]*|(?:dashboard trial|dashboard client|dashboard|project)\s*[:;]\s*)(.*?)(?=\b(?:isu|issue|kendala|request|req|keluhan|problem|kebutuhan|detail)\b)'
    teks = re.sub(pattern_klien_berlabel, '', teks)

    # 1b. Sisa "klien: <nama>" yang mungkin masih ada (tiket tanpa label isu)
    pattern_klien_sisa = r'\b(?:klien|client|clien|lien)[\s:;,\.]*[^.!?\n]*?(?=[.!?\n]|$)'
    teks = re.sub(pattern_klien_sisa, '', teks)

    # 2. Hapus label isu/kendala (tidak berubah)
    pattern_isu_detail = r'\b(?:isu|issue|kendala|request|req|keluhan|problem|kebutuhan|detail)\s*[:;]\s*'
    teks = re.sub(pattern_isu_detail, '', teks)

    # 3. Finalisasi
    teks = teks.strip(' ;:,-')
    teks = re.sub(r'\s+', ' ', teks)

    return teks


def mask_cred(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'@\w+', 'nama orang', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email@domain.com', text)
    return text


def mask_greeting(text):
    if not isinstance(text, str):
        return text
    sapaan_pattern = (
        r'\b('
        r'selamat pagi|selamat siang|selamat sore|selamat malam|'
        r'pagi|siang|sore|malam|'
        r'hal+o+|hel+o+|hi|permisi|assalamu\'?alaikum|assalamualaikum|'
        r'punteun|punten|nuhun|'
        r'mohon dibantu|mohon bantuan(?:nya)?|minta tolong|tolong dibantu|tolong bantuan(?:nya)?|'
        r'terima\s?kasih\s?sebelumnya|terimakasih\s?sebelumnya|'
        r'terima\s?kasih\s?banyak|terimakasih\s?banyak|'
        r'terima\s?kasih|terimakasih|makasih|'
        r'thanks|thank\s?you|guys'
        r')\b'
    )
    text = re.sub(sapaan_pattern, '', text, flags=re.IGNORECASE)
    return text


def rm_emotikon(teks):
    if not isinstance(teks, str):
        return teks
    emotikon_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+'
    return re.sub(emotikon_pattern, '', teks)


def parse_url(teks):
    if not isinstance(teks, str):
        return teks

    pola_url = r'https?://[\w\-\.\/\?\&\=\%]+'

    def ekstrak_domain(match):
        url = match.group(0)
        if url.endswith('.') or url.endswith(','):
            url = url[:-1]
        try:
            netloc = urlparse(url).netloc
            parts = netloc.split('.')

            if len(parts) >= 3 and parts[-2] in ['co', 'go', 'ac', 'or', 'sch', 'my']:
                domain_utama = parts[-3]
            elif len(parts) >= 2:
                domain_utama = parts[-2]
            else:
                domain_utama = parts[0]

            if domain_utama == 'www' and len(parts) >= 2:
                domain_utama = parts[-1]

            return f"tautan {domain_utama.lower()}"
        except Exception:
            return "tautan"

    teks = re.sub(pola_url, ekstrak_domain, teks)
    return teks


def preprocess_text(
        text,
        rp_new_line,
        do_lowercase,
        do_rm_placeholder,
        do_mask_cred,
        do_mask_greeting,
        do_rm_emoticon,
        do_parse_url
        ):
    # replace new line special char
    if rp_new_line:
        text = re.sub(r'\n+', '. ', text)

    # lowercase
    if do_lowercase:
        text = text.lower()

    # remove placeholder
    if do_rm_placeholder:
        text = rm_placeholder(text)

    # mask credentials
    if do_mask_cred:
        text = mask_cred(text)

    # mask greeting
    if do_mask_greeting:
        text = mask_greeting(text)

    # remove emoticon
    if do_rm_emoticon:
        text = rm_emotikon(text)

    # parse URL
    if do_parse_url:
        text = parse_url(text)

    return text


def get_sliding_window_embedding(teks, model, tokenizer, max_length=256, stride=128):
    original_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 100_000
    tokens = tokenizer.encode(teks, add_special_tokens=False)
    tokenizer.model_max_length = original_max_length

    # Jika pendek, langsung encode
    if len(tokens) <= max_length - 2:
        return model.encode(teks)

    window_size = max_length - 2  # ruang untuk [CLS] dan [SEP]
    step = window_size - stride    # seberapa jauh window bergeser tiap iterasi

    chunk_embeddings = []
    start = 0

    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunk_emb = model.encode(chunk_text)
        chunk_embeddings.append(chunk_emb)

        if end == len(tokens):  # sudah sampai akhir
            break
        start += step

    return np.mean(chunk_embeddings, axis=0)


def run_skm(dimension, ncluster, X, niter=20):
    skm = faiss.Kmeans(
        d=dimension,
        k=ncluster,
        niter=niter,
        verbose=True,
        spherical=True
        )
    skm.train(X)

    centroids = skm.centroids
    _, labels = skm.index.search(X, 1)

    return centroids, labels.flatten()


def main():
    model_name = 'denaya/indoSBERT-large'
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer

    data_path = 'datasets/raw_tickets.csv'
    df_raw = pd.read_csv(data_path)

    df = df_raw[['DESKRIPSI']].dropna()
    df.rename(columns={'DESKRIPSI': 'RAW TICKET'}, inplace=True)

    preprocess_combinations = [
        [0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0], [1, 0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1, 0], [1, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 1, 0], [1, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0], [1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 1, 0, 1, 0], [0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0], [1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 1, 0], [1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0], [1, 0, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0, 1], [1, 1, 0, 1, 0, 0, 1], [0, 0, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 1, 0, 0, 1], [1, 1, 1, 1, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 1], [1, 1, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0, 1], [0, 0, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 1], [1, 0, 1, 0, 0, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 1], [1, 1, 0, 1, 0, 1, 1], [0, 0, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 1], [1, 0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1, 1], [1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]
    ]

    column_names = [
        'rm new line', 'lowercasing', 'rm placeholder', 'masking cred',
        'masking sapaan', 'rm emoticon', 'parse url'
    ]

    min_clusters = 2
    max_clusters = 10
    k_range = range(min_clusters, max_clusters + 1)

    summary_rows = []
    os.makedirs('outputs', exist_ok=True)

    for idx, combination in enumerate(preprocess_combinations, start=1):
        sp_id = f"SP{idx}"
        print(f"\n=== Menjalankan kombinasi {sp_id}: {combination} ===")

        # 1. Buat ulang df_result dari df mentah untuk tiap kombinasi
        df_result = df.copy()

        rp_new_line, do_lowercase, do_rm_placeholder, do_mask_cred, do_mask_greeting, do_rm_emoticon, do_parse_url = combination

        # 2. Terapkan preprocessing sesuai kombinasi yang sedang berjalan
        df_result['CLEANED TICKET'] = df_result['RAW TICKET'].apply(
            lambda teks: preprocess_text(
                text=teks,
                rp_new_line=bool(rp_new_line),
                do_lowercase=bool(do_lowercase),
                do_rm_placeholder=bool(do_rm_placeholder),
                do_mask_cred=bool(do_mask_cred),
                do_mask_greeting=bool(do_mask_greeting),
                do_rm_emoticon=bool(do_rm_emoticon),
                do_parse_url=bool(do_parse_url)
            )
        )

        # 3. Mencegah Error: pastikan tidak ada data kosong (NaN) akibat proses pembersihan
        df_result['CLEANED TICKET'] = df_result['CLEANED TICKET'].fillna("").astype(str)

        # 4. Ekstraksi Embedding dengan Sliding Window
        tqdm.pandas(desc=f"Embedding {sp_id}")
        df_result['EMBEDDING'] = df_result['CLEANED TICKET'].progress_apply(
            lambda teks: get_sliding_window_embedding(
                teks=teks,
                model=model,
                tokenizer=tokenizer,
                max_length=256,
                stride=128
            )
        )

        X_embeddings = np.vstack(df_result['EMBEDDING'].values)
        embedding_dimension = X_embeddings.shape[1]

        X_embeddings = np.ascontiguousarray(X_embeddings, dtype=np.float32)
        faiss.normalize_L2(X_embeddings)

        results = []
        scenario_count = 0
        total_iterations = sum(k1 * len(k_range) for k1 in k_range)

        with tqdm(total=total_iterations, desc=f"Evaluasi Hierarchical Clustering {sp_id}") as pbar:
            for k1 in k_range:
                centroids_l1, labels_l1 = run_skm(embedding_dimension, k1, X_embeddings)
                silhouette_avg_l1 = silhouette_score(X_embeddings, labels_l1, metric='cosine')

                for cluster_id in range(k1):
                    cluster_indices = np.where(labels_l1 == cluster_id)[0]
                    cluster_embeddings = X_embeddings[cluster_indices]

                    for k2 in k_range:
                        if len(cluster_embeddings) >= k2:
                            centroids_l2, labels_l2 = run_skm(embedding_dimension, k2, cluster_embeddings)
                            silhouette_avg_l2 = silhouette_score(cluster_embeddings, labels_l2, metric='cosine')

                            results.append({
                                'scenario_id': f"{k1-1}.{cluster_id+1}.{k2-1}",
                                'cluster_id': f"{k1}.{cluster_id+1}",
                                'k1': k1,
                                'k2': k2,
                                'silhouette_avg_l1': silhouette_avg_l1,
                                'silhouette_avg_l2': silhouette_avg_l2
                            })
                            scenario_count += 1

                        pbar.update(1)

        print(f"Total scenario yang berhasil dievaluasi untuk {sp_id}: {scenario_count}")

        # 5. Hitung ringkasan (avg & max silhouette L1/L2) untuk kombinasi ini
        df_scenario = pd.DataFrame(results)

        if not df_scenario.empty:
            avg_sil_l1 = df_scenario['silhouette_avg_l1'].mean()
            avg_sil_l2 = df_scenario['silhouette_avg_l2'].mean()
            max_sil_l1 = df_scenario['silhouette_avg_l1'].max()
            max_sil_l2 = df_scenario['silhouette_avg_l2'].max()
        else:
            avg_sil_l1 = avg_sil_l2 = max_sil_l1 = max_sil_l2 = np.nan

        # (opsional) simpan detail skenario per kombinasi untuk audit/debug
        df_scenario.to_csv(f'outputs/detail_{sp_id}.csv', index=False)

        # 6. Susun baris ringkasan sesuai format pada gambar
        row = {'ID': sp_id}
        row.update(dict(zip(column_names, combination)))
        row['Avg Silhouette L1'] = round(float(avg_sil_l1), 4) if not pd.isna(avg_sil_l1) else np.nan
        row['Avg Silhouette L2'] = round(float(avg_sil_l2), 4) if not pd.isna(avg_sil_l2) else np.nan
        row['Max Silhouette L1'] = round(float(max_sil_l1), 4) if not pd.isna(max_sil_l1) else np.nan
        row['Max Silhouette L2'] = round(float(max_sil_l2), 4) if not pd.isna(max_sil_l2) else np.nan

        summary_rows.append(row)

        # 7. Tulis ulang summary CSV setiap iterasi (checkpoint, agar progres tidak hilang)
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv('outputs/summary_silhouette.csv', index=False)

    print("\nSelesai. Hasil ringkasan tersimpan di outputs/summary_silhouette.csv")


if __name__ == '__main__':
    main()