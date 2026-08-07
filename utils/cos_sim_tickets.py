import re
import ast
import numpy as np
import pandas as pd
from urllib.parse import urlparse


# ============================================================
# Config
# ============================================================
TICKETS_PATH = 'datasets/10tickets.csv'
CENTROIDS_PATH = 'results/centroids/k3_k2/centroids.csv'
MODEL_NAME = 'denaya/indoSBERT-large'
MAX_LENGTH = 256
STRIDE = 128
OUTPUT_PATH = 'results/cos_sim_tickets.csv'


# ============================================================
# Cleansing (identik dengan data_preparation.ipynb)
# ============================================================
def remove_newline(teks):
    if not isinstance(teks, str):
        return teks
    return re.sub(r'\n+', '. ', teks)


def remove_placeholder(teks):
    if not isinstance(teks, str):
        return teks

    pattern_klien_berlabel = r'\b(?:(?:klien|client|clien|lien)[\s:;,\.]*|(?:dashboard trial|dashboard client|dashboard|project)\s*[:;]\s*)(.*?)(?=\b(?:isu|issue|kendala|request|req|keluhan|problem|kebutuhan|detail)\b)'
    teks = re.sub(pattern_klien_berlabel, '', teks, flags=re.IGNORECASE)

    pattern_klien_sisa = r'\b(?:klien|client|clien|lien)[\s:;,\.]*[^.!?\n]*?(?=[.!?\n]|$)'
    teks = re.sub(pattern_klien_sisa, '', teks, flags=re.IGNORECASE)

    pattern_isu_detail = r'\b(?:isu|issue|kendala|request|req|keluhan|problem|kebutuhan|detail)\s*[:;]\s*'
    teks = re.sub(pattern_isu_detail, '', teks, flags=re.IGNORECASE)

    teks = teks.strip(' ;:,-')
    teks = re.sub(r'[ \t]+', ' ', teks)

    return teks


def remove_emoticon(teks):
    if not isinstance(teks, str):
        return teks
    emoticon_pattern = (
        r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
        r'\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF'
        r'\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+'
    )
    return re.sub(emoticon_pattern, '', teks, flags=re.IGNORECASE)


def cleanse_text(teks):
    teks = remove_newline(teks)
    teks = remove_placeholder(teks)
    teks = remove_emoticon(teks)
    return teks


# ============================================================
# Normalization (identik dengan data_preparation.ipynb)
# ============================================================
GREETING_PATTERN = (
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


def replace_account_and_email(teks):
    teks = re.sub(r'@\w+', 'nama orang', teks)
    teks = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email', teks)
    return teks


def replace_greetings(teks):
    return re.sub(GREETING_PATTERN, 'sapaan', teks, flags=re.IGNORECASE)


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

    return re.sub(pola_url, ekstrak_domain, teks)


def normalize_text(teks):
    teks = teks.lower()
    teks = replace_account_and_email(teks)
    teks = replace_greetings(teks)
    teks = parse_url(teks)
    return teks


# ============================================================
# Sentence Embedding dengan Sliding Window (identik dengan notebook)
# ============================================================
def get_sliding_window_embedding(teks, model, tokenizer, max_length=256, stride=128):
    original_max_length = tokenizer.model_max_length
    tokenizer.model_max_length = 100_000
    tokens = tokenizer.encode(teks, add_special_tokens=False)
    tokenizer.model_max_length = original_max_length

    if len(tokens) <= max_length - 2:
        return model.encode(teks)

    window_size = max_length - 2
    step = window_size - stride

    chunk_embeddings = []
    start = 0
    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunk_emb = model.encode(chunk_text)
        chunk_embeddings.append(chunk_emb)
        if end == len(tokens):
            break
        start += step

    return np.mean(chunk_embeddings, axis=0)


# ============================================================
# Utils: Load Centroid
# ============================================================
def load_centroids(path):
    df = pd.read_csv(path)
    df['vector'] = df['vector'].apply(ast.literal_eval)
    return df


def build_centroid_label(row):
    if pd.isna(row.sub_cluster_id) or str(row.sub_cluster_id).strip() == '':
        return f"{row.level}_C{int(row.cluster_id)}"
    return f"{row.level}_C{int(row.cluster_id)}_S{int(float(row.sub_cluster_id))}"


# ============================================================
# Main
# ============================================================
def main():
    from sentence_transformers import SentenceTransformer

    # 1. Load & bersihkan data tiket
    df_raw = pd.read_csv(TICKETS_PATH)
    df = df_raw[['DESKRIPSI']].dropna().rename(columns={'DESKRIPSI': 'RAW TICKET'})

    df['CLEANED TICKET'] = df['RAW TICKET'].apply(cleanse_text)
    df['NORMALIZED TICKET'] = df['CLEANED TICKET'].apply(normalize_text)
    df['NORMALIZED TICKET'] = df['NORMALIZED TICKET'].fillna('').astype(str)

    # 2. Load model & centroid
    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    df_centroids = load_centroids(CENTROIDS_PATH)
    # ast.literal_eval menghasilkan python float (float64), sedangkan model.encode()
    # menghasilkan float32 -> paksa ke float32 supaya dtype cocok saat cosine similarity
    centroid_matrix = np.vstack(df_centroids['vector'].to_numpy()).astype(np.float32)
    centroid_labels = [build_centroid_label(row) for row in df_centroids.itertuples()]

    # 3. Embed tiket dengan sliding window
    df['EMBEDDING'] = df['NORMALIZED TICKET'].apply(
        lambda teks: get_sliding_window_embedding(teks, model, tokenizer, MAX_LENGTH, STRIDE)
    )
    ticket_matrix = np.vstack(df['EMBEDDING'].to_numpy()).astype(np.float32)

    # 4. Cosine similarity antara tiap tiket & tiap centroid
    similarities = model.similarity(ticket_matrix, centroid_matrix)
    similarities = similarities.numpy() if hasattr(similarities, 'numpy') else np.array(similarities)

    # 5. Susun hasil
    result = pd.DataFrame(similarities, columns=centroid_labels)
    result.insert(0, 'RAW TICKET', df['RAW TICKET'].values)

    print(result)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nHasil disimpan ke: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()