import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

class TicketPreprocessing:
    def __init__(self):
        # 1. Inisialisasi StopWordRemover dari Sastrawi
        factory = StopWordRemoverFactory()
        sastrawi_stopwords = factory.get_stop_words()
        
        # 2. WHITE-LISTING: Karakteristik data IT sangat bergantung pada kata fungsi/negasi tertentu (Bab IV.2)
        # Jangan hapus kata-kata di bawah ini agar konteks klaster keluhan tidak bias (misal: "tidak loading", "belum crawlback")
        keywords_to_keep = {
            'belum', 'tidak', 'kurang', 'bisa', 'semua', 'atas', 'bawah', 
            'masuk', 'keluar', 'hilang', 'mati', 'kosong', 'gagal', 'salah'
        }
        
        # Filter stopword bawaan Sastrawi
        self.stopwords = set([word for word in sastrawi_stopwords if word not in keywords_to_keep])
        
        # 3. EXPANSION: Tambahkan noise kontekstual non-IT yang sering muncul di awal/akhir tiket agar Word Cloud bersih (FR-1.9)
        khas_tiket_noise = {
            'mas', 'mba', 'tim', 'it', 'mohon', 'dibantu', 'punteun', 'tolong', 
            'cek', 'bantu', 'halo', 'min', 'admin', 'dari', 'ke', 'perihal',
            'siang', 'pagi', 'sore', 'malam', 'selamat', 'terima', 'kasih', 'sebelumnya',
            'ya', 'yaa', 'kah', 'lah', 'pun', 'terimakasih', 'salam', 'sapaan', 'nama', 'orang'
        }
        self.stopwords.update(khas_tiket_noise)

        # Pola umum emoji dan simbol grafis (termasuk 🙏 sesuai karakteristik data di laporan)
        self.emoji_pattern = re.compile(
            "["
            r"\U0001F600-\U0001F64F"  # emoticons
            r"\U0001F300-\U0001F5FF"  # symbols & pictographs
            r"\U0001F680-\U0001F6FF"  # transport & map symbols
            r"\U0001F1E0-\U0001F1FF"  # flags
            r"\u2600-\u26FF\u2700-\u27BF" # Simbol tangan / salam / 🙏
            "]+", flags=re.UNICODE
        )
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        

        text = re.sub(r'(?i)\b(klien|client|clien)\s*(:|;)\s*\w+\b', ' ', text)
        
        # Hanya menghapus string label "isu:" atau "isu : " saja
        text = re.sub(r'(?i)\b(isu|issue|kendala|)\s*[:,\s]+\s*', ' ', text)
        
        # 2. PEMBERSIHAN LOG CHAT WHATSAPP COMPREHENSIVE
        text = re.sub(r'\[?\d{1,2}/\d{1,2}(?:/\d{2,4})?,\s+\d{1,2}[:\.]\d{2}(?:\s?[aApP][mM])?\]?\s*(?:\+\d+)?[\d\s-]*:\s*', ' ', text)
        
        # 3. CASE FOLDING (FR-1.2)
        text = text.lower()
        
        # 4. MASKING TAUTAN/URL (FR-1.4)
        text = re.sub(r'(https?://\S+|(?:\s|^)//\S+|\b[a-zA-Z0-9\-\.]+\.(?:com|id|co\.id|net|org|gov|be)\S*)', ' tautan ', text)
        text = re.sub(r'\bhttps?\b', ' tautan ', text)
            
        # 5. MASKING NAMA AKUN AKAN KOMUNIKASI '@' (FR-1.5)
        text = re.sub(r'@\w+', 'nama orang', text)
        
        # 6. MASKING ALAMAT EMAIL (FR-1.6)
        text = re.sub(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', 'surel', text)
        
        # 7. MASKING KATA/KALIMAT SAPAAN FORMAL KOMPREHENSIF (FR-1.7)
        sapaan_pattern = (
            r'\b(selamat pagi|selamat siang|selamat sore|selamat malam|'
            r'punteun|punten|mohon dibantu|mohon bantuan|mohon bantuannya|'
            r'terima kasih sebelumnya|terimakasih sebelumnya|terima kasih banyak|'
            r'terimakasih|terima kasih|thanks|thank you|guys)\b'
        )
        text = re.sub(sapaan_pattern, 'sapaan', text)
        
        # 8. REDUKSI KARAKTER HURUF BERULANG (E.g., "hiiii" -> "hii")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # 9. REDUKSI MASKING TAUTAN/SAPAAN BERULANG (FR-1.3)
        text = re.sub(r'\b(tautan\s+)+tautan\b', 'tautan', text)
        text = re.sub(r'\b(sapaan\s+)+sapaan\b', 'sapaan', text)
        
        # 10. MENGHAPUS EMOTIKON/EMOJI GRAFIS (FR-1.8)
        text = self.emoji_pattern.sub('', text)
        
        # 11. CLEAN UP SIMBOL DAN SPASI LIAR
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'^[^a-zA-Z0-9#]+', '', text)
        
        # SAFEGUARD: Jika hasil akhir murni kosong akibat teks asal hanya berisi simbol,
        # berikan token kustom agar baris data tidak terhapus (Menjaga keutuhan 1.639 data)
        # if text == "" or text.isspace():
        #     text = "sapaan"
            
        return text.strip()

    def remove_stopwords(self, text):
        # Memisahkan kata, menghapus stopword kustom, dan menggabungkannya kembali (FR-1.9)
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords]
        return ' '.join(filtered_words)

# ==========================================
# EKSEKUSI ALUR PREPROCESSING
# ==========================================

try:
    df = pd.read_csv('datasets/tickets.csv')
    if df.empty:
        print("Dataset is empty.")
        exit()
    else:
        print(f"Dataset loaded successfully. Total data: {len(df)} baris.")
except FileNotFoundError:
    print("File 'datasets/tickets.csv' tidak ditemukan.")
    exit()

raw_descriptions = df['DESKRIPSI'].tolist()
preprocessor = TicketPreprocessing()

# Dataset 1: Untuk Clustering IndoSBERT (Tanpa Stopword Removal - III.6.3)
print("Memproses dataset 1 untuk kebutuhan Clustering...")
dataset_clustering = [preprocessor.clean_text(desc) for desc in raw_descriptions]

# Dataset 2: Untuk Word Cloud (Dengan Stopword Removal Kustom - III.6.3)
print("Memproses dataset 2 untuk kebutuhan Word Cloud...")
dataset_wordcloud = [preprocessor.remove_stopwords(desc) for desc in dataset_clustering]

# Simpan hasil eksekusi (FR-1.10)
with open('datasets/preprocessed_tickets.txt', 'w', encoding='utf-8') as f:
    for desc in dataset_clustering:
        f.write(desc + '\n')

with open('datasets/preprocessed_wordcloud.txt', 'w', encoding='utf-8') as f:
    for desc in dataset_wordcloud:
        f.write(desc + '\n')

print("Proses preprocessing selesai dan kedua variasi dataset berhasil disimpan.")