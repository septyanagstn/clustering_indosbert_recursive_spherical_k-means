import os
import re
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# ── Konstanta Eksperimen APE ──────────────────────────────────────────────────
MODEL_NAME = "denaya/indoSBERT-large"  
DATASET_PATH = "datasets/preprocessed_tickets.txt"
MAX_TOKENS = 256 


# ── Stats Helpers ─────────────────────────────────────────────────────────────
def print_word_stats(descriptions: list[str]) -> None:
    """Menampilkan statistik jumlah kata dari dataset."""
    word_counts = [len(d.split()) for d in descriptions]
    print(f"Total tiket keluhan     : {len(descriptions)}")
    print(f"Maksimum kata per tiket : {max(word_counts)}")
    print(f"Minimum kata per tiket  : {min(word_counts)}")
    print(f"Rata-rata kata per tiket: {sum(word_counts) / len(word_counts):.2f}")


def print_token_stats(token_counts: list[int], total: int) -> None:
    """Menampilkan statistik token hasil encoding tokenizer IndoSBERT."""
    count_exceeding = sum(1 for c in token_counts if c > MAX_TOKENS)
    print(f"Total token keseluruhan : {sum(token_counts)}")
    print(f"Maksimum token ditemukan: {max(token_counts)}")
    print(f"Minimum token ditemukan : {min(token_counts)}")
    print(f"Rata-rata token/tiket   : {sum(token_counts) / total:.2f}")
    print(f"Melebihi batas {MAX_TOKENS} token: {count_exceeding}/{total} tiket")


def print_long_descriptions(descriptions: list[str], token_counts: list[int]) -> None:
    """Menampilkan detail tiket keluhan yang memotong batasan MAX_TOKENS."""
    long_items = [(i, c) for i, c in enumerate(token_counts) if c > MAX_TOKENS]
    if not long_items:
        print(f"Aman! Tidak ada deskripsi tiket yang melebihi {MAX_TOKENS} token.")
        return

    print(f"\nDetail deskripsi tiket yang melebihi {MAX_TOKENS} token:")
    for i, count in long_items:
        print(f"  [Baris ke-{i + 1}] Jumlah Token: {count}\n  Teks: {descriptions[i]}\n")


# ── Visualisasi Distribusi ────────────────────────────────────────────────────
def plot_boxplots(descriptions: list[str], token_counts: list[int]) -> None:
    """Membuat visualisasi boxplot perbandingan distribusi kata dan token."""
    word_counts = [len(d.split()) for d in descriptions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Distribusi Kata dan Token Dataset Tiket Keluhan (IndoSBERT)", fontsize=14, fontweight="bold")

    for ax, data, label, color, threshold in [
        (axes[0], word_counts,  "Jumlah Kata",  "steelblue",  None),
        (axes[1], token_counts, "Jumlah Token", "darkorange", MAX_TOKENS),
    ]:
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor(color)
        if threshold:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Batas Maks ({threshold})")
            ax.legend()
        ax.set_ylabel(label)
        ax.set_title(f"Distribusi {label}")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    
    # Memastikan folder output grafis aman jika belum ada
    output_image = "datasets/token_distribution_boxplot.png"
    plt.savefig(output_image, dpi=150)
    plt.show()
    print(f"Boxplot distribusi berhasil disimpan di '{output_image}'")


# ── Main Runner ───────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Validasi Keberadaan Berkas Input (.txt hasil preprocessor)
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Berkas '{DATASET_PATH}' tidak ditemukan.")
        print("Silakan jalankan berkas 'text_preprocessor.py' terlebih dahulu.")
        return

    # 2. Membaca Berkas Baris demi Baris secara native teks Python
    print(f"Membaca berkas hasil preprocessing: {DATASET_PATH}")
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        # Mengambil untaian kalimat yang tidak kosong
        descriptions = [line.strip() for line in f.readlines() if line.strip()]

    # Tampilkan statistik kata dasar
    print("\n── Statistik Kata Dasar (Preprocessed) ──────────")
    print_word_stats(descriptions)

    # 3. Memuat Tokenizer Asli Milik Model denaya/indoSBERT-large
    print(f"\nMemuat Tokenizer IndoSBERT: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Proses Tokenisasi Riil mengekstrak panjang sekuens token
    token_counts = [
        len(tokenizer.encode(d, add_special_tokens=True))
        for d in descriptions
    ]

    # Tampilkan statistik token hasil ekstraksi
    print("\n── Statistik Token Model IndoSBERT ────────────────")
    print_token_stats(token_counts, len(descriptions))

    # Tampilkan baris pencilan (outliers) jika ada yang melebihi batas masukan
    print_long_descriptions(descriptions, token_counts)

    # Plot visualisasi boxplot pendukung laporan Bab IV
    plot_boxplots(descriptions, token_counts)


if __name__ == "__main__":
    main()