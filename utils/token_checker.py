from transformers import AutoTokenizer
import pandas as pd
import re
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "indobenchmark/indobert-large-p1"
DATASET_PATH = "datasets/raw_tickets.csv"
COLUMN_NAME = "DESKRIPSI"
MAX_TOKENS = 512


# ── Preprocessing ─────────────────────────────────────────────────────────────
def min_preprocess(text: str) -> str:
    """Minimal preprocessing: normalize whitespace and lowercase."""
    text = re.sub(r'\n+', ' ', text)
    text = text.lower()
    # text = re.sub(r'http\S+', 'tautan', text)  # URL masking (disabled)
    return text


# ── Stats Helpers ─────────────────────────────────────────────────────────────
def print_word_stats(descriptions: list[str]) -> None:
    """Print word count statistics for a list of descriptions."""
    word_counts = [len(d.split()) for d in descriptions]
    print(f"Total descriptions      : {len(descriptions)}")
    print(f"Max words               : {max(word_counts)}")
    print(f"Min words               : {min(word_counts)}")
    print(f"Average words           : {sum(word_counts) / len(word_counts):.2f}")


def print_token_stats(token_counts: list[int], total: int) -> None:
    """Print token count statistics."""
    count_exceeding = sum(1 for c in token_counts if c > MAX_TOKENS)
    print(f"Total tokens            : {sum(token_counts)}")
    print(f"Max tokens              : {max(token_counts)}")
    print(f"Min tokens              : {min(token_counts)}")
    print(f"Average tokens          : {sum(token_counts) / total:.2f}")
    print(f"Exceeds {MAX_TOKENS} tokens   : {count_exceeding}/{total}")


def print_long_descriptions(descriptions: list[str], token_counts: list[int]) -> None:
    """Print descriptions that exceed the MAX_TOKENS limit."""
    long_items = [(i, c) for i, c in enumerate(token_counts) if c > MAX_TOKENS]
    if not long_items:
        print("No descriptions exceed 512 tokens.")
        return

    print(f"\nDescriptions exceeding {MAX_TOKENS} tokens:")
    for i, count in long_items:
        print(f"  [{i + 1}] Tokens: {count}\n  {descriptions[i]}\n")


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_boxplots(descriptions: list[str], token_counts: list[int]) -> None:
    """Plot word and token distribution boxplots side by side."""
    word_counts = [len(d.split()) for d in descriptions]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Distribution of Words and Tokens", fontsize=14, fontweight="bold")

    for ax, data, label, color, threshold in [
        (axes[0], word_counts,  "Word Count",  "steelblue",  None),
        (axes[1], token_counts, "Token Count", "darkorange", MAX_TOKENS),
    ]:
        bp = ax.boxplot(data, patch_artist=True, widths=0.4,
                        medianprops=dict(color="white", linewidth=2))
        bp["boxes"][0].set_facecolor(color)
        if threshold:
            ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Limit ({threshold})")
            ax.legend()
        ax.set_ylabel(label)
        ax.set_title(f"{label} Distribution")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("distribution_boxplot.png", dpi=150)
    plt.show()
    print("Boxplot saved to distribution_boxplot.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # Load dataset
    df = pd.read_csv(DATASET_PATH)
    descriptions = df[COLUMN_NAME].tolist()

    # Word-level stats (before preprocessing)
    print("── Word Stats (raw) ──────────────────────────────")
    print_word_stats(descriptions)

    # Preprocess
    descriptions = [min_preprocess(d) for d in descriptions]

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_counts = [
        len(tokenizer.encode(d, add_special_tokens=True))
        for d in descriptions
    ]

    # Token-level stats
    print("\n── Token Stats ───────────────────────────────────")
    print_token_stats(token_counts, len(descriptions))

    # Show long descriptions
    print()
    print_long_descriptions(descriptions, token_counts)

    # Plot boxplots
    plot_boxplots(descriptions, token_counts)


if __name__ == "__main__":
    main()