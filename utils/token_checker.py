from transformers import AutoTokenizer
import pandas as pd
import re

def min_preprocess(text):
    # replace \n with space
    text = re.sub(r'\n+', ' ', text)
    # case folding
    text = text.lower()
    # masking url with tautan
    # text = re.sub(r'http\S+', 'tautan', text)
    return text

# Initialize the tokenizer using indobenchmark/indobert-large-p1
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-large-p1")

# load the dataset from the csv file, use only the "DESKRIPSI" column
df = pd.read_csv("datasets/raw_tickets.csv")
descriptions = df["DESKRIPSI"].tolist()

# print total descriptions
total_descriptions = len(descriptions)
print(f"Total descriptions: {total_descriptions}")

# print max words in the descriptions
max_words = max(len(desc.split()) for desc in descriptions)
print(f"Maximum number of words in the descriptions: {max_words}")

# print min words in the descriptions
min_words = min(len(desc.split()) for desc in descriptions)
print(f"Minimum number of words in the descriptions: {min_words}")

# print average words in the descriptions
average_words = sum(len(desc.split()) for desc in descriptions) / total_descriptions
print(f"Average number of words in the descriptions: {average_words:.2f}")

# preprocess each description
descriptions = [min_preprocess(desc) for desc in descriptions]

# check the number of tokens in each description
token_ids = [tokenizer.encode(desc, add_special_tokens=True) for desc in descriptions]
token_counts = [len(ids) for ids in token_ids]

# print total descriptions with more than 512 tokens so it will be look like this: "Total descriptions with more than 512 tokens: 10/{total_descriptions}"
count_exceeding = sum(1 for count in token_counts if count > 512)
total_descriptions = len(descriptions)
print(f"Total descriptions with more than 512 tokens: {count_exceeding}/{total_descriptions}")

# print average number of tokens in the descriptions
average_tokens = sum(token_counts) / total_descriptions
print(f"Average number of tokens in the descriptions: {average_tokens:.2f}")

# print maximum number of tokens in the descriptions
max_tokens = max(token_counts)
print(f"Maximum number of tokens in the descriptions: {max_tokens}")

# print minimum number of tokens in the descriptions
min_tokens = min(token_counts)
print(f"Minimum number of tokens in the descriptions: {min_tokens}")

# print the descriptions with more than 512 tokens
if count_exceeding > 0:
    print("\nDescriptions with more than 512 tokens:")
    for i, count in enumerate(token_counts):
        if count > 512:
            print(f"Description {i+1} (Tokens: {count}): {descriptions[i]}\n")
else:
    print("No descriptions exceed 512 tokens.")