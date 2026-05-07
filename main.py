import pandas as pd
from src.sentence_embeddings import SentenceEmbedding

def main():
    # load data from preprocessed file but without the new line character
    with open('datasets/preprocessed_tickets.txt', 'r', encoding='utf-8') as f:
        preprocessed_data = [line.strip() for line in f.readlines()]

    print(preprocessed_data[:5])  # print first 5 preprocessed descriptions
    
    # create instance of SentenceEmbedding class
    sentence_embedding = SentenceEmbedding()

    # encode the preprocessed data into sentence embeddings
    embeddings = sentence_embedding.encode(preprocessed_data)

    dim, n_samples = embeddings.shape
    print(f"Dimensionality of embeddings: {dim}")
    print(f"Number of samples: {n_samples}")

    # save the sentence embeddings to a file
    sentence_embedding.save_to_file(embeddings, filename='sentence_embeddings.npy')
    print("Sentence embeddings have been saved to 'sentence_embeddings.npy'.")


if __name__ == "__main__":
    main()