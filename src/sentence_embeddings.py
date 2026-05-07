from sentence_transformers import SentenceTransformer
import numpy as np

# create class for sentence embedding
class SentenceEmbedding:
    # constructor to initialize the model
    def __init__(self, model_name='denaya/indoSBERT-large'):
        self.model = SentenceTransformer(model_name)

    # method to encode the text into sentence embedding
    def encode(self, text):
        return self.model.encode(text)
    
    # method to save the sentence embedding to a file
    def save_to_file(self, embedding, filename='sentence_embedding.npy'):
        np.save(filename, embedding)