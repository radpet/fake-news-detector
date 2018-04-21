import numpy as np


class GloveEmbeddings:
    def __init__(self, path, dim):
        self.path = path
        self.dim = dim
        self.embeddings_index = {}

    def load(self):
        with open(self.path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
        print('Found %s word vectors.' % len(self.embeddings_index))
        return self

    def get_embedding_matrix_for_tokenizer(self, tokenizer):
        if len(self.embeddings_index) == 0:
            raise RuntimeError('Embeddings not loaded, call load() first')

        word_index = tokenizer.word_index
        embedding_matrix = np.zeros((len(word_index) + 1, self.dim))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
