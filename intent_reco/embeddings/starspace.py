import numpy as np
import csv
from sklearn.metrics.pairwise import cosine_similarity

from intent_reco.embeddings.base import EmbeddingModelBase


class StarSpace(EmbeddingModelBase):
    """
    StarSpace embedding algorithm.
    :param emb_path: path to the embeddings file
    :param k: number of vectors to load to vocabulary
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, k=None, verbose=False):
        super().__init__(verbose=verbose)

        self.vocab = {}
        self.labels = []
        self.label_vectors = []

        with open(emb_path) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
                if k is None or i < k:
                    w = row[0]
                    vec = np.asarray(row[1:], dtype=np.float)
                    if w.startswith('__label__'):
                        w = w.replace('__label__', '')
                        self.labels.append(w)
                        self.label_vectors.append(vec)
                    else:
                        self.vocab[w] = vec

        self.label_vectors = np.asarray(self.label_vectors)
        self.dim = next(iter(self.vocab.values())).shape[0]

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        try:
            return self.vocab[word]
        except KeyError:
            return self.handle_oov(word)

    def classify_sentences(self, sentences):
        """
        Classify sentences into classes trained in the StarSpace model.
        :param sentences: list of sentences to classify
        :return: list of classes
        """
        labels = []
        vectors = self.transform_sentences(sentences)
        for vec in vectors:
            sim = cosine_similarity(vec.reshape(1, -1), self.label_vectors)
            labels.append(self.labels[np.asscalar(np.argmax(sim))])
        return labels
