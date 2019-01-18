import numpy as np
import spacy
from glove import Glove

from intent_reco.embeddings.base import EmbeddingModelBase


class GloVe(EmbeddingModelBase):
    """
    GloVe embedding algorithm from glove-python package.
    :param emb_path: path to the embeddings file
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, verbose=False):
        super().__init__(verbose=verbose)
        self.model = Glove.load_stanford(emb_path)
        self.dim = self.model.word_vectors.shape[1]

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        try:
            idx = self.model.dictionary[word]
            return self.model.word_vectors[idx]
        except KeyError:
            return self.handle_oov(word)


class GloVeSpacy(EmbeddingModelBase):
    """
    GloVe embedding algorithm included in SpaCy package.
    :param emb_path: path to the embeddings file
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, verbose=False):
        super().__init__(verbose=verbose)

        self.model = spacy.blank('en')
        with open(emb_path, 'rb') as f:
            tmp = f.readline().split()
            dim = len(tmp)-1

        self.model.vocab.reset_vectors(width=int(dim))
        with open(emb_path, 'rb') as f:
            for line in f:
                line = line.rstrip().decode('utf-8')
                pieces = line.rsplit(' ', int(dim))
                word = pieces[0]
                vector = np.asarray([float(v) for v in pieces[1:]], dtype='f')
                self.model.vocab.set_vector(word, vector)

        # self.model = spacy.load('en', vectors=False)
        # with open(GLOVE_EMBEDDINGS, 'r', encoding='utf-8') as f:
        #     self.model.vocab.load_vectors(f)

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        w = word.lower()
        tmp = self.model(w)
        return tmp.vector

    def transform_sentence(self, sentence):
        """
        Transform the <sentence> into vector representation.
        :param sentence: input sentence
        :return: sentence numpy vector
        """
        return self.transform_word(sentence)
