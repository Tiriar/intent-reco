import numpy as np
from nltk import TweetTokenizer

from utils.preprocessing import tokenize


class EmbeddingModelBase:
    """
    Baseline for the embedding algorithms.
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.tokenizer = TweetTokenizer()
        self.dim = None

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        raise NotImplementedError

    def transform_sentence(self, sentence):
        """
        Transform the <sentence> into vector representation.
        :param sentence: input sentence
        :return: sentence numpy vector
        """
        s = tokenize(sentence, tknzr=self.tokenizer, to_lower=True)
        tokens = s.split()
        vectors = []
        for token in tokens:
            vectors.append(self.transform_word(token))
        return np.average(vectors, axis=0)

    def transform_sentences(self, sentences):
        """
        Transform a list of sentences into vector representation.
        :param sentences: list of sentences
        :return: list of numpy vectors
        """
        return [self.transform_sentence(s) for s in sentences]

    def handle_oov(self, word):
        """
        Function for handling out-of-vocabulary words.
        :param word: oov word
        :return: oov word numpy vector
        """
        if self.verbose:
            print('-- WARNING:', word, 'not in vocabulary!')
        return np.zeros((self.dim,))
