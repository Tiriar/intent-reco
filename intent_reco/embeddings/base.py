# -*- coding: utf-8 -*-
"""
    intent_reco.embeddings.base
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Base classes for the embedding algorithms.

    @author: tomas.brich@seznam.cz
"""

import numpy as np
from nltk import TweetTokenizer

from intent_reco import VERBOSE
from intent_reco.utils.preprocessing import tokenize


class EmbeddingModelBase:
    """Baseline for the embedding algorithms."""
    def __init__(self):
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
        vectors = [self.transform_word(token) for token in tokens]
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

        if VERBOSE:
            print('-- WARNING:', word, 'not in vocabulary!')
        return np.zeros((self.dim,))
