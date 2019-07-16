# -*- coding: utf-8 -*-
"""
    intent_reco.embeddings.infersent
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    InferSent embedding algorithm wrappers.

    @author: tomas.brich@seznam.cz
"""

import torch

import intent_reco.embeddings.infersent_models as im


class InferSent:
    """
    InferSent embedding algorithm.
    :param emb_path: path to the word embeddings file
    :param model_path: path to the model pickle file
    :param vocab_data: list of sentences for building vocabulary
        (if left None, <k> first word embedding vectors will be loaded as the vocabulary)
    :param k: number of vectors to load from the embeddings file when <vocab_data> is None
    """
    def __init__(self, emb_path, model_path, vocab_data=None, k=100000):
        with open(emb_path) as f:
            header = f.readline().split()
        model_params = {'bsize': 64, 'word_emb_dim': int(header[1]), 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
        self.model = im.InferSent(model_params)
        self.model.load_state_dict(torch.load(model_path))
        self.model.set_w2v_path(emb_path)
        if vocab_data:
            self.model.build_vocab(vocab_data, tokenize=True)
        else:
            self.model.build_vocab_k_words(thr=k)

    def transform_sentences(self, sentences):
        """
        Transform a list of sentences into vector representation.
        :param sentences: list of sentences
        :return: list of numpy vectors
        """

        return self.model.encode(sentences, tokenize=True)

    def visualize(self, sentence):
        """
        Visualize the importance of each word in a sentence.
        :param sentence: input sentence
        """

        self.model.visualize(sentence, tokenize=True)
