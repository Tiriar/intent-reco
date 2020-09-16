# -*- coding: utf-8 -*-
"""
    intent_reco.embeddings.compressed
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Wrappers for compressed embedding models.

    @author: tomas.brich@seznam.cz
"""

import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from intent_reco import VERBOSE
from intent_reco.embeddings.base import EmbeddingModelBase


class CompressedModel(EmbeddingModelBase):
    """
    Wrapper for loading compressed models.
    :param emb_path: path to the embeddings file
    :param cb_path: path to the codebook
    """
    def __init__(self, emb_path, cb_path):
        super().__init__()

        self.normalized = None
        self.distinct_cb = None
        self.decode_func = None

        self.words = None
        self.vocab = dict()
        self.sizes = dict()

        self.cb = []
        self.cb_size = None
        self.cb_dim = None

        self.labels = []
        self.label_vectors = []

        self.load_model(emb_path, cb_path)

    def load_model(self, emb_path, cb_path):
        """
        Load the embedding model (set class variables).
        :param emb_path: path to the embeddings file
        :param cb_path: path to the codebook
        """

        with open(cb_path) as f:
            header = f.readline().split()
            self.cb_size = int(header[0])
            self.cb_dim = int(header[1])
            self.cb = [np.asarray(line.strip().split(), dtype=np.float) for line in f]
        int_type = np.uint16 if self.cb_size > 256 else np.uint8

        with open(emb_path) as f:
            header = f.readline().split()
            self.words = int(header[0])
            cmp_dim = int(header[1])
            self.dim = cmp_dim * self.cb_dim
            self.normalized = True if 'NORM' in header else False
            self.distinct_cb = True if 'DIST' in header else False
            self.decode_func = self.decode_vec_distinct if self.distinct_cb else self.decode_vec

            for line in f:
                tmp = line.strip().split()
                if self.normalized:
                    w = ' '.join(tmp[:len(tmp)-cmp_dim-1])
                    size = float(tmp.pop())
                else:
                    w = ' '.join(tmp[:len(tmp)-cmp_dim])
                    size = 1

                vec = np.asarray(tmp[-cmp_dim:], dtype=int_type)

                if w.startswith('__label__'):
                    w = w.lstrip('__label__')
                    self.labels.append(w)
                    self.label_vectors.append(self.decode_func(vec) * size)
                else:
                    self.vocab[w] = vec
                    self.sizes[w] = size

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """

        try:
            vec = self.decode_func(self.vocab[word])
            # if self.normalized:
            #     vec *= self.sizes[word]
            return vec
        except KeyError:
            return self.handle_oov(word)

    def classify_sentences(self, sentences):
        """
        Classify sentences into classes trained in the StarSpace model.
        :param sentences: list of sentences to classify
        :return: list of classes
        """

        vectors = self.transform_sentences(sentences)
        labels = [self.labels[np.argmax(cosine_similarity(vec.reshape(1, -1), self.label_vectors)).item()]
                  for vec in vectors]
        return labels

    def decode_vec(self, vec):
        """
        Decode compressed vector.
        :param vec: numpy vector
        :return: decoded numpy vector
        """

        out = [self.cb[idx] for idx in vec]
        return np.concatenate(out)

    def decode_vec_distinct(self, vec):
        """
        Decode compressed vector (version for distinct codebooks).
        :param vec: numpy vector
        :return: decoded numpy vector
        """

        out = [self.cb[i * self.cb_size + idx] for i, idx in enumerate(vec)]
        return np.concatenate(out)


class CompressedModelPickled(CompressedModel):
    """
    Wrapper for loading pickled compressed models.
    :param emb_path: path to the pickled compressed model
    """
    def __init__(self, emb_path):
        super().__init__(emb_path=emb_path, cb_path=None)

    def load_model(self, emb_path, cb_path):
        """
        Load the embedding model (set class variables).
        :param emb_path: path to the pickled compressed model
        :param cb_path: (not used, inherited from superclass)
        """

        with open(emb_path, 'rb') as f:
            data = pickle.load(f)

        self.dim = data['dim']
        self.normalized = data['normalized']
        self.distinct_cb = data['distinct_codebooks']
        self.decode_func = self.decode_vec_distinct if self.distinct_cb else self.decode_vec

        self.words = data['words']
        self.vocab = data['vectors']
        self.sizes = data['norms']

        self.cb = data['codebook']
        self.cb_size = data['codebook_size']
        self.cb_dim = data['codebook_dim']

        self.labels = data['labels']
        self.label_vectors = data['label_vectors']


class CompressedModelSWPickled(CompressedModelPickled):
    """
    Wrapper for loading pickled compressed models containing subword embeddings.
    :param emb_path: path to the pickled compressed model
    :param sw_path: path to the pickled compressed subwords
    """
    def __init__(self, emb_path, sw_path):
        super().__init__(emb_path=emb_path)

        with open(sw_path, 'rb') as f:
            data = pickle.load(f)

        self.subwords = data['words']
        self.sw_vocab = data['vectors']
        self.sw_sizes = data['norms']

        self.sw_cb = data['codebook']
        self.sw_cb_size = data['codebook_size']
        self.sw_cb_dim = data['codebook_dim']

        self.decode_sw_func = self.decode_sw_vec_distinct if self.distinct_cb else self.decode_sw_vec

    def handle_oov(self, word):
        """
        Function for handling out-of-vocabulary words.
        Creates OOV embedding from subword embeddings.
        :param word: oov word
        :return: oov word numpy vector
        """

        if VERBOSE:
            print('Creating embedding for OOV word:', word)

        sws = self.get_subwords(word)
        vectors = []
        for sw in sws:
            try:
                vec = self.decode_sw_func(self.sw_vocab[sw])
                if self.normalized:
                    vec *= self.sw_sizes[sw]
                vectors.append(vec)
            except KeyError:
                vectors.append(np.zeros((self.dim,)))
        return np.average(vectors, axis=0)

    @staticmethod
    def get_subwords(word, nmin=3, nmax=6):
        """
        Get a list of n-grams in <word> with n between <nmin> and <nmax>.
        :param word: input word
        :param nmin: lower bound on n
        :param nmax: upper bound on n
        :return: list of n-grams in <word>
        """

        word = '<' + word + '>'
        return [word[i:j] for i in range(len(word))
                for j in range(i + nmin, 1 + min(i + nmax, len(word)))]

    def decode_sw_vec(self, vec):
        """
        Decode compressed subword vector.
        :param vec: numpy vector
        :return: decoded numpy vector
        """

        out = [self.sw_cb[idx] for idx in vec]
        return np.concatenate(out)

    def decode_sw_vec_distinct(self, vec):
        """
        Decode compressed subword vector (version for distinct codebooks).
        :param vec: numpy vector
        :return: decoded numpy vector
        """

        out = [self.sw_cb[i * self.sw_cb_size + idx] for i, idx in enumerate(vec)]
        return np.concatenate(out)


def decode_compressed_model(model_path, cb_path, out_path):
    """
    Decodes a compressed embedding model and writes it as a text file.
    :param model_path: path to the embedding model
    :param cb_path: path to the codebook
    :param out_path: path of the output text file
    """

    from tqdm import tqdm
    model = CompressedModel(model_path, cb_path)
    out = [str(model.words) + ' ' + str(model.dim) + '\n']

    print('Decoding', len(model.vocab), 'words...')
    for word in tqdm(model.vocab, mininterval=1.0):
        tmp = word
        vec = model.decode_func(model.vocab[word]) * model.sizes[word]
        for num in vec:
            tmp += ' ' + '{0:.7f}'.format(num)
        out.append(tmp + '\n')

    print('Writing output...')
    with open(out_path, 'w') as f:
        f.writelines(out)


def pickle_compressed_model(model_path, cb_path, out_path):
    """
    Pickles a compressed embedding model (model + codebook).
    :param model_path: path to the embedding model
    :param cb_path: path to the codebook
    :param out_path: path of the output pickle file
    """

    model = CompressedModel(model_path, cb_path)
    data = {'vectors': model.vocab,
            'norms': model.sizes,
            'words': model.words,
            'dim': model.dim,
            'normalized': model.normalized,
            'codebook': model.cb,
            'codebook_size': model.cb_size,
            'codebook_dim': model.cb_dim,
            'distinct_codebooks': model.distinct_cb,
            'labels': model.labels,
            'label_vectors': model.label_vectors}

    with open(out_path, 'wb') as f:
        pickle.dump(data, f)
