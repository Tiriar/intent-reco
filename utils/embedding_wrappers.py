"""
Module for storing embedding algorithm wrappers.

Currently used algorithms:
- InferSent: https://github.com/facebookresearch/InferSent
- sent2vec: https://github.com/epfml/sent2vec
- GloVe: https://github.com/maciejkula/glove-python / SpaCy implementation
- word2vec: Gensim implementation
- FastText: https://github.com/facebookresearch/fastText / Gensim implementation
- StarSpace: https://github.com/facebookresearch/StarSpace
- (TF-IDF: scikit-learn implementation)

+ Compressed models: Wrapper for models compressed by 'model_compression.py' module
"""

import numpy as np
import csv
import spacy
import torch
import sent2vec
import utils.utils_sent2vec as s2v

from glove import Glove
from gensim.models import KeyedVectors, Word2Vec
from fastText import load_model as ftload
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity


class InferSent:
    """Class for InferSent embedding algorithm."""
    def __init__(self, emb_path, pckl_path, gpu=False, vocab_data=None, k=100000):
        """
        InferSent initializer.
        :param emb_path: path to the embeddings file
        :param pckl_path: path to the model pickle file
        :param gpu: True if you want to use GPU for computations (default False)
        :param vocab_data: list of sentences for building vocabulary
            (if left None, <k> first word embedding vectors will be loaded as the vocabulary)
        :param k: number of vectors to load from the embeddings file when <vocab_data> is None
        """
        if gpu:
            self.model = torch.load(pckl_path, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0'})
        else:
            self.model = torch.load(pckl_path, map_location=lambda storage, loc: storage)
        self.model.set_glove_path(emb_path)
        if vocab_data:
            self.model.build_vocab(vocab_data, tokenize=True)
        else:
            self.model.build_vocab_k_words(K=k)

    def transform(self, sents):
        """
        Transform the sentences into vector representation.
        :param sents: list of sentences to transform
        :return: list of transformed sentences
        """
        return self.model.encode(sents, tokenize=True)

    def visualize(self, sent):
        """
        Visualize the importance of each word in a sentence for the algorithm.
        :param sent: input sentence
        """
        self.model.visualize(sent, tokenize=True)


class Sent2Vec:
    """Class for sent2vec embedding algorithm."""
    def __init__(self, emb_path):
        """
        Sent2vec initializer.
        :param emb_path: path to the embeddings file
        """
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(emb_path)

    def transform(self, sents):
        """
        Transform the sentences into vector representation.
        The sentences are tokenized using Tweet tokenizer.
        :param sents: list of sentences to transform
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        transformed = []
        for s in sents:
            transformed.append(self.model.embed_sentence(s))
        return transformed


class GloVe:
    """Class for GloVe embedding algorithm from glove-python package."""
    def __init__(self, emb_path):
        """
        GloVe initializer.
        :param emb_path: path to the embeddings file
        """
        self.model = Glove.load_stanford(emb_path)
        self.dim = self.model.word_vectors.shape[1]

    def transform(self, sents, verbose=False):
        """
        Transform the sentences into vector representation.
        Creates an average of word embeddings for each sentence.
        The sentences are tokenized using Tweet tokenizer.
        :param sents: list of sentences to transform
        :param verbose: print tokens not found in the vocabulary
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        vecs = []
        oov_tabu = []
        for s in sents:
            tokens = s.split()
            word_vecs = []
            for token in tokens:
                try:
                    idx = self.model.dictionary[token]
                    word_vecs.append(self.model.word_vectors[idx])
                except KeyError:
                    if verbose and token not in oov_tabu:
                        oov_tabu.append(token)
                        print('-- warning:', token, 'not in vocabulary!')
            if len(word_vecs) == 0:
                word_vecs.append(np.zeros((self.dim,)))
            vecs.append(np.average(word_vecs, axis=0))
        return vecs

    # yields worse results than transform
    def transform_native(self, sents):
        """
        Transform the sentences into vector representation (native glove-python implementation).
        The sentences are tokenized using Tweet tokenizer.
        :param sents: list of sentences to transform
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        vecs = []
        for s in sents:
            tokens = s.split()
            vecs.append(self.model.transform_paragraph(tokens, ignore_missing=True))
        return vecs


class GloVeSpacy:
    """Class for GloVe embedding algorithm included in SpaCy package."""
    def __init__(self, emb_path):
        """
        GloVeSpacy initializer.
        :param emb_path: path to the embeddings file
        """
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

    def transform(self, sents):
        """
        Transform the sentences into vector representation (native SpaCy implementation).
        :param sents: list of sentences to transform
        :return: list of transformed sentences
        """
        vecs = []
        for s in sents:
            s = s.lower()
            tmp = self.model(s)
            vecs.append(tmp.vector)
        return vecs


class WordEmbedding:
    """Class for word2vec and FastText embedding algorithms contained in Gensim package."""
    def __init__(self, emb_path, algorithm='word2vec', gensim_trained=False, k=None):
        """
        WordEmbedding initializer.
        :param emb_path: path to the embeddings file
        :param algorithm: word2vec or fasttext (default word2vec)
        :param gensim_trained: the model was trained using Gensim
        :param k: number of vectors to load to vocabulary
        """
        assert algorithm in ['word2vec', 'fasttext']
        if algorithm == 'word2vec':
            if gensim_trained:
                self.model = Word2Vec.load(emb_path)
            else:
                self.model = KeyedVectors.load_word2vec_format(emb_path, binary=True, limit=k)
        elif algorithm == 'fasttext':
            self.model = KeyedVectors.load_word2vec_format(emb_path, limit=k)
        self.stemmer = SnowballStemmer('english')
        self.gensim_trained = gensim_trained

    def transform(self, sents, verbose=False):
        """
        Transform the sentences into vector representation.
        Creates an average of word embeddings for each sentence.
        The sentences are tokenized using Tweet tokenizer.
        Tries to find matches for OOV words by stemming them.
        :param sents: list of sentences to transform
        :param verbose: print tokens not found in the vocabulary
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        vecs = []
        oov_tabu = []
        for s in sents:
            tokens = s.split()
            word_vecs = []
            for token in tokens:
                try:
                    if self.gensim_trained:
                        word_vecs.append(self.model.wv[token])
                    else:
                        word_vecs.append(self.model.word_vec(token))
                except KeyError:
                    stemmed = self.stemmer.stem(token)
                    if token != stemmed:
                        try:
                            if self.gensim_trained:
                                word_vecs.append(self.model.wv[stemmed])
                            else:
                                word_vecs.append(self.model.word_vec(stemmed))
                        except KeyError:
                            if verbose and token not in oov_tabu:
                                oov_tabu.append(token)
                                print('-- warning:', token, '/', stemmed, 'not in vocabulary!')
                    elif verbose and token not in oov_tabu:
                        oov_tabu.append(token)
                        print('-- warning:', token, 'not in vocabulary!')
            if len(word_vecs) == 0:
                word_vecs.append(np.zeros((self.model.vector_size,)))
            vecs.append(np.average(word_vecs, axis=0))
        return vecs


class FastText:
    """Class for FastText embedding algorithm."""
    def __init__(self, emb_path):
        """
        FastText initializer.
        :param emb_path: path to the binary embeddings file (model.bin)
        """
        self.model = ftload(emb_path)

    def transform(self, sents):
        """
        Transform the sentences into vector representation.
        Creates an average of word embeddings for each sentence.
        The sentences are tokenized using Tweet tokenizer.
        :param sents: list of sentences to transform
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        vecs = []
        for s in sents:
            tokens = s.split()
            word_vecs = []
            for token in tokens:
                word_vecs.append(self.model.get_word_vector(token))
            if len(word_vecs) == 0:
                word_vecs.append(np.zeros((self.model.get_dimension(),)))
            vecs.append(np.average(word_vecs, axis=0))
        return vecs

    # yields worse results than transform
    def transform_native(self, sents):
        """
        Transform the sentences into vector representation
        (native FastText implementation).
        :param sents: list of sentences to transform
        :return: list of transformed sentences
        """
        vecs = []
        for s in sents:
            vecs.append(self.model.get_sentence_vector(s))
        return vecs

    def classify_sentences(self, sents):
        """
        Classify sentences into classes trained in the FastText model.
        :param sents: list of sentences to classify
        :return: list of classes
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        labels = self.model.predict(sents)[0]
        labels = [w[0].replace('__label__', '') for w in labels]
        return labels

    def test_subword_similarity(self, word):
        """
        Tests the similarity of a word embedding and an embedding
        created by the average of its sub-words embeddings.
        :param word: tested word
        :return: (cosine similarity, [sub-words found])
        """
        we = self.model.get_word_vector(word)
        subwords, sids = self.model.get_subwords(word)
        sids = sids.tolist()
        subwords.pop(0)
        sids.pop(0)
        vecs = []
        for idx in sids:
            vecs.append(self.model.get_input_vector(idx))
        swe = np.average(vecs, axis=0)
        return cosine_similarity(we.reshape(1, -1), swe.reshape(1, -1)), subwords


class StarSpace:
    """Class for StarSpace embedding algorithm."""
    def __init__(self, emb_path, k=None):
        """
        StarSpace initializer.
        :param emb_path: path to the embeddings file
        :param k: number of vectors to load to vocabulary
        """
        self.vocab = {}
        self.labels = []
        self.label_vecs = []
        with open(emb_path) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for i, row in enumerate(reader):
                if k is None or i < k:
                    w = row[0]
                    vec = np.asarray(row[1:], dtype=np.float)
                    if w.startswith('__label__'):
                        w = w.replace('__label__', '')
                        self.labels.append(w)
                        self.label_vecs.append(vec)
                    else:
                        self.vocab[w] = vec
        self.label_vecs = np.asarray(self.label_vecs)
        self.dim = next(iter(self.vocab.values())).shape[0]

    def transform(self, sents, verbose=False):
        """
        Transform the sentences into vector representation.
        :param sents: list of sentences to transform
        :param verbose: print tokens not found in the vocabulary
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        vecs = []
        oov_tabu = []
        for s in sents:
            tokens = s.split()
            word_vecs = []
            for token in tokens:
                try:
                    word_vecs.append(self.vocab[token])
                except KeyError:
                    if verbose and token not in oov_tabu:
                        oov_tabu.append(token)
                        print('-- warning:', token, 'not in vocabulary!')
            if len(word_vecs) == 0:
                word_vecs.append(np.zeros((self.dim,)))
            vecs.append(np.average(word_vecs, axis=0))
        return vecs

    def classify_sentences(self, sents):
        """
        Classify sentences into classes trained in the StarSpace model.
        :param sents: list of sentences to classify
        :return: list of classes
        """
        labels = []
        sents = self.transform(sents)
        for s in sents:
            sim = cosine_similarity(s.reshape(1, -1), self.label_vecs)
            labels.append(self.labels[np.asscalar(np.argmax(sim))])
        return labels


class TFIDF:
    """Class for TF-IDF embedding algorithm."""
    def __init__(self, trn_data):
        """
        TF-IDF initializer.
        :param trn_data: training data (sentences)
        """
        self.model = TfidfVectorizer(analyzer='char',
                                     ngram_range=(1, 3),
                                     smooth_idf=1,
                                     stop_words=[''],
                                     use_idf=1)
        data = s2v.preprocess_sentences(trn_data, use_pos_tagger=False)
        self.model.fit(data)

    def transform(self, sents):
        """
        Transform the sentences into vector representation.
        :param sents: list of sentences to transform
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        return self.model.transform(sents)


class CompressedModel:
    """Wrapper for loading compressed models."""
    def __init__(self, emb_path, cb_path):
        """
        CompressedModel initializer.
        :param emb_path: path to the embeddings file
        :param cb_path: path to the codebook
        """
        self.cb = []
        with open(cb_path) as f:
            header = f.readline().split()
            self.cb_size = int(header[0])
            self.cb_dim = int(header[1])

            for line in f:
                vec = np.asarray(line.strip().split(), dtype=np.float)
                self.cb.append(vec)
        int_type = np.uint16 if self.cb_size > 256 else np.uint8

        self.vocab = {}
        self.sizes = {}
        self.labels = []
        self.label_vecs = []
        with open(emb_path) as f:
            header = f.readline().split()
            self.words = int(header[0])
            self.dim = int(header[1])
            self.normalized = True if 'NORM' in header else False
            self.distinct_cb = True if 'DIST' in header else False
            self.decode_func = self.decode_vec_distinct if self.distinct_cb else self.decode_vec

            for line in f:
                tmp = line.strip().split()
                if self.normalized:
                    w = ' '.join(tmp[:len(tmp)-self.dim-1])
                    size = float(tmp.pop())
                else:
                    w = ' '.join(tmp[:len(tmp)-self.dim])
                    size = 1

                vec = np.asarray(tmp[-self.dim:], dtype=int_type)

                if w.startswith('__label__'):
                    w = w.replace('__label__', '')
                    self.labels.append(w)
                    self.label_vecs.append(self.decode_func(vec)*size)
                else:
                    self.sizes[w] = size
                    self.vocab[w] = vec

    def transform(self, sents, verbose=False):
        """
        Transform the sentences into vector representation.
        :param sents: list of sentences to transform
        :param verbose: print tokens not found in the vocabulary
        :return: list of transformed sentences
        """
        sents = s2v.preprocess_sentences(sents, use_pos_tagger=False)
        vecs = []
        oov_tabu = []
        for s in sents:
            tokens = s.split()
            word_vecs = []
            for token in tokens:
                try:
                    tmp = self.vocab[token]
                    tmp = self.decode_func(tmp)
                    if self.normalized:
                        tmp *= self.sizes[token]
                    word_vecs.append(tmp)
                except KeyError:
                    if verbose and token not in oov_tabu:
                        oov_tabu.append(token)
                        print('-- warning:', token, 'not in vocabulary!')
            if len(word_vecs) == 0:
                word_vecs.append(np.zeros((self.dim*self.cb_dim,)))
            vecs.append(np.average(word_vecs, axis=0))
        return vecs

    def classify_sentences(self, sents):
        """
        Classify sentences into classes trained in the StarSpace model.
        :param sents: list of sentences to classify
        :return: list of classes
        """
        labels = []
        sents = self.transform(sents)
        for s in sents:
            sim = cosine_similarity(s.reshape(1, -1), self.label_vecs)
            labels.append(self.labels[np.asscalar(np.argmax(sim))])
        return labels

    def decode_vec(self, vec):
        """
        Decode compressed vector.
        :param vec: numpy vector
        :return: decoded numpy vector
        """
        out = []
        for idx in vec:
            out.append(self.cb[idx])
        return np.concatenate(out)

    def decode_vec_distinct(self, vec):
        """
        Decode compressed vector (version for distinct codebooks).
        :param vec: numpy vector
        :return: decoded numpy vector
        """
        out = []
        for i, idx in enumerate(vec):
            out.append(self.cb[i * self.cb_size + idx])
        return np.concatenate(out)
