"""
Module for storing embedding algorithm wrappers.

Currently used algorithms:
- InferSent: https://github.com/facebookresearch/InferSent
- sent2vec: https://github.com/epfml/sent2vec
- GloVe: https://github.com/maciejkula/glove-python / SpaCy implementation
- word2vec: Gensim implementation
- FastText: https://github.com/facebookresearch/fastText / Gensim implementation
- StarSpace: https://github.com/facebookresearch/StarSpace

+ Compressed models: Wrapper for models compressed by 'model_compression.py' module
+ (TF-IDF: scikit-learn implementation)
"""

import numpy as np
import spacy
import torch
import pickle
import csv

from nltk import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import fastText
import sent2vec
from glove import Glove
from gensim.models import KeyedVectors, Word2Vec

import utils.utils_sent2vec as s2v
import utils.infersent_models as infersent_models


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
        s = s2v.tokenize(self.tokenizer, sentence, to_lower=True)
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


class InferSent:
    """
    InferSent embedding algorithm.
    :param emb_path: path to the word embeddings file
    :param pckl_path: path to the model pickle file
    :param vocab_data: list of sentences for building vocabulary
        (if left None, <k> first word embedding vectors will be loaded as the vocabulary)
    :param k: number of vectors to load from the embeddings file when <vocab_data> is None
    """
    def __init__(self, emb_path, pckl_path, vocab_data=None, k=100000):
        with open(emb_path) as f:
            header = f.readline().split()
        model_params = {'bsize': 64, 'word_emb_dim': int(header[1]), 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': 2}
        self.model = infersent_models.InferSent(model_params)
        self.model.load_state_dict(torch.load(pckl_path))
        self.model.set_w2v_path(emb_path)
        if vocab_data:
            self.model.build_vocab(vocab_data, tokenize=True)
        else:
            self.model.build_vocab_k_words(K=k)

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


class Sent2Vec:
    """
    Sent2Vec embedding algorithm.
    :param emb_path: path to the embeddings file
    """
    def __init__(self, emb_path):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(emb_path)

    def transform_sentences(self, sentences):
        """
        Transform a list of sentences into vector representation.
        :param sentences: list of sentences
        :return: list of numpy vectors
        """
        sents = s2v.preprocess_sentences(sentences, use_pos_tagger=False)
        return [self.model.embed_sentence(s) for s in sents]


class TFIDF:
    """
    TF-IDF algorithm.
    :param trn_data: training data (sentences)
    """
    def __init__(self, trn_data):
        self.model = TfidfVectorizer(analyzer='char',
                                     ngram_range=(1, 3),
                                     smooth_idf=1,
                                     stop_words=[''],
                                     use_idf=1)
        data = s2v.preprocess_sentences(trn_data, use_pos_tagger=False)
        self.model.fit(data)

    def transform_sentences(self, sentences):
        """
        Transform a list of sentences into vector representation.
        :param sentences: list of sentences
        :return: list of numpy vectors
        """
        sents = s2v.preprocess_sentences(sentences, use_pos_tagger=False)
        return self.model.transform(sents)


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


class WordEmbedding(EmbeddingModelBase):
    """
    Word2Vec and FastText embedding algorithms contained in Gensim package.
    :param emb_path: path to the embeddings file
    :param algorithm: word2vec or fasttext (default word2vec)
    :param gensim_trained: the model was trained using Gensim
    :param k: number of vectors to load to vocabulary (works only for FastText)
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, algorithm='word2vec', gensim_trained=False, k=None, verbose=False):
        super().__init__(verbose=verbose)

        assert algorithm in ['word2vec', 'fasttext']
        if algorithm == 'word2vec':
            if gensim_trained:
                self.model = Word2Vec.load(emb_path)
            else:
                self.model = KeyedVectors.load_word2vec_format(emb_path, binary=True, limit=k)
        elif algorithm == 'fasttext':
            self.model = KeyedVectors.load_word2vec_format(emb_path, limit=k)

        self.dim = self.model.vector_size
        self.stemmer = SnowballStemmer('english')
        self.gensim_trained = gensim_trained

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        try:
            if self.gensim_trained:
                return self.model.wv[word]
            else:
                return self.model.word_vec(word)
        except KeyError:
            return self.handle_oov(word)

    def handle_oov(self, word):
        """
        Function for handling out-of-vocabulary words.
        :param word: oov word
        :return: oov word numpy vector
        """
        stemmed = self.stemmer.stem(word)
        if word != stemmed:
            try:
                if self.gensim_trained:
                    return self.model.wv[stemmed]
                else:
                    return self.model.word_vec(stemmed)
            except KeyError:
                if self.verbose:
                    print('-- WARNING:', stemmed, '(stemmed) not in vocabulary!')
        return super().handle_oov(word)


class FastText(EmbeddingModelBase):
    """
    FastText embedding algorithm.
    :param emb_path: path to the binary embeddings file (model.bin)
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, verbose=False):
        super().__init__(verbose=verbose)
        self.model = fastText.load_model(emb_path)
        self.dim = self.model.get_dimension()

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        return self.model.get_word_vector(word)

    # def transform_sentence(self, sentence):
    #     """
    #     Transform the <sentence> into vector representation.
    #     :param sentence: input sentence
    #     :return: sentence numpy vector
    #     """
    #     return self.model.get_sentence_vector(sentence)

    def classify_sentences(self, sentences):
        """
        Classify sentences into classes trained in the FastText model.
        :param sentences: list of sentences to classify
        :return: list of classes
        """
        sents = s2v.preprocess_sentences(sentences, use_pos_tagger=False)
        labels = self.model.predict(sents)[0]
        return [w[0].replace('__label__', '') for w in labels]


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

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        try:
            return self.vocab[word]
        except KeyError:
            self.handle_oov(word)

    def classify_sentences(self, sentences):
        """
        Classify sentences into classes trained in the StarSpace model.
        :param sentences: list of sentences to classify
        :return: list of classes
        """
        labels = []
        vectors = self.transform_sentences(sentences)
        for vec in vectors:
            sim = cosine_similarity(vec.reshape(1, -1), self.label_vecs)
            labels.append(self.labels[np.asscalar(np.argmax(sim))])
        return labels


class CompressedModel(EmbeddingModelBase):
    """
    Wrapper for loading compressed models.
    :param emb_path: path to the embeddings file
    :param cb_path: path to the codebook
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, cb_path, verbose=False):
        super().__init__(verbose=verbose)

        self.normalized = None
        self.distinct_cb = None
        self.decode_func = None

        self.words = None
        self.vocab = {}
        self.sizes = {}

        self.cb = []
        self.cb_size = None
        self.cb_dim = None

        self.labels = []
        self.label_vecs = []

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
            for line in f:
                vec = np.asarray(line.strip().split(), dtype=np.float)
                self.cb.append(vec)
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
                    w = w.replace('__label__', '')
                    self.labels.append(w)
                    self.label_vecs.append(self.decode_func(vec)*size)
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
            if self.normalized:
                vec *= self.sizes[word]
            return vec
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
            sim = cosine_similarity(vec.reshape(1, -1), self.label_vecs)
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


class CompressedModelPickled(CompressedModel):
    """
    Wrapper for loading pickled compressed models.
    :param emb_path: path to the pickled compressed model
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, verbose=False):
        super().__init__(emb_path=emb_path, cb_path=None, verbose=verbose)

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
        self.label_vecs = data['label_vectors']


class CompressedModelOOVPickled(CompressedModelPickled):
    """
    Wrapper for loading pickled compressed models containing subword embeddings.
    :param emb_path: path to the pickled compressed model
    :param sw_path: path to the pickled compressed subwords
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, sw_path, verbose=False):
        super().__init__(emb_path=emb_path, verbose=verbose)

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
        if self.verbose:
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
        out = []
        for idx in vec:
            out.append(self.sw_cb[idx])
        return np.concatenate(out)

    def decode_sw_vec_distinct(self, vec):
        """
        Decode compressed subword vector (version for distinct codebooks).
        :param vec: numpy vector
        :return: decoded numpy vector
        """
        out = []
        for i, idx in enumerate(vec):
            out.append(self.sw_cb[i * self.sw_cb_size + idx])
        return np.concatenate(out)
