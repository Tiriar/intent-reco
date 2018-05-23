"""Useful functions used throughout the project."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec


def is_number(s):
    """
    Checks if input string is a number.
    :param s: input string
    :return: True if <s> is a number / False
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def convert_numbers(s, token='<num>'):
    """
    Converts numbers in a given sentence into a token.
    :param s: input sentence
    :param token: token for conversion (default '<num>')
    :return: converted sentence
    """
    s = s.split()
    for i in range(len(s)):
        if is_number(s[i]):
            s[i] = token
    return ' '.join(s)


def normalize_score(x, n_max=5):
    """
    Normalize the input score vector between 0 and <n_max>.
    :param x: vector to normalize
    :param n_max: max score
    :return: normalized vector
    """
    max_score = max(x)
    min_score = min(x)
    return n_max * (x - min_score) / (max_score - min_score)


def compute_score_pairs(v1, v2):
    """
    Compute normalized score based on cosine similarity between two lists of vectors.
    :param v1: first list of vectors
    :param v2: second list of vectors
    :return: list of normalized scores
    """
    sc = np.diag(cosine_similarity(v1, v2))
    return normalize_score(sc.ravel())


def train_word2vec(path, name='w2v_model', dim=300, epoch=16, hs=0, neg=5, sg=0, threads=3):
    """
    Trains a word2vec model.
    :param path: path to the training file
    :param name: name of the output model file
    :param dim: embedding dimension
    :param epoch: number of epochs (data passes)
    :param hs: 1 --> use hierarchical softmax
               0 --> use negative sampling if <neg> > 0
    :param neg: how many negatives should be sampled
    :param sg: 1 --> Skip-Gram
               0 --> CBOW
    :param threads: number of CPU threads to use
    """
    if hs > 0:
        neg = 0

    sentences = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            sentences.append(line)
    model = Word2Vec(sentences, min_count=1, size=dim, iter=epoch,
                     hs=hs, negative=neg, sg=sg, workers=threads)
    model.save(name)
