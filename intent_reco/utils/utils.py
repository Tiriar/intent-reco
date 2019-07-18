# -*- coding: utf-8 -*-
"""
    intent_reco.utils.utils
    ~~~~~~~~~~~~~~~~~~~~~~~

    Useful functions used throughout the project.

    @author: tomas.brich@seznam.cz
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_indices(inp, l):
    """
    Get indices of items in <inp> in list <l>. Gives -1 for items not present in <l>.
    :param inp: list of searched items
    :param l: list to search in
    :return: list of indices in <l>
    """

    indices = []
    for item in inp:
        try:
            idx = l.index(item)
        except ValueError:
            idx = -1
        indices.append(idx)
    return indices


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
