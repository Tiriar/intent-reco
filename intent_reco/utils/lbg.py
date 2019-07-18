# -*- coding: utf-8 -*-
"""
    intent_reco.utils.lbg
    ~~~~~~~~~~~~~~~~~~~~~

    Linde-Buzo-Gray algorithm implementation.

    Inspired by https://github.com/internaut/py-lbg

    @author: tomas.brich@seznam.cz
"""

from functools import reduce
from sys import maxsize

import numpy as np

DATA_SIZE = 0


def generate_codebook(data, cb_size, eps=0.00001):
    """
    Generate codebook of size <cb_size> with convergence value <eps>. Will return a numpy array with the generated
    codebook, a list with absolute weights and a list with relative weights (the weight denotes how many vectors for
    <data> are in the proximity of the codevector).
    :param data: input data with N k-dimensional vectors
    :param cb_size: codebook size --> power-of-2-value
    :param eps: convergence value
    :return codebook of size <cb_size>, absolute weights, relative weights
    """

    global DATA_SIZE

    DATA_SIZE = len(data)
    assert DATA_SIZE > 0
    data = np.asarray(data)

    # calculate initial codevector - average vector of whole input data
    c0 = np.average(data, axis=0)
    codebook = np.asarray([c0])
    abs_weights = [DATA_SIZE]
    rel_weights = [1.0]

    # compute initial distortion
    tmp = np.sum((data - c0) ** 2, axis=1)
    avg_dist = reduce(lambda s, d: s + d / DATA_SIZE, tmp, 0.0)

    # split codevectors
    while len(codebook) < cb_size:
        codebook, abs_weights, rel_weights, avg_dist = split_codebook(data, codebook, eps, avg_dist)

    return codebook, abs_weights, rel_weights


def split_codebook(data, codebook, eps, initial_avg_dist):
    """
    Split the codebook so that each codevector in the codebook is split into two.
    :param data: input data
    :param codebook: codebook to split
    :param eps: convergence value
    :param initial_avg_dist: initial average distortion
    :return new codebook, absolute weights, relative weights and average distortion
    """

    # split codevectors
    new_codevectors = []
    for c in codebook:
        c1 = c * (1 + eps)
        c2 = c * (1 - eps)
        new_codevectors.extend((c1, c2))

    codebook = np.asarray(new_codevectors)
    len_codebook = len(codebook)
    abs_weights = [0] * len_codebook
    rel_weights = [0.0] * len_codebook

    print('- splitting to size', len_codebook)

    # minimize average distortion - k-means iteration
    avg_dist = 0
    err = eps + 1
    num_iter = 0
    while err > eps:
        # find closest codevector for each vector in data
        dist = np.full((DATA_SIZE,), maxsize, dtype=np.float)
        closest_c_list = np.zeros((DATA_SIZE,), dtype=np.int)
        for i, c in enumerate(codebook):
            dist_tmp = np.sum((data-c)**2, axis=1)
            cond = dist_tmp < dist
            dist[cond] = dist_tmp[cond]
            closest_c_list[cond] = i

        # get vector in data and its idx for each codevector
        vecs_near_c, vec_ids_near_c = dict(), dict()
        for i in range(len_codebook):
            cond = closest_c_list == i
            ids = np.where(cond)
            vec_ids_near_c[i] = ids
            vecs_near_c[i] = data[ids]
        closest_c_list = codebook[closest_c_list]

        # update codevectors
        for i in range(len_codebook):     # for each codevector index
            vecs = vecs_near_c.get(i)     # get its proximity input vectors
            num_vecs_near_c = len(vecs)
            if num_vecs_near_c > 0:
                new_c = np.average(vecs, axis=0)    # calculate the new center
                codebook[i] = new_c                 # update in codebook
                closest_c_list[vec_ids_near_c[i]] = new_c

                # update the weights
                abs_weights[i] = num_vecs_near_c
                rel_weights[i] = num_vecs_near_c / DATA_SIZE

        # recalculate average distortion value
        prev_avg_dist = avg_dist if avg_dist > 0 else initial_avg_dist
        closest_c_list = np.asarray(closest_c_list)
        tmp = np.sum((data - closest_c_list) ** 2, axis=1)
        avg_dist = reduce(lambda s, d: s + d / DATA_SIZE, tmp, 0.0)

        # recalculate the new error value
        err = (prev_avg_dist - avg_dist) / prev_avg_dist
        # print('> iteration', num_iter, 'avg_dist', avg_dist, 'prev_avg_dist', prev_avg_dist, 'err', err)

        num_iter += 1

    return codebook, abs_weights, rel_weights, avg_dist
