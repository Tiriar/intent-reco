# -*- coding: utf-8 -*-
"""
    intent_reco.model_compression
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Module for compressing the embedding models.

    @author: tomas.brich@seznam.cz
"""

import argparse
from sys import maxsize

# import matplotlib.pyplot as plt
import numpy as np
# from sklearn.decomposition import PCA

from intent_reco import DATA_DIR
from intent_reco.embeddings.compressed import pickle_compressed_model
from intent_reco.utils.data import load_model_ft_bin, load_model_txt, load_sts
from intent_reco.utils.lbg import generate_codebook
from intent_reco.utils.preprocessing import tokenize_sentences
from intent_reco.utils.utils import get_indices


def chunks(l, n):
    """
    Yields successive <n>-sized chunks from <l>.
    :param l: list of values
    :param n: chunk size
    :return: yields new chunk
    """

    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_vecs(d, n=4, limit=None, distinct=False):
    """
    Splits vectors in <d> into sub-vectors of size <n> and returns them in a list
    (or dict with a list for each sub-vector position).
    :param d: input vectors
    :param n: size of sub-vectors
    :param limit: pick a random subset from the vectors of <limit> size
    :param distinct: distinct codebooks version
    :return: list of sub-vectors
    """

    elems = d[np.random.choice(d.shape[0], limit, replace=False), :] \
        if limit is not None and limit < len(d) else d

    if distinct:
        len_pos = len(list(chunks(elems[0], n)))
        vectors = {i: [] for i in range(len_pos)}
        for v in elems:
            for i, chunk in enumerate(chunks(v, n)):
                vectors[i].append(chunk)
    else:
        vectors = []
        for v in elems:
            vectors += list(chunks(v, n))

    return vectors


def convert_vec(v, n, cdb):
    """
    Converts vector to its compressed form based on codebook.
    :param v: input vector
    :param n: size of sub-vectors
    :param cdb: codebook
    :return: converted vector
    """

    ch = np.asarray(list(chunks(v, n)))
    dist = np.full((len(ch),), maxsize, dtype=np.float)
    conv = np.zeros((len(ch),), dtype=np.int)
    for i, cd in enumerate(cdb):
        dist_tmp = np.sum((ch-cd)**2, axis=1)
        cond = dist_tmp < dist
        dist[cond] = dist_tmp[cond]
        conv[cond] = i
    return conv.tolist()


def convert_vec_distinct(v, n, cdb):
    """
    Converts vector to its compressed form based on codebook (distinct for each sub-vector position).
    :param v: input vector
    :param n: size of sub-vectors
    :param cdb: codebook dictionary
    :return: converted vector
    """

    conv = []
    for i, chunk in enumerate(list(chunks(v, n))):
        dist = np.sum((cdb[i]-chunk)**2, axis=1)
        conv.append(np.argmin(dist).item())
    return conv


def prune_by_norm(words, vectors, vsize, trn=None, keep=10000):
    """
    Prune the vocabulary based on vector norms in a way that at least one word from each training sample is kept.
    In case all samples are covered, more words are added until the resulting vocabulary has <keep> words.
    :param words: input vocabulary
    :param vectors: input embedding vectors
    :param vsize: input embedding norms
    :param trn: list of training samples (if None, words are chosen solely by norms)
    :param keep: number of words to keep (can be more based on the training set)
    :return: pruned <words>, <vectors> and <vsize>
    """

    words_keep = []

    # cover the training set
    if trn is not None:
        for el in trn:
            tokens = el.split()
            indices = get_indices(tokens, words)
            tsize = [-1 if i < 0 else vsize[i] for i in indices]
            max_idx = np.argmax(tsize).item()
            if tsize[max_idx] < 0:
                continue
            best_w = tokens[max_idx]
            if best_w not in words_keep:
                words_keep.append(best_w)

    # add words to get <keep> words
    words_sorted = [x for _, x in sorted(zip(vsize, words))]
    kept = len(words_keep) + 1
    for w in words_sorted:
        if kept > keep:
            break
        if w not in words_keep:
            words_keep.append(w)
            kept += 1

    # create the pruned lists
    words_out, vectors_out, vsize_out = [], [], []
    for i, w in enumerate(words):
        if w in words_keep:
            words_out.append(w)
            vectors_out.append(vectors[i])
            vsize_out.append(vsize[i])
    vectors_out = np.asarray(vectors_out)

    return words_out, vectors_out, vsize_out


def prune_by_trn(words, vectors, vsize, trn):
    """
    Prune the vocabulary so that only words in the training set are kept.
    :param words: input vocabulary
    :param vectors: input embedding vectors
    :param vsize: input embedding norms
    :param trn: list of training samples
    :return: pruned <words>, <vectors> and <vsize>
    """

    tokens = []
    for el in trn:
        tokens += el.split()
    tokens = set(tokens)

    words_out, vectors_out, vsize_out = [], [], []
    for i, w in enumerate(words):
        if w in tokens:
            words_out.append(w)
            vectors_out.append(vectors[i])
            vsize_out.append(vsize[i])

    return words_out, vectors_out, vsize_out


# def visualize_vectors(vs):
#     """
#     Plots vectors in <vs>.
#     :param vs: input vectors
#     """
#
#     plt.figure()
#     for v in vs:
#         plt.plot([0, v[0]], [0, v[1]])
#     plt.show()


def codebook_to_strings(codebook, out_list):
    """
    Converts codebook to a representation writable to a text file.
    :param codebook: input codebook
    :param out_list: list in which the output is stored
    """

    for code in codebook:
        tmp = ''
        for n in code:
            tmp += str(n) + ' '
        out_list.append(tmp.rstrip() + '\n')


def compress(emb_path, emb_dim=300, prune_freq=None, prune_norm=None, trn_path=None, reduce_dim=None,
             quantize=False, normalize=False, distinct=False, d_sv=5, d_cb=256, qnt_trn=10000,
             out_name='compressed', pickle_output=False, precision=5):
    """
    Main model compression function.
    :param emb_path: path to the embedding model
    :param emb_dim: input embedding dimension
    :param prune_freq: number of words to keep after pruning by vector frequency
    :param prune_norm: number of words to keep after pruning by vector norm
    :param trn_path: path to a training file - keep words present in this file
    :param reduce_dim: embedding dimension after dimensionality reduction
    :param quantize: use vector quantization
    :param normalize: normalize the vectors to unit length before quantization
    :param distinct: create a distinct codebook for each sub-vector position
    :param d_sv: size of sub-vectors the embeddings are split into
    :param d_cb: codebook size
    :param qnt_trn: maximum number of randomly picked vectors for computing the codebook
    :param out_name: name of the output model (without extension)
    :param pickle_output: create also a pickled version of the quantized model
    :param precision: maximum number of decimals used in the output model
    """

    if not quantize:
        normalize, distinct = False, False
    if reduce_dim is not None and reduce_dim >= emb_dim:
        reduce_dim = None

    out = out_name + '.txt'
    out_cb = out_name + '_cb.txt'

    trn_words = None
    if trn_path:
        trn_words = []
        with open(trn_path) as f:
            for line in f:
                trn_words += line.strip().split()
        trn_words = set(trn_words)

    print('Loading data (+ pruning vocabulary by frequency)...')
    if emb_path.endswith('.bin'):
        vocab, vecs, sizes = load_model_ft_bin(emb_path, k=prune_freq, normalize=normalize, keep=trn_words)
    else:
        vocab, vecs, sizes = load_model_txt(emb_path, k=prune_freq, normalize=normalize, dim=emb_dim,
                                            header=True, keep=trn_words)

    if prune_norm:
        # ToDo: Possibility to prune by any training set, not just STS.
        print('Pruning vocabulary by norm...')
        sts = load_sts(DATA_DIR + 'stsbenchmark/sts-train.csv')
        sts = tokenize_sentences(sts['X1'] + sts['X2'], to_lower=True)
        vocab, vecs, sizes = prune_by_norm(vocab, vecs, sizes, trn=sts, keep=prune_norm)
        # vocab, vecs, sizes = prune_by_trn(vocab, vecs, sizes, trn=sts)
        print('- pruned vocabulary size:', len(vocab))

    if reduce_dim:
        print('Reducing dimension...')
        emb_dim = reduce_dim
        # pca = PCA(n_components=reduce_dim, copy=False)
        # vecs = pca.fit_transform(vecs)
        vecs = vecs[:, :reduce_dim]

    if quantize:
        print('Computing codebook...')
        cb_out = []
        lbg_data = split_vecs(vecs, n=d_sv, limit=qnt_trn, distinct=distinct)
        if distinct:
            cb = dict()
            for pos in lbg_data:
                print('--- position:', pos, '---')
                cb[pos] = generate_codebook(lbg_data[pos], cb_size=d_cb)[0]
            for pos in cb:
                codebook_to_strings(cb[pos].round(precision), cb_out)
        else:
            cb = generate_codebook(lbg_data, cb_size=d_cb)[0]
            codebook_to_strings(cb.round(precision), cb_out)

        print('Writing codebook...')
        with open(out_cb, 'w', encoding='utf-8') as file:
            header = str(d_cb) + ' ' + str(d_sv) + '\n'
            file.write(header)
            file.writelines(cb_out)

        print('Quantizing vectors...')
        convert_func = convert_vec_distinct if distinct else convert_vec
        vecs = np.asarray([convert_func(vec, d_sv, cb) for vec in vecs])

    print('Preparing compressed model...')
    emb_out = []
    if not quantize:
        vecs = vecs.round(precision)
    for idx, word in enumerate(vocab):
        s = word
        for num in vecs[idx]:
            s += ' ' + str(num)
        if normalize:
            s += ' ' + str(round(sizes[idx], precision))
        emb_out.append(s + '\n')

    print('Writing compressed model...')
    dim = int(emb_dim / d_sv) if quantize else emb_dim
    with open(out, 'w', encoding='utf-8') as file:
        header = str(len(emb_out)) + ' ' + str(dim)
        if normalize:
            header += ' NORM'
        if distinct:
            header += ' DIST'
        header += '\n'
        file.write(header)
        file.writelines(emb_out)

    if pickle_output and quantize:
        print('Pickling...')
        pickle_compressed_model(out, out_cb, out_name + '.pickle')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding model compression (default values in [])')

    # data
    parser.add_argument('--emb_path', type=str, metavar='<STRING>', help='path to the embedding model')
    parser.add_argument('--emb_dim', type=int, metavar='<INT>', default=300, help='input embedding dimension [300]')

    # pruning
    parser.add_argument('--prune_freq', type=int, metavar='<INT>', default=None,
                        help='number of words to keep after pruning by vector frequency [not used]')
    parser.add_argument('--prune_norm', type=int, metavar='<INT>', default=None,
                        help='number of words to keep after pruning by vector norm [not used]')
    parser.add_argument('--trn_path', type=str, metavar='<STRING>', default=None,
                        help='path to a training file - keep words present in this file [not used]')

    # dimensionality reduction
    parser.add_argument('--reduce_dim', type=int, metavar='<INT>', default=None,
                        help='embedding dimension after dimensionality reduction [not used]')

    # quantization
    parser.add_argument('-q', '--quantize', action='store_true', help='use vector quantization')
    parser.add_argument('-n', '--normalize', action='store_true',
                        help='normalize the vectors to unit length before quantization '
                             '(original norm stored in the compressed model)')
    parser.add_argument('-d', '--distinct', action='store_true',
                        help='create a distinct codebook for each sub-vector position')
    parser.add_argument('--d_sv', type=int, metavar='<INT>', default=5,
                        help='size of sub-vectors the embeddings are split into [5]')
    parser.add_argument('--d_cb', type=int, metavar='<INT>', default=256, help='codebook size [256]')
    parser.add_argument('--qnt_trn', type=int, metavar='<INT>', default=10000,
                        help='maximum number of randomly picked vectors for computing the codebook [10000]')

    # output
    parser.add_argument('--out_name', type=str, metavar='<STRING>', default='compressed',
                        help='name of the output model (without extension) ["compressed"]')
    parser.add_argument('-p', '--pickle', action='store_true',
                        help='create also a pickled version of the quantized model')
    parser.add_argument('--precision', type=int, metavar='<INT>', default=5,
                        help='maximum number of decimals used in the output model [5]')

    params = parser.parse_args()
    print('Parsed arguments:\n', params, end='\n\n')

    compress(params.emb_path, params.emb_dim, params.prune_freq, params.prune_norm, params.trn_path, params.reduce_dim,
             params.quantize, params.normalize, params.distinct, params.d_sv, params.d_cb, params.qnt_trn,
             params.out_name, params.pickle, params.precision)
