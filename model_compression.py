"""Module for compressing the embedding models."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from sys import maxsize
# from sklearn.decomposition import PCA

from utils.lbg import generate_codebook
from utils.utils import get_indices
from utils.data import load_sts, load_model_txt, load_model_ft_bin
from utils.preprocessing import tokenize_sentences
from embeddings.compressed import pickle_compressed_model


def chunks(l, n):
    """
    Yields successive n-sized chunks from l.
    :param l: list of values
    :param n: chunk size
    :return: yields new chunk
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def split_vecs(d, n=4, limit=None):
    """
    Splits vectors in <d> into sub-vectors of size <n> and returns them in a list.
    :param d: input vectors
    :param n: size of sub-vectors
    :param limit: pick a random subset from the vectors of <limit> size
    :return: list of sub-vectors
    """
    if limit is not None and limit < len(d):
        elems = d[np.random.choice(d.shape[0], limit, replace=False), :]
    else:
        elems = d
    vectors = []
    for v in elems:
        vectors += list(chunks(v, n))
    return vectors


def split_vecs_distinct(d, n=4, limit=None):
    """
    Splits vectors in <d> into sub-vectors of size <n> and returns
    a list of sub-vectors for each sub-vector position.
    :param d: input vectors
    :param n: size of sub-vectors
    :param limit: pick a random subset from the vectors of <limit> size
    :return: dict: {<sub-vector position>: <list of sub-vectors>}
    """
    if limit is not None and limit < len(d):
        elems = d[np.random.choice(d.shape[0], limit, replace=False), :]
    else:
        elems = d
    len_pos = len(list(chunks(elems[0], n)))

    vectors = {}
    for i in range(len_pos):
        vectors[i] = []

    for v in elems:
        for i, chunk in enumerate(chunks(v, n)):
            vectors[i].append(chunk)
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
        j = np.asscalar(np.argmin(dist))
        conv.append(j)
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
            tsize = []
            for i in indices:
                tsize.append(-1 if i < 0 else vsize[i])
            max_idx = int(np.argmax(tsize))
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
    words_out = []
    vectors_out = []
    vsize_out = []
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

    words_out = []
    vectors_out = []
    vsize_out = []
    for i, w in enumerate(words):
        if w in tokens:
            words_out.append(w)
            vectors_out.append(vectors[i])
            vsize_out.append(vsize[i])

    return words_out, vectors_out, vsize_out


def visualize_vectors(vs):
    """
    Plots vectors in <vs>.
    :param vs: input vectors
    """
    plt.figure()
    for v in vs:
        plt.plot([0, v[0]], [0, v[1]])
    plt.show()


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


parser = argparse.ArgumentParser(description='Embedding model compression')

# data
parser.add_argument('--emb_path', type=str, metavar='<STRING>', default='data/twitter_unigrams.bin',
                    help='path to the embedding model')
parser.add_argument('--emb_dim', type=int, metavar='<INT>', default=700,
                    help='input embedding dimension [700]')

# pruning
parser.add_argument('--prune_freq', type=int, metavar='<INT>', default=None,
                    help='number of words to keep after pruning by vector frequency [no pruning]')
parser.add_argument('--prune_norm', type=int, metavar='<INT>', default=None,
                    help='number of words to keep after pruning by vector norm [no pruning]')
parser.add_argument('-t', '--trn_keep', action='store_true',
                    help='keep words present in a training set')
parser.add_argument('--trn_path', type=str, metavar='<STRING>',
                    default='data/stsbenchmark/unsupervised_training/sts-train-prep.txt',
                    help='path to the training file (tokenized plain text) used for <trn_keep>')

# dimensionality reduction
parser.add_argument('-r', '--reduce_dim', action='store_true',
                    help='apply dimensionality reduction')
parser.add_argument('--dim', type=int, metavar='<INT>', default=100,
                    help='embedding dimension after dimensionality reduction [100]')

# quantization
parser.add_argument('-q', '--quantize', action='store_true',
                    help='use vector quantization')
parser.add_argument('-n', '--normalize', action='store_true',
                    help='normalize the vectors to unit length before quantization '
                         '(original norm stored in the compressed model)')
parser.add_argument('-d', '--distinct', action='store_true',
                    help='create a distinct codebook for each sub-vector dimension')
parser.add_argument('--d_sv', type=int, metavar='<INT>', default=10,
                    help='size of sub-vectors the embeddings are split into [10]')
parser.add_argument('--d_cb', type=int, metavar='<INT>', default=128,
                    help='codebook size [128]')
parser.add_argument('--qnt_trn', type=int, metavar='<INT>', default=10000,
                    help='maximum number of randomly picked vectors for computing the codebook [10000]')

# output
parser.add_argument('--out_name', type=str, metavar='<STRING>', default='model_compressed',
                    help='name of the output model (without extension) [\'model_compressed\']')
parser.add_argument('-p', '--pickle', action='store_true',
                    help='create also a pickled version of the quantized model')
parser.add_argument('--precision', type=int, metavar='<INT>', default=5,
                    help='maximum number of decimals used in the output model [5]')

params = parser.parse_args()
print('Parsed arguments:\n', params, end='\n\n')
if not params.quantize:
    params.normalize = False
    params.distinct = False
if params.dim > params.emb_dim:
    params.reduce_dim = False

OUT = params.out_name + '.txt'
OUT_CB = params.out_name + '_cb.txt'
OUT_PKL = params.out_name + '.pickle'
prec = params.precision

if __name__ == '__main__':
    if params.trn_keep:
        trn_words = []
        with open(params.trn_path) as f:
            for line in f:
                trn_words += line.strip().split()
        trn_words = set(trn_words)
    else:
        trn_words = None

    print('Loading data (+ pruning vocabulary by frequency)...')
    if params.emb_path.endswith('.bin'):
        vocab, vecs, sizes = load_model_ft_bin(params.emb_path, k=params.prune_freq, normalize=params.normalize,
                                               keep=trn_words)
    else:
        vocab, vecs, sizes = load_model_txt(params.emb_path, k=params.prune_freq, normalize=params.normalize,
                                            dim=params.emb_dim, header=True, keep=trn_words)

    if params.prune_norm:
        # TODO: Possibility to prune by any training set, not just STS.
        print('Pruning vocabulary by norm...')
        sts = load_sts('data/stsbenchmark/sts-train.csv')
        sts = tokenize_sentences(sts['X1'] + sts['X2'], to_lower=True)
        vocab, vecs, sizes = prune_by_norm(vocab, vecs, sizes, trn=sts, keep=params.prune_norm)
        # vocab, vecs, sizes = prune_by_trn(vocab, vecs, sizes, trn=sts)
        print('- pruned vocabulary size:', len(vocab))

    if params.reduce_dim:
        print('Reducing dimension...')
        params.emb_dim = params.dim
        # pca = PCA(n_components=params.dim, copy=False)
        # vecs = pca.fit_transform(vecs)
        vecs = vecs[:, :params.dim]

    if params.quantize:
        # TODO: Quantize also the vector sizes after normalization?
        print('Computing codebook...')
        cb_out = []
        if params.distinct:
            lbg_data = split_vecs_distinct(vecs, n=params.d_sv, limit=params.qnt_trn)
            cb = {}
            for pos in lbg_data:
                print('--- position:', pos, '---')
                cb[pos] = generate_codebook(lbg_data[pos], cb_size=params.d_cb)[0]
            for pos in cb:
                codebook_to_strings(cb[pos].round(prec), cb_out)
        else:
            lbg_data = split_vecs(vecs, n=params.d_sv, limit=params.qnt_trn)
            cb = generate_codebook(lbg_data, cb_size=params.d_cb)[0]
            codebook_to_strings(cb.round(prec), cb_out)

        print('Writing codebook...')
        with open(OUT_CB, 'w', encoding='utf-8') as file:
            header = str(params.d_cb) + ' ' + str(params.d_sv) + '\n'
            file.write(header)
            file.writelines(cb_out)

        print('Quantizing vectors...')
        convert_func = convert_vec_distinct if params.distinct else convert_vec
        vecs_quantized = []
        for vec in vecs:
            vecs_quantized.append(convert_func(vec, params.d_sv, cb))
        vecs = np.asarray(vecs_quantized)

    print('Preparing compressed model...')
    emb_out = []
    if not params.quantize:
        vecs = vecs.round(prec)
    for idx, word in enumerate(vocab):
        s = word
        for num in vecs[idx]:
            s += ' ' + str(num)
        if params.normalize:
            s += ' ' + str(round(sizes[idx], prec))
        emb_out.append(s + '\n')

    print('Writing compressed model...')
    dim = int(params.emb_dim/params.d_sv) if params.quantize else params.emb_dim
    with open(OUT, 'w', encoding='utf-8') as file:
        header = str(len(emb_out)) + ' ' + str(dim)
        if params.normalize:
            header += ' NORM'
        if params.distinct:
            header += ' DIST'
        header += '\n'
        file.write(header)
        file.writelines(emb_out)

    if params.pickle and params.quantize:
        print('Pickling...')
        pickle_compressed_model(OUT, OUT_CB, OUT_PKL)
