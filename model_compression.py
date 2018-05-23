"""Module for compressing the embedding models."""

import lbg
import numpy as np
import matplotlib.pyplot as plt
from sys import maxsize
from random import sample, seed
from operator import itemgetter
from utils_sent2vec import preprocess_sentences


def load_embeddings(path, dim=300, k=None, header=False, normalize=False):
    """
    Loads the embedding vectors into dictionary.
    :param path: path to the embeddings file
    :param dim: embedding dimension
    :param k: number of vectors to load
    :param header: skip header
    :param normalize: normalize the vectors to unit length
    :return: dict: {'<words>': <vectors>}, dict: {'<words>': <vector sizes>}
    """
    d = {}
    vsize = {}
    i = 0
    with open(path, 'r', encoding='utf-8') as f:
        if header:
            next(f)
        for line in f:
            if k is None or i < k:
                tmp = line.strip().split()
                w = ' '.join(tmp[:len(tmp)-dim])
                v = np.asarray(tmp[-dim:], dtype=np.float)
                n = np.linalg.norm(v)
                d[w] = v / n if normalize else v
                vsize[w] = n
                i += 1
            else:
                break
    return d, vsize


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
    :param d: input dictionary
    :param n: size of sub-vectors
    :param limit: pick a random subset from the vectors of <limit> size
    :return: list of sub-vectors
    """
    if limit is not None and limit < len(d):
        elems = sample(list(d.values()), limit)
        # visualize_vectors(elems)
    else:
        elems = d.values()

    vecs = []
    for v in elems:
        vecs += list(chunks(v, n))
    return vecs


def split_vecs_distinct(d, n=4, limit=None):
    """
    Splits vectors in <d> into sub-vectors of size <n> and returns
    a list of sub-vectors for each position in a dictionary.
    :param d: input dictionary
    :param n: size of sub-vectors
    :param limit: pick a random subset from the vectors of <limit> size
    :return: dict: {<sub-vector position>: <list of sub-vectors>}
    """
    if limit is not None and limit < len(d):
        elems = sample(list(d.values()), limit)
        len_pos = len(list(chunks(elems[0], n)))
    else:
        elems = d.values()
        len_pos = len(list(chunks(next(iter(d.values())), n)))

    vecs = {}
    for i in range(len_pos):
        vecs[i] = []

    for v in elems:
        for i, chunk in enumerate(chunks(v, n)):
            vecs[i].append(chunk)
    return vecs


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
        idx = np.asscalar(np.argmin(dist))
        conv.append(idx)
    return conv


def prune_by_norm(d, vsize, trn=None, keep=10000):
    """
    Prune the vocabulary based on vector norms in a way that at least one word from each training sample is kept.
    In case all samples are covered, more words are added until the resulting vocabulary has <keep> words.
    :param d: input dictionary
    :param vsize: embedding sizes dictionary
    :param trn: list of training samples (if None, words are chosen solely by norms)
    :param keep: number of words to keep (can be more based on the training set)
    :return: pruned <d> and <vsize>
    """
    d_out = {}
    vsize_out = {}

    # cover the training set
    if trn is not None:
        for el in trn:
            tokens = el.split()
            tsize = [vsize.get(t, -1) for t in tokens]
            max_idx = int(np.argmax(tsize))
            if tsize[max_idx] < 0:
                continue
            best_w = tokens[max_idx]
            if best_w not in d_out:
                d_out[best_w] = d[best_w]
                vsize_out[best_w] = vsize[best_w]

    # add words to get <keep> words
    words_sorted = sorted(vsize.items(), key=itemgetter(1))
    kept = len(d_out) + 1
    for w in words_sorted:
        if kept > keep:
            break
        if w[0] not in d_out:
            d_out[w[0]] = d[w[0]]
            vsize_out[w[0]] = w[1]
            kept += 1

    return d_out, vsize_out


def prune_by_trn(d, vsize, trn):
    """
    Prune the vocabulary so that only words in the training set are kept.
    :param d: input dictionary
    :param vsize: embedding sizes dictionary
    :param trn: list of training samples
    :return: pruned <d> and <vsize>
    """
    d_out = {}
    vsize_out = {}

    tokens = []
    for el in trn:
        tokens += el.split()
    tokens = set(tokens)

    for w in d:
        if w in tokens:
            d_out[w] = d[w]
            vsize_out[w] = vsize[w]

    return d_out, vsize_out


def visualize_vectors(vs):
    """
    Plots vectors in <vs>.
    :param vs: input vectors
    """
    plt.figure()
    for v in vs:
        plt.plot([0, v[0]], [0, v[1]])
    plt.show()


FASTTEXT_EMBEDDINGS_VEC = 'data/wiki.en.vec'
TRN_PATH = 'data/trn_set.csv'

DIM = 300               # Input embedding dimension
LIMIT = 200000          # Use only first <LIMIT> vectors - usually highest frequency words
PRUNE = False           # Prune the vocabulary using embedding norms
PRUNE_ONLY = True       # Only prune the vocabulary without quantization (when PRUNE = True)
PRUNE_KEEP = 20000      # Number of words to keep when pruning (K)
D_SV = 4                # Size of sub-vectors the embedding vectors are split into
D_CB = 8                # Codebook size
TRN_SIZE = 10000        # Maximum number of randomly picked vectors for computing the codebook
NORMALIZE = True        # Normalize the embeddings to unit length (original size stored as additional dimension)
DISTINCT_CB = False     # Create a distinct codebook for each sub-vector position

if PRUNE and PRUNE_ONLY:
    NORMALIZE = False

FT_PRUNED = 'model_pruned_' + str(PRUNE_KEEP) + '.txt'
FT_COMPRESSED = 'model_' + str(D_SV) + 'sv_' + str(D_CB) + 'cb'
FT_COMPRESSED_CB = 'model_' + str(D_SV) + 'sv_' + str(D_CB) + 'cb'
if DISTINCT_CB:
    FT_COMPRESSED += '_dist'
    FT_COMPRESSED_CB += '_dist'
elif NORMALIZE:
    FT_COMPRESSED += '_norm'
    FT_COMPRESSED_CB += '_norm'
FT_COMPRESSED += '.txt'
FT_COMPRESSED_CB += '_cb.txt'

# TODO: Quantize also the vector sizes after normalization?
if __name__ == '__main__':
    seed(1)

    print('Loading data...')
    data, sizes = load_embeddings(FASTTEXT_EMBEDDINGS_VEC, dim=DIM, k=LIMIT, header=True, normalize=NORMALIZE)
    if PRUNE:
        print('Pruning vocabulary...')
        with open(TRN_PATH) as fd:
            train = fd.readlines()
        train = preprocess_sentences(train, use_pos_tagger=False)
        data, sizes = prune_by_norm(data, sizes, trn=train, keep=PRUNE_KEEP)
        # data, sizes = prune_by_trn(data, sizes, train)
        print('- pruned vocabulary size:', len(data))

        if PRUNE_ONLY:
            print('Writing pruned embeddings...')
            emb_out = []
            for key, value in data.items():
                s = key
                for num in value:
                    s += ' ' + str(num)
                emb_out.append(s + '\n')

            with open(FT_PRUNED, 'w+', encoding='utf-8') as file:
                file.write(str(len(data)) + ' ' + str(DIM) + '\n')
                file.writelines(emb_out)

    if not PRUNE or not PRUNE_ONLY:
        if DISTINCT_CB:
            print('Computing codebook...')
            lbg_data = split_vecs_distinct(data, n=D_SV, limit=TRN_SIZE)
            cb = {}
            for pos in lbg_data:
                print('--- position:', pos, '---')
                cb[pos] = lbg.generate_codebook(lbg_data[pos], cb_size=D_CB)[0]

            cb_out = []
            for pos in cb:
                for c in cb[pos]:
                    s = ''
                    for num in c:
                        s += str(num) + ' '
                    cb_out.append(s.strip() + '\n')

            print('Computing compressed embeddings...')
            dout = {}
            for key in data:
                dout[key] = convert_vec_distinct(data[key], D_SV, cb)
                if NORMALIZE:
                    dout[key].append(sizes[key])

        else:
            print('Computing codebook...')
            lbg_data = split_vecs(data, n=D_SV, limit=TRN_SIZE)
            cb, cb_abs_w, cb_rel_w = lbg.generate_codebook(lbg_data, cb_size=D_CB)
            print('centroid weights:', cb_rel_w)

            cb_out = []
            for c in cb:
                s = ''
                for num in c:
                    s += str(num) + ' '
                cb_out.append(s.strip() + '\n')

            print('Computing compressed embeddings...')
            dout = {}
            for key in data:
                dout[key] = convert_vec(data[key], D_SV, cb)
                if NORMALIZE:
                    dout[key].append(sizes[key])

        print('Writing codebook...')
        with open(FT_COMPRESSED_CB, 'w+', encoding='utf-8') as file:
            file.writelines(cb_out)

        print('Writing compressed embeddings...')
        emb_out = []
        for key, value in dout.items():
            s = key
            for num in value:
                s += ' ' + str(num)
            emb_out.append(s + '\n')

        with open(FT_COMPRESSED, 'w+', encoding='utf-8') as file:
            file.writelines(emb_out)
