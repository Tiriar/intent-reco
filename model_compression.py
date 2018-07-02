"""Module for compressing the embedding models."""

import numpy as np
import utils.lbg as lbg
import matplotlib.pyplot as plt
from sys import maxsize
from utils.utils import get_indices
from utils.utils_data import load_sts, load_model_txt, load_model_ft_bin
from utils.utils_sent2vec import preprocess_sentences


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
    # visualize_vectors(elems)
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


EMBEDDINGS = 'data/torontobooks_unigrams.bin'

DIM = 700               # Input embedding dimension
LIMIT = 100000          # Use only first <LIMIT> vectors - usually highest frequency words
PRUNE = False           # Prune the vocabulary using embedding norms
PRUNE_ONLY = True       # Only prune the vocabulary without quantization (when <PRUNE> = True)
PRUNE_KEEP = 20000      # Number of words to keep when pruning (K)
D_SV = 10               # Size of sub-vectors the embedding vectors are split into
D_CB = 128              # Codebook size
TRN_SIZE = 10000        # Maximum number of randomly picked vectors for computing the codebook
NORMALIZE = True        # Normalize the embeddings to unit length (original size stored as additional dimension)
DISTINCT_CB = False     # Create a distinct codebook for each sub-vector position
NORM_PRECISION = 5      # Number of decimals to use for writing vector norms when <NORMALIZE> = True

if PRUNE and PRUNE_ONLY:
    NORMALIZE = False

EMB_PRUNED = 'model_pruned_' + str(PRUNE_KEEP) + '.txt'
EMB_COMPRESSED = 'model_' + str(D_SV) + 'sv_' + str(D_CB) + 'cb'
EMB_COMPRESSED_CB = 'model_' + str(D_SV) + 'sv_' + str(D_CB) + 'cb'
if DISTINCT_CB:
    EMB_COMPRESSED += '_dist'
    EMB_COMPRESSED_CB += '_dist'
elif NORMALIZE:
    EMB_COMPRESSED += '_norm'
    EMB_COMPRESSED_CB += '_norm'
EMB_COMPRESSED += '.txt'
EMB_COMPRESSED_CB += '_cb.txt'

# TODO: Quantize also the vector sizes after normalization?
if __name__ == '__main__':
    print('Loading data...')
    if EMBEDDINGS.endswith('.bin'):
        vocab, vecs, sizes = load_model_ft_bin(EMBEDDINGS, k=LIMIT, normalize=NORMALIZE)
    else:
        vocab, vecs, sizes = load_model_txt(EMBEDDINGS, dim=DIM, k=LIMIT, header=True, normalize=NORMALIZE)

    if PRUNE:
        print('Pruning vocabulary...')
        sts = load_sts('data/stsbenchmark/sts-train.csv')
        sts = preprocess_sentences(sts['X1'] + sts['X2'], use_pos_tagger=False)
        vocab, vecs, sizes = prune_by_norm(vocab, vecs, sizes, trn=sts, keep=PRUNE_KEEP)
        # vocab, vecs, sizes = prune_by_trn(vocab, vecs, sizes, sts)
        print('- pruned vocabulary size:', len(vocab))

        if PRUNE_ONLY:
            print('Writing pruned embeddings...')
            emb_out = []
            for idx, word in enumerate(vocab):
                s = word
                for num in vecs[idx]:
                    s += ' ' + str(num)
                emb_out.append(s + '\n')

            with open(EMB_PRUNED, 'w+', encoding='utf-8') as file:
                file.write(str(len(vocab)) + ' ' + str(DIM) + '\n')
                file.writelines(emb_out)

    if not PRUNE or not PRUNE_ONLY:
        if DISTINCT_CB:
            print('Computing codebook...')
            lbg_data = split_vecs_distinct(vecs, n=D_SV, limit=TRN_SIZE)
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

        else:
            print('Computing codebook...')
            lbg_data = split_vecs(vecs, n=D_SV, limit=TRN_SIZE)
            cb, cb_abs_w, cb_rel_w = lbg.generate_codebook(lbg_data, cb_size=D_CB)
            print('centroid weights:', cb_rel_w)

            cb_out = []
            for c in cb:
                s = ''
                for num in c:
                    s += str(num) + ' '
                cb_out.append(s.strip() + '\n')

        print('Writing codebook...')
        with open(EMB_COMPRESSED_CB, 'w+', encoding='utf-8') as file:
            header = str(D_CB) + ' ' + str(D_SV) + '\n'
            file.write(header)
            file.writelines(cb_out)

        print('Computing compressed embeddings...')
        prec = '{0:.' + str(NORM_PRECISION) + 'f}'
        convert_func = convert_vec_distinct if DISTINCT_CB else convert_vec
        emb_out = []
        for idx, word in enumerate(vocab):
            s = word
            vec = convert_func(vecs[idx], D_SV, cb)
            for num in vec:
                s += ' ' + str(num)
            if NORMALIZE:
                s += ' ' + prec.format(sizes[idx])
            emb_out.append(s + '\n')

        print('Writing compressed embeddings...')
        with open(EMB_COMPRESSED, 'w+', encoding='utf-8') as file:
            header = str(len(emb_out)) + ' ' + str(int(DIM/D_SV))
            if NORMALIZE:
                header += ' NORM'
            if DISTINCT_CB:
                header += ' DIST'
            header += '\n'
            file.write(header)
            file.writelines(emb_out)
