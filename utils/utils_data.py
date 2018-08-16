"""Functions used for data loading and pre-processing."""

import os
import csv
import struct
import pickle
import nltk.data
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
from smart_open import smart_open
from fastText import load_model as ft_load
from utils.utils import convert_numbers
from utils.utils_sent2vec import preprocess_sentences
from utils.embedding_wrappers import CompressedModel


def decode_compressed_model(model_path, cb_path, out_path):
    """
    Decodes a compressed embedding model and writes it as a text file.
    :param model_path: path to the embedding model
    :param cb_path: path to the codebook
    :param out_path: path of the output text file
    """
    model = CompressedModel(model_path, cb_path)
    out = [str(model.words) + ' ' + str(model.dim * model.cb_dim) + '\n']

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
            'label_vectors': model.label_vecs}

    with open(out_path, 'wb') as f:
        pickle.dump(data, f)


def load_model_txt(path, dim=300, k=None, header=False, normalize=False):
    """
    Loads the embedding vectors and their Euclidean norms into dictionaries.
    :param path: path to the embeddings file
    :param dim: embedding dimension
    :param k: number of vectors to load (load all if None)
    :param header: skip header
    :param normalize: normalize the vectors to unit length
    :return: [vocabulary], [vectors], [vector norms]
    """
    vocab = []
    vecs = []
    vsize = []
    with open(path, 'r', encoding='utf-8') as f:
        if header:
            next(f)
        for i, line in enumerate(f):
            if k is None or i < k:
                tmp = line.strip().split()
                vocab.append(' '.join(tmp[:len(tmp)-dim]))
                v = np.asarray(tmp[-dim:], dtype=np.float)
                n = np.linalg.norm(v)
                if normalize:
                    v /= n
                vecs.append(v)
                vsize.append(n)
            else:
                break
    vecs = np.asarray(vecs)
    return vocab, vecs, vsize


def load_model_ft_bin(path, k=None, normalize=False):
    """
    Loads the embedding vectors in FastText binary format.
    :param path: path to the embeddings file
    :param k: number of vectors to load (load all if None)
    :param normalize: normalize the vectors to unit length
    :return: [vocabulary], [vectors], [vector norms]
    """
    vocab, vecs = load_fasttext_format(path, k=k)
    vsize = []
    for i, v in enumerate(vecs):
        n = np.linalg.norm(v)
        if normalize:
            vecs[i] /= n
        vsize.append(n)
    return vocab, vecs, vsize


def load_fasttext_format(path, k=None):
    """
    Loads a FastText-format binary file.
    :param path: path to the binary file
    :param k: number of vectors to keep (still needs to load all)
    :return: [vocabulary], [vectors]
    """
    with smart_open(path, 'rb') as f:
        magic = struct_unpack(f, '@2i')[0]
        new_format = True if magic == 793712314 else False
        if new_format:
            struct_unpack(f, '@12i1d')
        else:
            struct_unpack(f, '@10i1d')

        vocab_size = struct_unpack(f, '@3i')[0]
        struct_unpack(f, '@1q')
        pruneidx_size = None
        if new_format:
            pruneidx_size, = struct_unpack(f, '@q')

        vocab = []
        for i in range(vocab_size):
            word_bytes = b''
            char_byte = f.read(1)
            while char_byte != b'\x00':
                word_bytes += char_byte
                char_byte = f.read(1)
            word = word_bytes.decode('utf8')
            count, _ = struct_unpack(f, '@qb')
            vocab.append(word)
        del vocab[0]

        if new_format:
            for i in range(pruneidx_size):
                struct_unpack(f, '@2i')
            struct_unpack(f, '@?')

        num_vectors, dim = struct_unpack(f, '@2q')
        dtype = None
        float_size = struct.calcsize('@f')
        if float_size == 4:
            dtype = np.dtype(np.float32)
        elif float_size == 8:
            dtype = np.dtype(np.float64)

        vectors = np.fromfile(f, dtype=dtype, count=num_vectors * dim)
        vectors = vectors.reshape((num_vectors, dim))
        vectors = np.delete(vectors, 0, axis=0)

    if k is not None and k < len(vocab):
        vocab = vocab[:k]
        vectors = vectors[:k]

    return vocab, vectors


def get_ft_subwords(path):
    """
    Brute-forces the subword embeddings from a FastText binary model
    and writes them to 'subwords.txt' in a word2vec format.
    The subwords are sorted by (length, frequency).
    :param path: path to the binary file
    """
    ft = ft_load(path)
    vocab = ft.get_words()
    vocab_len = len(vocab)
    subwords = {}
    sw_ids = {}

    print('Processing', vocab_len, 'words...')
    for word in tqdm(vocab, mininterval=1.0):
        sw, ids = ft.get_subwords(word)
        ids = ids.tolist()
        if ids[0] < vocab_len:
            del sw[0]
            del ids[0]

        for subword, idx in zip(sw, ids):
            if subword not in subwords:
                subwords[subword] = 1
                sw_ids[subword] = idx
            else:
                subwords[subword] += 1

    print('Computing output...')
    sw_sorted = sorted(subwords, key=lambda x: (len(x), -subwords[x]))
    out = [str(len(sw_sorted)) + ' ' + str(ft.get_dimension()) + '\n']
    for sw in tqdm(sw_sorted, mininterval=1.0):
        tmp = sw
        vec = ft.get_input_vector(sw_ids[sw])
        for num in vec:
            tmp += ' ' + str(num)
        out.append(tmp + '\n')

    print('Writing output...')
    with open('subwords.txt', 'w') as f:
        f.writelines(out)


def struct_unpack(file_handle, fmt):
    """
    Unpack binary file structure.
    :param file_handle: handle of the binary file
    :param fmt: loading options
    :return: unpacked structure
    """
    num_bytes = struct.calcsize(fmt)
    return struct.unpack(fmt, file_handle.read(num_bytes))


def load_sts(path, lower=False):
    """
    Load file in the STS Benchmark csv format.
    :param path: path to the file
    :param lower: lower-case the text
    :return: dictionary with keys 'X1', 'X2', 'y'
    """
    d = {'X1': [], 'X2': [], 'y': []}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('\t')
            if lower:
                d['X1'].append(text[5].strip().lower())
                d['X2'].append(text[6].strip().lower())
            else:
                d['X1'].append(text[5].strip())
                d['X2'].append(text[6].strip())
            d['y'].append(float(text[4]))
    return d


def load_alquist(path, lower=False):
    """
    Load file in the Alquist tsv/csv format.
    :param path: path to the file
    :param lower: lower-case the text
    :return: dictionary with keys 'X', 'y'
    """
    d = {'X': [], 'y': []}
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            d['X'].append(row[2].strip())
            d['y'].append(row[0].strip())
            if lower:
                d['X'][-1] = d['X'][-1].lower()
    return d


def sts_starspace(path, mode='train'):
    """
    Prepares STS Benchmark files in a format needed by StarSpace trainMode 3.
    :param path: path to the folder where STS files are located
    :param mode: train / dev / test
    """
    assert mode in ['train', 'dev', 'test']
    data = load_sts(os.path.join(path, 'sts-{}.csv'.format(mode)), lower=True)
    x1 = preprocess_sentences(data['X1'])
    x2 = preprocess_sentences(data['X2'])

    out = []
    for s1, s2, y in zip(x1, x2, data['y']):
        if mode not in ['train', 'dev'] or y > 4:
            s1 = convert_numbers(s1)
            s2 = convert_numbers(s2)
            out.append(s1 + '\t' + s2 + '\n')

    with open(os.path.join(path, 'StarSpace_preprocessed/sts-{}.txt'.format(mode)), 'w+') as f:
        f.writelines(out)


def sts_unsupervised(path, preprocess=False):
    """
    Prepares file for unsupervised learning based on the STS csv file.
    :param path: path to the STS train file
    :param preprocess: lowercase and tokenize the sentences with Tweet tokenizer
    """
    data = load_sts(path)
    sentences = data['X1'] + data['X2']

    tmp = os.path.split(path)
    if preprocess:
        sentences = preprocess_sentences(sentences, use_pos_tagger=False)
        opath = os.path.join(tmp[0], 'unsupervised_training', os.path.splitext(tmp[1])[0] + '-prep.txt')
    else:
        opath = os.path.join(tmp[0], 'unsupervised_training', os.path.splitext(tmp[1])[0] + '.txt')

    for i in range(len(sentences)):
        sentences[i] += '\n'

    with open(opath, 'w+') as f:
        f.writelines(sentences)


def alquist_starspace(path):
    """
    Prepares Alquist files in a format needed by StarSpace.
    :param path: path to Alquist train file
    """
    data = load_alquist(path)
    sentences = data['X']
    intents = data['y']
    sentences = preprocess_sentences(sentences, use_pos_tagger=False)

    out = []
    for s, i in zip(sentences, intents):
        out.append(s + '\t' + '__label__' + i + '\n')

    tmp = os.path.split(path)
    opath = os.path.join(tmp[0], 'StarSpace_preprocessed', os.path.splitext(tmp[1])[0] + '.txt')
    with open(opath, 'w+') as f:
        f.writelines(out)


def alquist_unsupervised(path, preprocess=False):
    """
    Prepares file for unsupervised learning based on the Alquist csv file.
    :param path: path to the Alquist train file
    :param preprocess: lowercase and tokenize the sentences with Tweet tokenizer
    """
    data = load_alquist(path)
    sentences = data['X']

    tmp = os.path.split(path)
    if preprocess:
        sentences = preprocess_sentences(sentences, use_pos_tagger=False)
        opath = os.path.join(tmp[0], 'unsupervised_training', os.path.splitext(tmp[1])[0] + '-prep.txt')
    else:
        opath = os.path.join(tmp[0], 'unsupervised_training', os.path.splitext(tmp[1])[0] + '.txt')

    endings = '.!?'
    for i, s in enumerate(sentences):
        s = ' '.join(s.split())
        if s[-1] not in endings:
            end = ' .' if preprocess else '.'
            s = s + end
        sentences[i] = s + '\n'

    with open(opath, 'w+') as f:
        f.writelines(sentences)


def read_warc(path, clean_html=True):
    """
    Generator that reads a WARC file and yields the header and the content.
    :param path: path to the WARC file
    :param clean_html: remove html tags and decode html signs
    :return: header -> dict of header entries, content -> list of content lines
    """
    with open(path) as fh:
        while True:
            try:
                line = next(fh)
                if line == 'WARC/1.0\n':
                    header = {}
                    line = next(fh)
                    while line != '\n':
                        key, value = line.split(': ', 1)
                        header[key] = value.rstrip()
                        line = next(fh)

                    content = []
                    line = next(fh)
                    while line != '\n':
                        line = line.rstrip()
                        text = BeautifulSoup(line, 'lxml').text if clean_html else line
                        content.append(text)
                        line = next(fh)

                    yield header, content

            except StopIteration:
                return


def common_crawl_unsupervised(path, k=None, one_sent=False):
    """
    Prepares file for unsupervised learning based on the Common Crawl WARC file.
    :param path: path to the WARC file
    :param k: keep only first <k> training samples
    :param one_sent: one sentence per line needed
    """
    data = read_warc(path, clean_html=True)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    tmp = os.path.split(path)
    opath = os.path.join(tmp[0], os.path.splitext(tmp[1])[0] + '.txt')

    samples = 0
    out = []
    for el in tqdm(data, mininterval=1.0):
        if k is not None and samples >= k:
            break
        if one_sent:
            content = []
            for entry in el[1]:
                content += tokenizer.tokenize(entry)
        else:
            content = el[1]
        content = preprocess_sentences(content, use_pos_tagger=False)
        samples += len(content)
        out += content
    for i in range(len(out)):
        out[i] += '\n'

    with open(opath, 'w+') as f:
        f.writelines(out)
