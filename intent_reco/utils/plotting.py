# -*- coding: utf-8 -*-
"""
    intent_reco.utils.plotting
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Functions used for plotting results.

    @author: tomas.brich@seznam.cz
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def load_compression_results():
    """
    Loads the compression results from results.txt for plotting.
    :return: dictionary of loaded data
    """

    d = {'base': {'chunks': [], 'cb_size': [], 'size': [], 'sp': [], 'pr': []},
         'norm': {'chunks': [], 'cb_size': [], 'size': [], 'sp': [], 'pr': []},
         'dist': {'chunks': [], 'cb_size': [], 'size': [], 'sp': [], 'pr': []},
         'n_prune': {'K': [], 'size': [], 'sp': [], 'pr': []},
         'f_prune': {'K': [], 'size': [], 'sp': [], 'pr': []}}

    with open('results.txt', 'r') as f:
        for line in f:
            if line == '- VOCABULARY PRUNING USING VECTOR NORM:\n':
                break
        [next(f) for _ in range(7)]

        for line in f:
            if line == '- VOCABULARY PRUNING USING WORD FREQUENCY:\n':
                break
            extract_compression_data(line, d, 'n_prune')
        [next(f) for _ in range(2)]

        for line in f:
            if line == '- QUANTIZATION WITHOUT NORMALIZATION:\n':
                break
            extract_compression_data(line, d, 'f_prune')
        [next(f) for _ in range(2)]

        for line in f:
            if line == '- QUANTIZATION WITH NORMALIZATION:\n':
                break
            extract_compression_data(line, d, 'base')
        [next(f) for _ in range(2)]

        for line in f:
            if line == '- QUANTIZATION WITH NORMALIZATION AND DISTINCT CBs:\n':
                break
            extract_compression_data(line, d, 'norm')
        [next(f) for _ in range(2)]

        for line in f:
            if '=' in line:
                break
            extract_compression_data(line, d, 'dist')
    return d


def extract_compression_data(s, d, mode):
    """
    Extracts compression data from a line in results.txt and saves it to <d>.
    :param s: string - one line from results.txt
    :param d: data dictionary
    :param mode: 'n_prune' or 'f_prune' - vocabulary pruning
                 'base', 'norm' or 'dist' - quantization
    """

    tmp = s.split('|')
    if len(tmp) < 7 or '-' in tmp[2]:
        return
    tmp = [x.strip() for x in tmp]

    if mode in ['n_prune', 'f_prune']:
        tmp[0] = tmp[0].replace('(', ' ').replace(')', ' ')
        d[mode]['K'].append(int(tmp[0].split()[1]))
    else:
        tmp[0] = tmp[0].replace('-', ' ').replace('(', ' ')
        cut = [int(s) for s in tmp[0].split() if s.isdigit()]
        d[mode]['chunks'].append(int(cut[0]))
        d[mode]['cb_size'].append(int(cut[1]))

    d[mode]['size'].append(int(tmp[2]))
    d[mode]['sp'].append(float(tmp[3]))
    d[mode]['pr'].append(float(tmp[5]))


def plot_quantization_results():
    """Plot quantization results."""

    titles = {'base': 'Without normalization',
              'norm': 'With normalization',
              'dist': 'With normalization and distinct codebooks'}
    d = load_compression_results()

    out = dict()
    for mode in titles.keys():
        out[mode] = dict()
        for i, ch in enumerate(d[mode]['chunks']):
            if ch not in out[mode].keys():
                out[mode][ch] = [[], []]
            out[mode][ch][0].append(d[mode]['cb_size'][i])
            out[mode][ch][1].append(d[mode]['pr'][i])

    for mode in out.keys():
        plt.figure()
        # plt.title(titles[mode])
        plt.xlabel('$D_{CB}$ [-]', fontsize=14)
        plt.ylabel('Pearson coefficient on STS test [-]', fontsize=14)
        plt.axhline(0.582, color='k')
        for ch in out[mode]:
            plt.plot(out[mode][ch][0], out[mode][ch][1], label='$D_{SV}$ = ' + str(ch), lw=2)
        plt.xscale('log', basex=2)
        plt.xlim((2**1, 2**9))
        plt.ylim((0.37, 0.6))
        plt.grid(True)
        plt.legend(fontsize=14, loc='lower right')

    plt.show()


def plot_pruning_results():
    """Plot vocabulary pruning results."""

    d = load_compression_results()
    labels = {'n_prune': 'pruning by vector norm',
              'f_prune': 'pruning by word frequency'}

    plt.figure()
    # plt.title('Vocabulary pruning')
    plt.xlabel('$K$ [-]', fontsize=14)
    plt.ylabel('Pearson coefficient on STS test [-]', fontsize=14)
    plt.axhline(0.582, color='k')
    for mode in labels.keys():
        plt.plot(d[mode]['K'], d[mode]['pr'], label=labels[mode], lw=2)
    plt.xlim((0, 160e3))
    plt.ylim((0.2, 0.6))
    plt.grid(True)
    plt.legend(fontsize=14, loc='lower right')

    plt.show()


def load_training_results():
    """
    Loads the training results from results.txt for plotting.
    :return: dictionary of loaded data
    """

    d = {'sts': {}, 'c4c': {}}
    with open('results.txt', 'r') as f:
        for line in f:
            if 'MY TRAINED MODELS' in line:
                break
        [next(f) for _ in range(2)]

        alg = ''
        for line in f:
            if '-------' in line:
                break
            alg = extract_training_data(line, d, 'sts', alg)

        for line in f:
            if '-------' in line:
                break
            alg = extract_training_data(line, d, 'c4c', alg)
    return d


def extract_training_data(s, d, mode, alg):
    """
    Extracts training data from a line in results.txt and saves it to <d>.
    :param s: string - one line from results.txt
    :param d: data dictionary
    :param mode: 'sts' or 'c4c'
    :param alg: last algorithm name
    :return: current algorithm name
    """

    tmp = s.split('|')
    if len(tmp) < 6 or '-' in tmp[3]:
        return
    tmp = [x.strip() for x in tmp]

    alg = alg if tmp[0] == '' else tmp[0]
    if alg not in d[mode]:
        d[mode][alg] = {'epochs': [], 'sp': [], 'pr': [], 'time': [], 'perc': []}

    if mode == 'c4c':
        perc = tmp[1].split()
        perc = 100 if len(perc) < 3 else int(perc[-1].replace('%', ''))
        d[mode][alg]['perc'].append(perc)

    d[mode][alg]['epochs'].append(int(tmp[2].split()[1].replace('e', '')))
    d[mode][alg]['sp'].append(float(tmp[3]))
    d[mode][alg]['pr'].append(float(tmp[4]))

    tmp[5] = tmp[5].replace('s', '').split('m')
    d[mode][alg]['time'].append(float(tmp[5][0]) + float(tmp[5][1]) / 60)

    return alg


def plot_training_results():
    """Plot training results."""

    d = load_training_results()

    plt.figure()
    # plt.title('Training models on STS Benchmark')
    plt.xlabel('epochs [-]', fontsize=14)
    plt.ylabel('Pearson coefficient on STS test [-]', fontsize=14)
    for alg in d['sts'].keys():
        plt.plot(d['sts'][alg]['epochs'], d['sts'][alg]['pr'], label=alg, lw=2)
    plt.ylim((0, 0.7))
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.figure()
    # plt.title('Training models on STS Benchmark')
    plt.xlabel('training CPU time [minutes]', fontsize=14)
    plt.ylabel('Pearson coefficient on STS test [-]', fontsize=14)
    for alg in d['sts'].keys():
        plt.plot(d['sts'][alg]['time'], d['sts'][alg]['pr'], label=alg, lw=2)
    plt.ylim((0, 0.7))
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.figure()
    # plt.title('Training models on C4Corpus')
    plt.xlabel('percentage of data [%]', fontsize=14)
    plt.ylabel('Pearson coefficient on STS test [-]', fontsize=14)
    for alg in d['c4c'].keys():
        plt.plot(d['c4c'][alg]['perc'], d['c4c'][alg]['pr'], label=alg, lw=2)
    plt.xlim((0, 100))
    plt.ylim((0, 0.8))
    plt.xticks(d['c4c']['Sent2Vec unigram']['perc'])
    plt.grid(True)
    plt.legend(fontsize=14)

    plt.show()


def score_hist(score):
    """
    Plots a histogram of given scores.
    :param score: score vector
    """

    plt.figure()
    plt.hist(score, bins='auto')
    plt.show()


def score_scatter(y, yhat):
    """
    Plots a scatter graph comparing computed and target score.
    :param y: target score
    :param yhat: computed score
    """

    plt.figure()
    plt.scatter(y, yhat)
    plt.plot([0, 5], [0, 5], 'g')
    plt.xlabel('target score')
    plt.ylabel('computed score')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.show()


def frequency_norm_graph(path, dim=300, header=False, freq=1000):
    """
    Plots a word frequency / word embedding norm graph based on an embedding file.
    :param path: path to the embedding file in text format (assuming sorted by word frequency)
    :param dim: dimension of the embeddings
    :param header: the file includes a one-line header
    :param freq: evaluate only every <freq>-th sample
    """

    norms = []
    i = [0, 0]
    with open(path, 'r') as f:
        if header:
            next(f)
        for line in f:
            if not i[0] % freq:
                i[1] += 1
                line = line.split()
                vec = np.array(line[-dim:]).astype(np.float)
                norms.append(np.linalg.norm(vec))
            i[0] += 1
    print(i[1], 'word embeddings evaluated.')

    plt.plot(norms, linewidth=2)
    plt.xlabel('index in the embedding file')
    plt.ylabel('word embedding norm')
    plt.grid(True)
    plt.show()


def pearson_vs_spearman():
    """Plots a comparison between Pearson and Spearman correlation coefficients."""

    plt.figure()
    x = np.linspace(-1.5, 1.5, 30)

    y = np.linspace(-1.5, 1.5, 30)
    pr = pearsonr(x, y)[0]
    sp = spearmanr(x, y)[0]
    plt.subplot(2, 3, 1)
    plt.plot([-1.5, 1.5], [-1.5, 1.5], 'g', lw=3, zorder=1)
    plt.scatter(x, y, 50, 'b', zorder=2)
    plt.xticks(()), plt.yticks(())
    plt.title('PR = {:+.3f}, SP = {:+.3f}'.format(pr, sp), fontweight='bold', size=18)

    y = np.linspace(1.5, -1.5, 30)
    pr = pearsonr(x, y)[0]
    sp = spearmanr(x, y)[0]
    plt.subplot(2, 3, 2)
    plt.plot([-1.5, 1.5], [1.5, -1.5], 'g', lw=3, zorder=1)
    plt.scatter(x, y, 50, 'b', zorder=2)
    plt.xticks(()), plt.yticks(())
    plt.title('PR = {:+.3f}, SP = {:+.3f}'.format(pr, sp), fontweight='bold', size=18)

    y = x**3
    pr = pearsonr(x, y)[0]
    sp = spearmanr(x, y)[0]
    plt.subplot(2, 3, 3)
    plt.plot([-1.5, 1.5], [-1.5, 1.5], 'g', lw=3, zorder=1)
    plt.scatter(x, y, 50, 'b', zorder=2)
    plt.xticks(()), plt.yticks(())
    plt.title('PR = {:+.3f}, SP = {:+.3f}'.format(pr, sp), fontweight='bold', size=18)

    y = 1 - (x + 0.5) ** 2
    y[:10] = 1
    pr = pearsonr(x, y)[0]
    sp = spearmanr(x, y)[0]
    plt.subplot(2, 3, 4)
    plt.plot([-1.5, 1.5], [1.5, -1.5], 'g', lw=3, zorder=1)
    plt.scatter(x, y, 50, 'b', zorder=2)
    plt.xticks(()), plt.yticks(())
    plt.title('PR = {:+.3f}, SP = {:+.3f}'.format(pr, sp), fontweight='bold', size=18)

    y = x ** 2
    pr = pearsonr(x, y)[0]
    sp = spearmanr(x, y)[0]
    plt.subplot(2, 3, 5)
    plt.scatter(x, y, 50, 'b')
    plt.xticks(()), plt.yticks(())
    plt.title('PR = {:+.3f}, SP = {:+.3f}'.format(pr, sp), fontweight='bold', size=18)

    y = np.random.random(30) * 1.5
    pr = pearsonr(x, y)[0]
    sp = spearmanr(x, y)[0]
    plt.subplot(2, 3, 6)
    plt.scatter(x, y, 50, 'b', zorder=2)
    plt.xticks(()), plt.yticks(())
    plt.title('PR = {:+.3f}, SP = {:+.3f}'.format(pr, sp), fontweight='bold', size=18)

    plt.show()
