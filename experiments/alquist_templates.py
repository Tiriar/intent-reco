# -*- coding: utf-8 -*-
"""
    experiments.alquist_templates
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Module for plotting a learning curve for intent recognition on the Alquist dataset.

    The scikit-learn module is not used for this purpose as we want to see the learning curves for each intent
    individually and we also want to increase the training size for each intent with the same increment even though
    the intent-sets have varying sizes.

    @author: tomas.brich@seznam.cz
"""

from random import seed, shuffle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from intent_reco import DATA_DIR
from intent_reco.embeddings.compressed import CompressedModel


def load_file(path, lower=False):
    """
    Loads the Alquist data csv file.
    :param path: path to the csv file
    :param lower: lower-case the text
    :return: dictionary with intents as keys
    """

    d = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.split('\t')
            intt = tmp[0].strip()
            text = tmp[2].strip()
            if lower:
                text = text.lower()
            if intt not in d:
                d[intt] = []
            if text not in BLACKLIST and text not in d[intt]:
                d[intt].append(text)
    return d


def split_data(d, rate=0.5):
    """
    Split the data into two sets.
    :param d: input data - dictionary with intents as keys
    :param rate: (0-1) - rate in which to split the data
    :return: two sets of data
    """

    out1, out2 = dict(), dict()
    for intt in d:
        x = list(d[intt])
        shuffle(x)
        size = int(round(rate*len(x)))
        if size == 0 or size == len(x):
            continue
        out1[intt] = x[:size]
        out2[intt] = x[size:]
    return out1, out2


def split_data_absolute(d, size=50, skip_size=None):
    """
    Split the data into two sets where one set is given by its number of samples.
    :param d: input data - dictionary with intents as keys
    :param size: number of samples in the first set
    :param skip_size: skip intents with datasets smaller than <skip_size>
    :return: two sets of data (as dicts)
    """

    out1, out2 = dict(), dict()
    for intt in d:
        if (skip_size is not None and len(d[intt]) < skip_size) or len(d[intt]) <= size:
            continue
        x = list(d[intt])
        shuffle(x)
        out1[intt] = x[:size]
        out2[intt] = x[size:]
    return out1, out2


def find_closest_intent(sv, y, best=0):
    """
    Find the most similar vectors (cosine similarity) in <y> to <sv>.
    :param sv: sample vector
    :param y: set of vectors with known intents to compare <sv> with
    :param best: set previous best cosine similarity (used for learning curve)
    :return: closest intent and its best cosine similarity
    """

    sv = sv.reshape(1, -1)
    best_intt = None
    for intt in y:
        m = np.max(cosine_similarity(y[intt], sv))
        if m > best:
            best = m
            best_intt = intt
    return best_intt, best


STARSPACE_PATH = DATA_DIR + 'starspace_C4C_2e_50k.txt'
STARSPACE_CB_PATH = DATA_DIR + 'starspace_C4C_2e_50k_cb.txt'
DATA_PATH = DATA_DIR + 'alquist/dm-data-snapshot-uniq.csv'
# BLACKLIST = ['no', 'yes', 'yeah', 'okay', 'sure', 'right']
BLACKLIST = []

if __name__ == '__main__':
    # Initialize random seed
    seed()

    # Load the Alquist data and an embedding model
    data = load_file(DATA_PATH, lower=True)
    model = CompressedModel(STARSPACE_PATH, STARSPACE_CB_PATH)

    num_epochs = 3  # Epochs of the learning curve computation (for computing average)
    num_iters = 15  # Number of iterations with increasing training size
    increment = 10  # Training size increase in each iteration
    skip = 200      # Skip intents with number of samples less than skip

    # Prepare a correct classification counter
    correct = {intent: [[] for _ in range(num_epochs)]
               for intent, cont in data.items() if len(cont) >= skip}

    # Start an epoch
    epoch = 1
    while epoch < num_epochs+1:
        print(f'=====Epoch {epoch}=====')

        # Split the data
        trn, tst = split_data_absolute(data, increment, skip_size=skip)
        intents = {intent: [None]*len(val) for intent, val in tst.items()}
        scores = {intent: [0]*len(val) for intent, val in tst.items()}

        # Start iterating
        it = 1
        while it < num_iters+1:
            print(f'Iteration {it} - Training size: {it * increment}')

            # Compute training set embeddings
            trnv = {intent: model.transform_sentences(val) for intent, val in trn.items()}

            # Compute testing set embeddings
            for intent, val in tst.items():
                correct[intent][epoch-1].append(0)
                tstv = model.transform_sentences(val)

                # Find best matching intent for each sample in the testing set
                for i in range(len(tstv)):
                    bestint, scores[intent][i] = find_closest_intent(tstv[i], trnv, scores[intent][i])
                    if bestint is not None:
                        intents[intent][i] = bestint
                    if intents[intent][i] == intent:
                        correct[intent][epoch-1][-1] += 1

            # Move samples to the training set
            for intent in trn:
                trn[intent] = tst[intent][:increment]
                tst[intent] = tst[intent][increment:]
                intents[intent] = intents[intent][increment:]
                scores[intent] = scores[intent][increment:]
            it += 1
        epoch += 1

    # ToDo: Change to F1 scores.
    # ToDo: Change to numpy arrays.

    # Compute average correct counts and standard deviations
    avg, std = dict(), dict()
    for intent in correct:
        avg[intent] = [sum(e) for e in zip(*correct[intent])]
        avg[intent] = [e/num_epochs for e in avg[intent]]
        std[intent] = np.std(correct[intent], axis=0)

    # Prepare figure
    plt.figure()
    # plt.suptitle('Learning curve for sent2vec (toronto) intent recognition', fontsize=16, fontweight='bold')
    ax = plt.subplot(111)

    # Plot uncertainties
    avg_rate = dict()
    sizes = list(range(increment, num_iters * increment + 1, increment))
    for intent in sorted(avg):
        data_size = len(data[intent])
        tst_sizes = [data_size - trn_size for trn_size in sizes]
        avg_rate[intent] = [avg[intent][i]/tst_sizes[i] for i in range(num_iters)]

        std_rate = [std[intent][i]/tst_sizes[i] for i in range(num_iters)]
        err_min = [avg_rate[intent][i]-std_rate[i] for i in range(num_iters)]
        err_max = [avg_rate[intent][i]+std_rate[i] for i in range(num_iters)]
        ax.fill_between(sizes, err_min, err_max, alpha=0.2)

    # Plot averages
    for intent in sorted(avg):
        data_size = len(data[intent])
        ax.plot(sizes, avg_rate[intent], label=f'{intent} ({data_size})', lw=2)

    # Plot settings
    ax.grid(True)
    ax.set_xlim(sizes[0], sizes[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('Training set size (samples per intent) [-]', fontsize=14)
    ax.set_ylabel('Accuracy [-]', fontsize=14)
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
    ax.legend(loc='lower right', fontsize=14)
    plt.show()
