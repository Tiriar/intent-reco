# -*- coding: utf-8 -*-
"""
    experiments.compare_intent_unsupervised
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Script for testing the performance of an intent recognition module trained in an unsupervised manner.

    @author: tomas.brich@seznam.cz
"""

from random import seed, shuffle

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from intent_reco import DATA_DIR
from intent_reco.embeddings.compressed import CompressedModel


def load_by_intent(path, lower=False, unique=False, min_count=None):
    """
    Loads the Alquist data csv file.
    :param path: path to the csv file
    :param lower: lower-case the text
    :param unique: keep only unique entries
    :param min_count: remove intents with less than <min_count> samples
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
            if not unique or text not in d[intt]:
                d[intt].append(text)

    if min_count is not None:
        to_remove = [intt for intt, content in d.items() if len(content) < min_count]
        for intt in to_remove:
            del d[intt]

    return d


DATA_TRN = DATA_DIR + 'alquist/dm-uniq-train-01.tsv'
DATA_VAL = DATA_DIR + 'alquist/dm-uniq-val-01.tsv'
# WORD2VEC_EMBEDDINGS = DATA_DIR + 'GoogleNews-vectors-negative300.bin'
# FASTTEXT_EMBEDDINGS_VEC = DATA_DIR + 'wiki.en.vec'
# SENT2VEC_EMBEDDINGS = DATA_DIR + 'torontobooks_unigrams.bin'
WORD2VEC_EMBEDDINGS = DATA_DIR + 'my_models_unsupervised/w2v_alquist_uniq_01'
FASTTEXT_EMBEDDINGS_VEC = DATA_DIR + 'my_models_unsupervised/ft_alquist_uniq_01.vec'
FASTTEXT_EMBEDDINGS_BIN = DATA_DIR + 'my_models_unsupervised/ft_alquist_uniq_01.bin'
SENT2VEC_EMBEDDINGS_UNI = DATA_DIR + 'my_models_unsupervised/s2v_alquist_uniq_01_unigrams.bin'
SENT2VEC_EMBEDDINGS_BI = DATA_DIR + 'my_models_unsupervised/s2v_alquist_uniq_01_bigrams.bin'
STARSPACE_PATH = DATA_DIR + 'starspace_C4C_2e_50k.txt'
STARSPACE_CB_PATH = DATA_DIR + 'starspace_C4C_2e_50k_cb.txt'

if __name__ == '__main__':
    seed()
    templates = 30

    # Load the training and validation sets
    trn = load_by_intent(DATA_TRN, lower=True, unique=True, min_count=200)
    val = load_by_intent(DATA_VAL)

    # Keep only intents present in trn
    print('Intents in validation set:')
    to_pop = []
    for intent in val:
        to_pop.append(intent) if intent not in trn else print(intent, f'({len(val[intent])})')
    for intent in to_pop:
        del val[intent]

    # Pick random <templates> from trn for intent templates
    for intent in trn:
        x = trn[intent]
        shuffle(x)
        trn[intent] = x[:templates]

    # Change the structure for simplicity
    x_trn, y_trn, x_val, y_val = [], [], [], []
    for intent in trn:
        x_trn += trn[intent]
        y_trn += [intent] * len(trn[intent])
    for intent in val:
        x_val += val[intent]
        y_val += [intent] * len(val[intent])

    label_set = sorted(set(y_trn + y_val))
    labels_index = dict(zip(label_set, range(len(label_set))))
    y = [labels_index[x] for x in y_val]

    # Load the model and transform the sentences
    # model = Sent2Vec()
    # model = WordEmbedding(WORD2VEC_EMBEDDINGS, algorithm='word2vec', gensim_trained=True)
    # model = WordEmbedding(FASTTEXT_EMBEDDINGS_VEC, algorithm='fasttext')
    # model = FastText(FASTTEXT_EMBEDDINGS_BIN)
    model = CompressedModel(STARSPACE_PATH, STARSPACE_CB_PATH)

    vec_trn = model.transform_sentences(x_trn)
    vec_val = model.transform_sentences(x_val)

    # Compute predictions
    y_pred = []
    print('\nValidation samples:', len(x_val))
    for v in tqdm(vec_val, mininterval=1.0):
        sim = cosine_similarity(v.reshape(1, -1), vec_trn)
        pred = y_trn[np.argmax(sim).item()]
        y_pred.append(labels_index[pred])

    # Print results
    print('Accuracy:', accuracy_score(y, y_pred))
    print('F1 score:', f1_score(y, y_pred, average='macro'))
