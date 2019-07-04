"""Script for testing the performance of an intent recognition module trained in an unsupervised manner."""

import numpy as np
from random import seed, shuffle
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

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
    d = {}
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
        to_remove = []
        for intt in d:
            if len(d[intt]) < min_count:
                to_remove.append(intt)
        for intt in to_remove:
            del d[intt]

    return d


DATA_TRN = '../data/alquist/dm-uniq-train-01.tsv'
DATA_VAL = '../data/alquist/dm-uniq-val-01.tsv'
# FASTTEXT_EMBEDDINGS_VEC = 'data/wiki.en.vec'
# WORD2VEC_EMBEDDINGS = 'data/GoogleNews-vectors-negative300.bin'
# SENT2VEC_EMBEDDINGS = 'data/torontobooks_unigrams.bin'
WORD2VEC_EMBEDDINGS = '../data/my_models_unsupervised/w2v_alquist_uniq_01'
FASTTEXT_EMBEDDINGS_VEC = '../data/my_models_unsupervised/ft_alquist_uniq_01.vec'
FASTTEXT_EMBEDDINGS_BIN = '../data/my_models_unsupervised/ft_alquist_uniq_01.bin'
SENT2VEC_EMBEDDINGS_UNI = '../data/my_models_unsupervised/s2v_alquist_uniq_01_unigrams.bin'
SENT2VEC_EMBEDDINGS_BI = '../data/my_models_unsupervised/s2v_alquist_uniq_01_bigrams.bin'
STARSPACE_PATH = '../data/starspace_C4C_2e_50k.txt'
STARSPACE_CB_PATH = '../data/starspace_C4C_2e_50k_cb.txt'

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
        if intent not in trn:
            to_pop.append(intent)
        else:
            print(intent, '({})'.format(len(val[intent])))
    for intent in to_pop:
        del val[intent]

    # Pick random <templates> from trn for intent templates
    for intent in trn:
        x = trn[intent]
        shuffle(x)
        trn[intent] = x[:templates]

    # Change the structure for simplicity
    x_trn = []
    y_trn = []
    for intent in trn:
        x_trn += trn[intent]
        y_trn += [intent] * len(trn[intent])
    x_val = []
    y_val = []
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
    for i, v in enumerate(vec_val):
        sim = cosine_similarity(v.reshape(1, -1), vec_trn)
        pred = y_trn[np.asscalar(np.argmax(sim))]
        y_pred.append(labels_index[pred])
        if not i % 100 and i > 0:
            print(i, 'validation samples evaluated.')

    # Print results
    print('Accuracy:', accuracy_score(y, y_pred))
    print('F1 score:', f1_score(y, y_pred, average='macro'))
