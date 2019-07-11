# -*- coding: utf-8 -*-
"""
    experiments.alquist_data_stats
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Module for getting statistics on the Alquist dataset.

    @author: tomas.brich@seznam.cz
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords

from intent_reco import DATA_DIR
from intent_reco.utils.preprocessing import tokenize_sentences


def load_file_raw(path, lower=False):
    """
    Loads the Alquist data csv file.
    :param path: path to the csv file
    :param lower: lower-case the text
    :return: X (sentences) and y (intents)
    """

    x, y = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            tmp = line.split('\t')
            text = tmp[2].strip()
            if lower:
                text = text.lower()
            if text not in BLACKLIST:
                y.append(tmp[0].strip())
                x.append(text)
    return x, y


def count_unique_sentences(snts, ints):
    """
    Counts total number of unique sentences / unique per intent.
    :param snts: list of sentences
    :param ints: list of intents
    :return: dict with counts per intent + total
    """

    count = {'total': len(set(snts))}
    per_int = set(zip(snts, ints))
    for s, i in per_int:
        if i not in count:
            count[i] = 0
        count[i] += 1
    return count


def build_vocabulary(snts, remove=None):
    """
    Build vocabulary with word frequencies from a list of sentences.
    :param snts: list of sentences
    :param remove: list of words to remove
    :return: Counter of words
    """

    w = []
    for s in snts:
        w += s.split()
    count = Counter(x for x in w if x not in set(remove))
    return count


def words_per_intent(snts, ints, remove=None):
    """
    Creates counters of words for each intent.
    :param snts: list of sentences
    :param ints: list of intents
    :param remove: list of words to remove
    :return: dictionary of counters by intent
    """

    words = dict()
    for s, i in zip(snts, ints):
        if i not in words:
            words[i] = []
        words[i] += s.split()

    remove = set(remove)
    counts = {i: Counter(x for x in words[i] if x not in remove) for i in words}
    return counts


def plot_bar(n1, n2, xlabels, title, legend):
    """
    Plots a bar graph (without showing) for two counts.
    :param n1: list of first counts
    :param n2: list of second counts
    :param xlabels: X axis labels
    :param title: title of the graph
    :param legend: legend entries
    """

    n = len(n1)
    assert n == len(n2)

    ind = np.arange(n)
    width = 0.4
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, n1, width, color='r')
    rects2 = ax.bar(ind+width, n2, width, color='y')

    ax.set_title(title)
    ax.set_ylabel('Count')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xlabels, rotation='vertical')
    ax.legend((rects1[0], rects2[0]), legend)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2 - 0.1, height, '%d' % int(height),
                    ha='left', va='bottom', rotation=70)


DATA_PATH = DATA_DIR + 'alquist/dm-data-snapshot-uniq.csv'
STOPWORDS = stopwords.words('english')
PUNCTUATION = ['.', '?', '!']
# BLACKLIST = ['no', 'yes', 'yeah', 'okay', 'sure', 'right']
BLACKLIST = []

if __name__ == '__main__':
    sents, intents = load_file_raw(DATA_PATH)
    sents = tokenize_sentences(sents)
    int_uniq = sorted(set(intents))

    print('\n===SENTENCES (full / unique)===')
    cnt = Counter(intents)
    cnt_uniq = count_unique_sentences(sents, intents)
    print('Total:', len(sents), '/', cnt_uniq['total'])
    c1, c2 = [], []
    for intent in int_uniq:
        c1.append(cnt[intent])
        c2.append(cnt_uniq[intent])
        print(intent+':', c1[-1], '/', c2[-1])
    plot_bar(c1, c2, int_uniq, f"Sentence counts (total: {len(sents)} / {cnt_uniq['total']})", ('full', 'unique'))

    print('\n===VOCABULARY (full / without stopwords)===')
    voc = build_vocabulary(sents, remove=PUNCTUATION)
    voc_stop = build_vocabulary(sents, remove=STOPWORDS+PUNCTUATION)
    print('Words in corpus:', sum(voc.values()), '/', sum(voc_stop.values()))
    print('Vocabulary:', len(voc), '/', len(voc_stop))
    print('10 most common:')
    print(voc.most_common(10))
    print(voc_stop.most_common(10))

    print('\n===VOCABULARY PER INTENT (full / without stopwords)===')
    cnt = words_per_intent(sents, intents, remove=PUNCTUATION)
    cnt_stop = words_per_intent(sents, intents, remove=STOPWORDS+PUNCTUATION)
    c1, c2 = [], []
    for intent in int_uniq:
        c1.append(len(cnt[intent]))
        c2.append(len(cnt_stop[intent]))
        print(intent+':', c1[-1], '/', c2[-1])
        print(cnt[intent].most_common(5))
        print(cnt_stop[intent].most_common(5))
    plot_bar(c1, c2, int_uniq, f'Intent vocabularies (total: {len(voc)} / {len(voc_stop)})',
             ('full', 'without stop-words'))

    plt.show()
