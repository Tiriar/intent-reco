"""Utilities needed to run the sent2vec embedding model."""

import os
import time
import re
import numpy as np
from subprocess import call
from nltk import TweetTokenizer
from nltk.tokenize.stanford import StanfordTokenizer

FASTTEXT_EXEC_PATH = os.path.abspath("../sent2vec/fasttext")

BASE_SNLP_PATH = os.path.abspath("./data")
SNLP_TAGGER_JAR = os.path.join(BASE_SNLP_PATH, "stanford-postagger.jar")

MODEL_WIKI_UNIGRAMS = os.path.abspath("./data/wiki_unigrams.bin")
MODEL_WIKI_BIGRAMS = os.path.abspath("./data/wiki_bigrams.bin")
MODEL_TORONTOBOOKS_UNIGRAMS = os.path.abspath("./data/torontobooks_unigrams.bin")
MODEL_TORONTOBOOKS_BIGRAMS = os.path.abspath("./data/torontobooks_bigrams.bin")
MODEL_TWITTER_UNIGRAMS = os.path.abspath('./data/twitter_unigrams.bin')
MODEL_TWITTER_BIGRAMS = os.path.abspath('./data/twitter_bigrams.bin')


def tokenize(tknzr, sentence, to_lower=True):
    """
    Tokenize sentence.
    :param tknzr: a tokenizer implementing the NLTK tokenizer interface
    :param sentence: a string to be tokenized
    :param to_lower: lowercase or not
    :return: tokenized sentence
    """
    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '<url>', sentence)
    sentence = re.sub('(\@[^\s]+)', '<user>', sentence)
    filter(lambda word: ' ' not in word, sentence)
    return sentence


def format_token(token):
    """
    Format tokens.
    :param token: string token
    :return: formatted token
    """
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token


def tokenize_sentences(tknzr, sentences, to_lower=True):
    """
    Tokenize sentences.
    :param tknzr: a tokenizer implementing the NLTK tokenizer interface
    :param sentences: a list of sentences
    :param to_lower: lowercase or not
    :return: tokenized sentences
    """
    return [tokenize(tknzr, s, to_lower) for s in sentences]


def get_embeddings_for_preprocessed_sentences(sentences, model_path, fasttext_exec_path):
    """
    Get embeddings for preprocessed sentences.
    :param sentences: a list of preprocessed sentences
    :param model_path: a path to the sent2vec .bin model
    :param fasttext_exec_path: a path to the fasttext executable
    :return: vectorized sentences
    """
    timestamp = str(time.time())
    test_path = os.path.abspath('./'+timestamp+'_fasttext.test.txt')
    embeddings_path = os.path.abspath('./'+timestamp+'_fasttext.embeddings.txt')
    dump_text_to_disk(test_path, sentences)
    call(fasttext_exec_path +
         ' print-sentence-vectors ' +
         model_path + ' < ' +
         test_path + ' > ' +
         embeddings_path, shell=True)
    embeddings = read_embeddings(embeddings_path)
    os.remove(test_path)
    os.remove(embeddings_path)
    assert(len(sentences) == len(embeddings))
    return np.array(embeddings)


def read_embeddings(embeddings_path):
    """
    Read embeddings.
    :param embeddings_path: path to the embeddings
    :return: embeddings
    """
    with open(embeddings_path, 'r', encoding='utf-8') as in_stream:
        embeddings = []
        for line in in_stream:
            line = '['+line.replace(' ', ',')+']'
            embeddings.append(eval(line))
        return embeddings


def dump_text_to_disk(file_path, x, y=None):
    """
    Dump text to disk.
    :param file_path: where to dump the data
    :param x: list of sentences to dump
    :param y: labels, if any
    """
    with open(file_path, 'w', encoding='utf-8') as out_stream:
        if y is not None:
            for xi, yi in zip(x, y):
                out_stream.write('__label__'+str(yi)+' '+xi+' \n')
        else:
            for xi in x:
                out_stream.write(xi+' \n')


def get_sentence_embeddings(sentences, ngram='bigrams', model='twitter'):
    """
    Returns a numpy matrix of embeddings for one of the published models.
    It handles tokenization and can be given raw sentences.
    :param sentences: a list of raw sentences
    :param ngram: n-gram in ['unigrams', 'bigrams']
    :param model: model name in ['wiki', 'twitter', 'toronto']
    :return: vectorized sentences
    """
    model_path = get_model_path(ngram, model)

    if model in ['wiki', 'toronto']:
        tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
        s = ' <delimiter> '.join(sentences)
        tokenized_sentences = tokenize_sentences(tknzr, [s])
        tokenized_sentences = tokenized_sentences[0].split(' <delimiter> ')
        assert(len(tokenized_sentences) == len(sentences))
    else:
        tknzr = TweetTokenizer()
        tokenized_sentences = tokenize_sentences(tknzr, sentences)

    return get_embeddings_for_preprocessed_sentences(tokenized_sentences, model_path, FASTTEXT_EXEC_PATH)


def get_model_path(ngram, model):
    """
    Returns a path to the model file given n-gram and the model name.
    :param ngram: n-gram in ['unigrams', 'bigrams']
    :param model: model name in ['wiki', 'twitter', 'toronto']
    :return: model path
    """
    if model == 'wiki':
        return MODEL_WIKI_UNIGRAMS if ngram == 'unigrams' else MODEL_WIKI_BIGRAMS
    elif model == 'twitter':
        return MODEL_TWITTER_UNIGRAMS if ngram == 'unigrams' else MODEL_TWITTER_BIGRAMS
    else:
        return MODEL_TORONTOBOOKS_UNIGRAMS if ngram == 'unigrams' else MODEL_TORONTOBOOKS_BIGRAMS


def preprocess_sentences(sentences, use_pos_tagger=True, to_lower=True):
    """
    Prepares the sentences for sent2vec transformation.
    :param sentences: sentences to be processed
    :param use_pos_tagger: use the Stanford POS tagger for tokenizing
    :param to_lower: lowercase the sentences
    :return: processed sentences
    """
    if use_pos_tagger:
        tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
        s = ' <delimiter> '.join(sentences)
        tokenized_sentences = tokenize_sentences(tknzr, [s], to_lower)
        tokenized_sentences = tokenized_sentences[0].split(' <delimiter> ')
        assert(len(tokenized_sentences) == len(sentences))
    else:
        tknzr = TweetTokenizer()
        tokenized_sentences = tokenize_sentences(tknzr, sentences, to_lower)
    return tokenized_sentences


if __name__ == '__main__':
    sents = ['Once upon a time.', 'And now for something completely different.']

    my_embeddings = get_sentence_embeddings(sents, ngram='unigrams', model='wiki')
    print(my_embeddings.shape)
