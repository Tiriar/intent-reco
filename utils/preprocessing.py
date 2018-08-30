"""Utilities for sentence pre-processing during encoding."""

import re
from nltk import TweetTokenizer

TOKENIZER = TweetTokenizer()


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


def tokenize(sentence, tknzr=TOKENIZER, to_lower=True):
    """
    Tokenize sentence.
    :param sentence: a string to be tokenized
    :param tknzr: a tokenizer implementing the NLTK tokenizer interface
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


def tokenize_sentences(sentences, tknzr=TOKENIZER, to_lower=True):
    """
    Tokenize sentences.
    :param sentences: a list of sentences
    :param tknzr: a tokenizer implementing the NLTK tokenizer interface
    :param to_lower: lowercase or not
    :return: tokenized sentences
    """
    return [tokenize(s, tknzr=tknzr, to_lower=to_lower) for s in sentences]
