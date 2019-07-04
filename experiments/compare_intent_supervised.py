"""Script for testing the performance of an intent recognition module trained in a supervised manner."""

import numpy as np
import json
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, f1_score

from experiments.alquist_convnet import MAX_SEQUENCE_LENGTH, MAX_NB_WORDS
from intent_reco.utils.data import load_alquist
from intent_reco.embeddings.starspace import StarSpace
from intent_reco.embeddings.fasttext import FastText

MODE = 'convnet'
DATA_VAL = '../data/alquist/dm-uniq-val-01.tsv'
VOCAB_PATH = '../data/my_models/convnet_alquist_vocab.json'
STARSPACE_MODEL_PATH = '../data/my_models/starspace_alquist_uniq_01.tsv'
CONVNET_MODEL_PATH = '../data/my_models/convnet_alquist_uniq_01.h5'
FASTTEXT_MODEL_PATH = '../data/my_models/ft_alquist_uniq_01.bin'

if __name__ == '__main__':
    # Load the ConvNet dictionary to get the word and intents indices
    with open(VOCAB_PATH) as f:
        vocab = json.load(f)
    labels_index = vocab['labels']
    vocab = vocab['vocab']

    # Load the validation data
    data = load_alquist(DATA_VAL)
    sentences = data['X']
    intents = data['y']
    y = [labels_index[x] for x in intents]

    # Compute predictions
    if MODE == 'starspace':
        model = StarSpace(STARSPACE_MODEL_PATH)
        pred = model.classify_sentences(sentences)
        y_pred = [labels_index[x] for x in pred]
    elif MODE == 'fasttext':
        model = FastText(FASTTEXT_MODEL_PATH)
        pred = model.classify_sentences(sentences)
        y_pred = [labels_index[x] for x in pred]
    else:
        model = load_model(CONVNET_MODEL_PATH)
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
        tokenizer.word_index = vocab
        sequences = tokenizer.texts_to_sequences(sentences)
        sentences = pad_sequences(sequences, MAX_SEQUENCE_LENGTH)
        pred = model.predict(sentences)
        y_pred = np.argmax(pred, axis=1)

    # Print results
    print('Accuracy:', accuracy_score(y, y_pred))
    print('F1 score:', f1_score(y, y_pred, average='macro'))
