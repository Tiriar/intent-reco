# -*- coding: utf-8 -*-
"""
    experiments.alquist_convnet
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Module for training convolutional network based on a script by Jan Pichl (Alquist chatbot team).

    @author: tomas.brich@seznam.cz
"""

import csv
import json

import numpy as np
from collections import Counter
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv1D, Dense, Dropout, Embedding, Flatten, Input, MaxPooling1D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from intent_reco import DATA_DIR

# Constants
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 40
EMB_DIM = 300

# Files
GLOVE = DATA_DIR + 'glove.6B.300d.txt'
DATA_TRAIN = DATA_DIR + 'alquist/dm-uniq-train-01.tsv'
DATA_VAL = DATA_DIR + 'alquist/dm-uniq-val-01.tsv'
VOCAB_PATH = DATA_DIR + 'my_models/convnet_alquist_vocab.json'
TOKENIZER = Tokenizer(num_words=MAX_NB_WORDS)


def fit_transform(txts, padding):
    TOKENIZER.fit_on_texts(txts)
    sequences = TOKENIZER.texts_to_sequences(txts)

    wrd_index = TOKENIZER.word_index
    print(f'Found {len(wrd_index)} unique tokens.')

    d = pad_sequences(sequences, maxlen=padding)
    return d, wrd_index


def transform(txts, padding):
    for n, t in enumerate(txts):
        if t is None:
            print(n, txts[n - 2])
    sequences = TOKENIZER.texts_to_sequences(txts)

    d = pad_sequences(sequences, maxlen=padding)
    return d


if __name__ == '__main__':
    # Load data
    with open(DATA_TRAIN) as f:
        reader = csv.reader(f, delimiter='\t')
        samples_trn = [{'label': row[0], 'prev': row[1], 'message': row[2], 'focus': row[3]} for row in reader]

    with open(DATA_VAL) as f:
        reader = csv.reader(f, delimiter='\t')
        samples_val = [{'label': row[0], 'prev': row[1], 'message': row[2], 'focus': row[3]} for row in reader]

    samples = samples_trn + samples_val

    # Load the vocab json if it was already created
    # with open(VOCAB_PATH) as f:
    #     vocab = json.load(f)

    full_texts = [d['message'] for d in samples]
    texts = [d['message'] for d in samples]

    print(Counter([d['label'] for d in samples]))
    label_set = sorted(set([d['label'] for d in samples]))
    print(label_set)

    labels_index = dict(zip(label_set, range(len(label_set))))  # dictionary mapping label name to numeric id
    # labels_index = vocab['labels']
    print(labels_index)
    labels = [labels_index[d['label']] for d in samples]  # list of label ids

    print(f'Found {len(texts)} texts.')
    _, word_index = fit_transform(full_texts, MAX_SEQUENCE_LENGTH)
    # word_index = vocab['vocab']
    TOKENIZER.word_index = word_index
    data = transform(texts, MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels), len(labels_index))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # Split the data into training and validation sets
    trn_size = len(samples_trn)
    x_train = data[:trn_size]
    y_train = labels[:trn_size]
    x_val = data[trn_size:]
    y_val = labels[trn_size:]

    # Load GloVe embeddings
    embeddings_index, word = dict(), None
    with open(GLOVE) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Found {len(embeddings_index)} word vectors.')

    embedding_matrix = np.zeros((len(word_index) + 1, EMB_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1, EMB_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    print("\nUtterance branch")
    print(embedded_sequences)
    convolutions = []
    conv_filters = [1, 2, 3, 5]
    for cf in conv_filters:
        x = Conv1D(150, cf, activation='tanh')(embedded_sequences)
        print(x)
        x = MaxPooling1D(int(x.shape[1] - cf + 1))(x)
        print(x)
        x = Flatten()(x)
        print(x)
        convolutions.append(x)

    print("\nMerge")
    x = Concatenate()(convolutions)
    print(x)
    if len(conv_filters) > 1:
        x = Dense(int(EMB_DIM / 4 * len(conv_filters)), activation='tanh')(x)
        print(x)
    x = Dropout(0.5)(x)
    print(x)

    preds = Dense(len(labels_index), activation='softmax')(x)
    print(preds)

    with open(VOCAB_PATH, 'w') as f:
        json.dump({'labels': labels_index, 'vocab': word_index}, f)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    print(x_train.shape, y_train.shape, '\n')
    tensorboard = TensorBoard(log_dir=DATA_DIR + 'my_models/convnet_tensorboard/',
                              histogram_freq=0, write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint(DATA_DIR + 'my_models/convnet_model.h5', monitor='val_loss',
                                 verbose=0, save_best_only=True, save_weights_only=False,
                                 mode='auto', period=1)
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=16, batch_size=128,
              verbose=2, callbacks=[tensorboard, checkpoint])
