# -*- coding: utf-8 -*-
"""
    experiments.vocabulary_check
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Check embedding model vocabulary.

    @author: tomas.brich@seznam.cz
"""

from string import ascii_letters

from intent_reco import DATA_DIR
from intent_reco.utils.data import load_model_ft_bin, load_model_txt

# load the model
PATH = DATA_DIR + 'wiki.en.vec'
if PATH.endswith('.bin'):
    vocab = load_model_ft_bin(PATH, k=500000)[0]
else:
    vocab = load_model_txt(PATH, k=500000, dim=300, header=True)[0]

# compute number of tokens with unsuitable chars
STEP = 10000
ALLOWED_CHARS = set(ascii_letters + " '-")

count = 0
bl_chars, bl_tokens = [], {}
for idx, token in enumerate(vocab, 1):
    for ch in token:
        if ch not in ALLOWED_CHARS:
            if ch not in bl_tokens:
                bl_chars.append(ch)
                bl_tokens[ch] = []
            bl_tokens[ch].append((idx, token))
            count += 1
            break

    if not idx % STEP:
        print(f'{idx - STEP + 1}-{idx} --> {count}')
        count = 0

# show blacklisted chars and tokens
NUM_BL = 100
print(f'\nFirst {NUM_BL} blacklisted chars and tokens:')
for ch in bl_chars[:NUM_BL]:
    tokens = [f'{token} ({idx})' for idx, token in bl_tokens[ch][:10]]
    print(f'CHAR: {ch} TOKENS: {" | ".join(tokens)}')
