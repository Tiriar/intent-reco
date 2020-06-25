# -*- coding: utf-8 -*-
"""
    experiments.compress_model
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Notebook for compressing embedding models.
    Especially useful when compressing FastText models with sub-word embeddings.

    @author: tomas.brich@seznam.cz
"""

from intent_reco import DATA_DIR
from intent_reco.embeddings.fasttext import get_ft_subwords
from intent_reco.model_compression import compress

CORPUS = 'wiki'
LANG = 'en'

# base model
MODEL_PATH = DATA_DIR + f'{CORPUS}.{LANG}.vec'
OUT_NAME = DATA_DIR + f'{CORPUS}.{LANG}'

compress(
    emb_path=MODEL_PATH,
    emb_dim=300,
    prune_freq=50000,
    quantize=True,
    normalize=True,
    d_sv=5,
    d_cb=256,
    out_name=OUT_NAME,
    pickle_output=True
)

# sub-word model
MODEL_PATH = DATA_DIR + f'{CORPUS}.{LANG}.bin'
MODEL_SW_PATH = DATA_DIR + f'{CORPUS}.{LANG}.sw.vec'
OUT_NAME = DATA_DIR + f'{CORPUS}.{LANG}.sw'

get_ft_subwords(MODEL_PATH, MODEL_SW_PATH, limit=100000)
compress(
    emb_path=MODEL_SW_PATH,
    emb_dim=300,
    prune_freq=100000,
    quantize=True,
    normalize=True,
    d_sv=5,
    d_cb=256,
    out_name=OUT_NAME,
    pickle_output=True
)
