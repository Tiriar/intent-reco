# -*- coding: utf-8 -*-
"""
    experiments.compress_model
    ~~~~~~~~~~~~~~~~~~~~~~~~~~

    Notebook for compressing embedding models.

    @author: tomas.brich@seznam.cz
"""

from intent_reco import DATA_DIR
from intent_reco.embeddings.fasttext import get_ft_subwords

MODEL_BIN = DATA_DIR + 'wiki.en.bin'
MODEL_VEC = DATA_DIR + 'wiki.en.vec'

# FastText sub-words only
get_ft_subwords(MODEL_BIN, DATA_DIR + 'subwords.txt', limit=100000)

# nohup python model_compression.py --emb_path data/wiki.en.sw.vec --emb_dim 300 --prune_freq 100000 \
# -qnp --d_sv 5 --d_cb 256 --out_name wiki_de_sw_100k &
