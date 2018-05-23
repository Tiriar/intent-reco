"""Script for testing the performance of an intent recognition module on individual queries."""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from model_compression import chunks, convert_vec
from embedding_wrappers import CompressedModel
from lbg import generate_codebook

STARSPACE_PATH = 'data/starspace_C4C_2e_50k.txt'
STARSPACE_CB_PATH = 'data/starspace_C4C_2e_50k_cb.txt'
TEMPLATES = 'data/templates.json'


def find_intent(t, mat):
    """
    Print the closest intent based on templates <t> and similarity matrix <mat>.
    :param t: quantized template vector set
    :param mat: matrix of the precomputed cosine similarities
    """
    best = 0
    bestint = None
    bestsent = None
    for intent in t:
        for s_idx, s in enumerate(t[intent]):
            sim = 0
            for pos, num in enumerate(s):
                sim += mat[num][pos]

            if sim > best:
                best = sim
                bestint = intent
                bestsent = templates[intent][s_idx]
    print('Intent:', bestint, '\nCosine similarity:', best, '\nClosest template:', bestsent)


def normalize(t):
    """
    Normalize the template vectors in <t> and return their original norms.
    :param t: template set
    :return: template norms
    """
    vsize = {}
    for intent in t:
        vsize[intent] = []
        temp = []
        for v in t[intent]:
            n = np.linalg.norm(v)
            temp.append((v/n).tolist())
            vsize[intent].append(n)
        t[intent] = temp
    return vsize


def split_vecs(d, n=4):
    """
    Splits vectors in <d> into sub-vectors of size <n> and returns them in a list.
    :param d: input dictionary
    :param n: size of sub-vectors
    :return: list of sub-vectors
    """
    vecs = []
    for v in d:
        vecs += list(chunks(v, n))
    return vecs


print('Loading templates...')
with open(TEMPLATES) as f:
    templates = json.load(f)

# change the template structure for simplicity
for key in templates:
    tmp = []
    for sent in templates[key]:
        tmp.append(sent['text'])
        templates[key] = tmp

print('Loading the embedding model...')
model = CompressedModel(STARSPACE_PATH, STARSPACE_CB_PATH, dim=30, normalized=True)

print('Converting the templates...')
templates_vec = {}
for key in templates:
    templates_vec[key] = model.transform(templates[key])

print('Quantizing the templates...')
D_SV = 4
D_CB = 8

sizes = normalize(templates_vec)
lbg_data = []
for key in templates_vec:
    lbg_data += templates_vec[key]
dim = len(lbg_data[0])
lbg_data = split_vecs(lbg_data, n=D_SV)

codebook = generate_codebook(lbg_data, cb_size=D_CB)[0]
for key in templates_vec:
    for i in range(len(templates_vec[key])):
        templates_vec[key][i] = convert_vec(templates_vec[key][i], D_SV, codebook)

print('\n===READY===')
inp = None
while inp not in ['exit', 'stop']:
    if inp is not None:
        # pre-compute the similarity matrix
        vec = model.transform([inp])
        svs = split_vecs(vec, n=D_SV)
        matrix = cosine_similarity(codebook, svs)

        # find the intent
        find_intent(templates_vec, matrix)
    inp = input('Write query: ')
