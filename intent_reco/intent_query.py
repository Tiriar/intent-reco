"""Script for testing the performance of an intent recognition module on individual queries."""

import json
import numpy as np
from numpy.linalg import norm

from intent_reco.model_compression import split_vecs, convert_vec
from intent_reco.embeddings.compressed import CompressedModel
from intent_reco.utils.lbg import generate_codebook

EMBEDDING_PATH = '../data/starspace_C4C_2e_50k.txt'
EMBEDDING_CB_PATH = '../data/starspace_C4C_2e_50k_cb.txt'
TEMPLATES = '../data/templates.json'


def find_intent(t, tn, mat):
    """
    Print the closest intent based on templates <t> and similarity matrix <mat>.
    :param t: quantized template vector set
    :param tn: template norms
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
            sim /= tn[intent][s_idx]

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
model = CompressedModel(EMBEDDING_PATH, EMBEDDING_CB_PATH)

print('Converting the templates...')
templates_vec = {}
for key in templates:
    templates_vec[key] = model.transform_sentences(templates[key])

print('Quantizing the templates...')
D_SV = 2
D_CB = 16

# prepare the data for LBG
sizes = normalize(templates_vec)
lbg_data = []
for key in templates_vec:
    lbg_data += templates_vec[key]
dim = len(lbg_data[0])
lbg_data = split_vecs(lbg_data, n=D_SV)

# compute the quantization codebook
codebook = generate_codebook(lbg_data, cb_size=D_CB)[0]
for key in templates_vec:
    for i in range(len(templates_vec[key])):
        templates_vec[key][i] = convert_vec(templates_vec[key][i], D_SV, codebook)

# compute the norms of the quantized templates
template_norms = {}
for key in templates_vec:
    template_norms[key] = []
    for sent in templates_vec[key]:
        sq_sum = 0
        for idx in sent:
            sq_sum += sum(codebook[idx]**2)
        template_norms[key].append(sq_sum**(1/2))

print('\n===READY===')
inp = None
while inp not in ['exit', 'stop']:
    if inp is not None:
        # pre-compute the similarity matrix
        vec = model.transform_sentence(inp)
        svs = np.array(split_vecs([vec], n=D_SV))
        matrix = np.dot(codebook, svs.transpose()) / norm(vec)

        # find the intent
        find_intent(templates_vec, template_norms, matrix)
    inp = input('Write query: ')
