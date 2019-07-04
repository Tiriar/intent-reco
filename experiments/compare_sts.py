"""Module for comparing embedding algorithms using the STS Benchmark dataset."""

from scipy.stats import pearsonr, spearmanr
from timeit import default_timer as timer

from intent_reco.utils.utils import compute_score_pairs
from intent_reco.utils.data import load_sts
from intent_reco.utils.plotting import score_hist, score_scatter

# ===== Data settings =====
STS_SETS = ['test']
# STS_SETS = ['train', 'test']
# STS_SETS = ['train', 'dev', 'test']
STS_PATH = '../data/stsbenchmark/'
# STS_PATH = '../data/stsbenchmark/InferSent_preprocessed/'

# ===== Model settings =====
INFERSENT_PICKLE = '../data/infersent.allnli.pickle'
INFERSENT_EMBEDDINGS = '../data/glove.840B.300d.txt'
SENT2VEC_EMBEDDINGS = '../data/torontobooks_unigrams.bin'
GLOVE_EMBEDDINGS = '../data/glove.6B.300d.txt'
WORD2VEC_EMBEDDINGS = '../data/GoogleNews-vectors-negative300.bin'
FASTTEXT_EMBEDDINGS_BIN = '../data/wiki.en.bin'
FASTTEXT_EMBEDDINGS_VEC = '../data/wiki.en.vec'

# ===== My models =====
STARSPACE_SIMILARITY = '../data/my_models/starspace_sts_sentence_similarity.tsv'
STARSPACE_EMBEDDINGS = '../data/my_models_unsupervised/starspace_sts_32e.tsv'
SENT2VEC_STS_EMBEDDINGS = '../data/my_models_unsupervised/s2v_sts_unigrams_4096e.bin'
WORD2VEC_STS_EMBEDDINGS = '../data/my_models_unsupervised/w2v_sg_hs_sts_2048e'
FT_STS_BIN = '../data/my_models_unsupervised/ft_sts_1024e.bin'
FT_STS_VEC = '../data/my_models_unsupervised/ft_sts_1024e.vec'

# ===== Compressed model settings =====
EMB_COMPRESSED = '../data/my_models_compressed/twitter_200k_10sv_128cb_norm.txt'
EMB_COMPRESSED_CB = '../data/my_models_compressed/twitter_200k_10sv_128cb_norm_cb.txt'
EMB_PRUNED = '../data/my_models_compressed/model_pruned_3413.txt'

PLOT_RESULTS = False

if __name__ == '__main__':
    print('Loading the STS Benchmark dataset...')
    timer_start = timer()
    data = {}
    for sts in STS_SETS:
        data[sts] = load_sts(STS_PATH + 'sts-{}.csv'.format(sts), lower=True)
    print('Time elapsed:', timer()-timer_start, 's\n')

    print('Setting up the embedding model...')
    timer_start = timer()

    # from embeddings.infersent import InferSent
    # model = InferSent(INFERSENT_EMBEDDINGS, INFERSENT_PICKLE, vocab_data=data['test']['X1']+data['test']['X2'])

    # from embeddings.sent2vec import Sent2Vec
    # model = Sent2Vec(SENT2VEC_EMBEDDINGS)

    # from embeddings.glove import GloVe, GloVeSpacy
    # model = GloVe(GLOVE_EMBEDDINGS)
    # model = GloVeSpacy(GLOVE_EMBEDDINGS)

    # from embeddings.word2vec import WordEmbedding
    # model = WordEmbedding(WORD2VEC_EMBEDDINGS, algorithm='word2vec')
    # model = WordEmbedding(FASTTEXT_EMBEDDINGS_VEC, algorithm='fasttext', k=200000)

    # from embeddings.fasttext import FastText
    # model = FastText(FASTTEXT_EMBEDDINGS_BIN)

    # from embeddings.starspace import StarSpace
    # model = StarSpace(STARSPACE_EMBEDDINGS)

    from embeddings.compressed import CompressedModel
    model = CompressedModel(EMB_COMPRESSED, EMB_COMPRESSED_CB)

    print('Time elapsed:', timer()-timer_start, 's\n')

    print('Transforming the testing set...')
    timer_start = timer()
    v1 = model.transform_sentences(data['test']['X1'])
    v2 = model.transform_sentences(data['test']['X2'])
    print('Time elapsed:', timer()-timer_start, 's\n')

    print('Results:')
    score = compute_score_pairs(v1, v2)
    print('Pearson coef.:', pearsonr(score, data['test']['y'])[0])
    print('Spearman coef.:', spearmanr(score, data['test']['y'])[0])

    if PLOT_RESULTS:
        score_hist(score)
        score_scatter(data['test']['y'], score)
