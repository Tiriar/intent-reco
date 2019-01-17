from gensim.models import Word2Vec, KeyedVectors
from nltk.stem.snowball import SnowballStemmer

from intent_reco.embeddings.base import EmbeddingModelBase


class WordEmbedding(EmbeddingModelBase):
    """
    Word2Vec and FastText embedding algorithms contained in Gensim package.
    :param emb_path: path to the embeddings file
    :param algorithm: word2vec or fasttext (default word2vec)
    :param gensim_trained: the model was trained using Gensim
    :param k: number of vectors to load to vocabulary (works only for FastText)
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, algorithm='word2vec', gensim_trained=False, k=None, verbose=False):
        super().__init__(verbose=verbose)

        assert algorithm in ['word2vec', 'fasttext']
        if algorithm == 'word2vec':
            if gensim_trained:
                self.model = Word2Vec.load(emb_path)
            else:
                self.model = KeyedVectors.load_word2vec_format(emb_path, binary=True, limit=k)
        elif algorithm == 'fasttext':
            self.model = KeyedVectors.load_word2vec_format(emb_path, limit=k)

        self.dim = self.model.vector_size
        self.stemmer = SnowballStemmer('english')
        self.gensim_trained = gensim_trained

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        try:
            if self.gensim_trained:
                return self.model.wv[word]
            else:
                return self.model.word_vec(word)
        except KeyError:
            return self.handle_oov(word)

    def handle_oov(self, word):
        """
        Function for handling out-of-vocabulary words.
        :param word: oov word
        :return: oov word numpy vector
        """
        stemmed = self.stemmer.stem(word)
        if word != stemmed:
            try:
                if self.gensim_trained:
                    return self.model.wv[stemmed]
                else:
                    return self.model.word_vec(stemmed)
            except KeyError:
                if self.verbose:
                    print('-- WARNING:', stemmed, '(stemmed) not in vocabulary!')
        return super().handle_oov(word)


def train_word2vec(path, name='w2v_model', dim=300, epoch=16, hs=0, neg=5, sg=0, threads=3):
    """
    Trains a word2vec model.
    :param path: path to the training file
    :param name: name of the output model file
    :param dim: embedding dimension
    :param epoch: number of epochs (data passes)
    :param hs: 1 --> use hierarchical softmax
               0 --> use negative sampling if <neg> > 0
    :param neg: how many negatives should be sampled
    :param sg: 1 --> Skip-Gram
               0 --> CBOW
    :param threads: number of CPU threads to use
    """
    if hs > 0:
        neg = 0

    sentences = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split()
            sentences.append(line)
    model = Word2Vec(sentences, min_count=1, size=dim, iter=epoch,
                     hs=hs, negative=neg, sg=sg, workers=threads)
    model.save(name)
