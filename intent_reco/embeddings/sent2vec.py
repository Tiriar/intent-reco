import sent2vec

from intent_reco.utils.preprocessing import tokenize_sentences


class Sent2Vec:
    """
    Sent2Vec embedding algorithm.
    :param emb_path: path to the embeddings file
    """
    def __init__(self, emb_path):
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(emb_path)

    def transform_sentences(self, sentences):
        """
        Transform a list of sentences into vector representation.
        :param sentences: list of sentences
        :return: list of numpy vectors
        """
        return [self.model.embed_sentence(s) for s in tokenize_sentences(sentences)]
