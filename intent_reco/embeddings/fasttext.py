import fastText

from intent_reco.embeddings.base import EmbeddingModelBase
from intent_reco.utils.preprocessing import tokenize_sentences


class FastText(EmbeddingModelBase):
    """
    FastText embedding algorithm.
    :param emb_path: path to the binary embeddings file (model.bin)
    :param verbose: verbosity (mostly OOV warnings)
    """
    def __init__(self, emb_path, verbose=False):
        super().__init__(verbose=verbose)
        self.model = fastText.load_model(emb_path)
        self.dim = self.model.get_dimension()

    def transform_word(self, word):
        """
        Transform the <word> into vector representation.
        :param word: input word
        :return: word numpy vector
        """
        return self.model.get_word_vector(word)

    # def transform_sentence(self, sentence):
    #     """
    #     Transform the <sentence> into vector representation.
    #     :param sentence: input sentence
    #     :return: sentence numpy vector
    #     """
    #     return self.model.get_sentence_vector(sentence)

    def classify_sentences(self, sentences):
        """
        Classify sentences into classes trained in the FastText model.
        :param sentences: list of sentences to classify
        :return: list of classes
        """
        labels = self.model.predict(tokenize_sentences(sentences))[0]
        return [w[0].replace('__label__', '') for w in labels]


def get_ft_subwords(path):
    """
    Brute-forces the subword embeddings from a FastText binary model
    and writes them to 'subwords.txt' in a word2vec format.
    The subwords are sorted by (length, frequency).
    :param path: path to the binary file
    """
    from tqdm import tqdm

    ft = fastText.load_model(path)
    vocab = ft.get_words()
    vocab_len = len(vocab)
    subwords = {}
    sw_ids = {}

    print('Processing', vocab_len, 'words...')
    for word in tqdm(vocab, mininterval=1.0):
        sw, ids = ft.get_subwords(word)
        ids = ids.tolist()
        if ids[0] < vocab_len:
            del sw[0]
            del ids[0]

        for subword, idx in zip(sw, ids):
            if subword not in subwords:
                subwords[subword] = 1
                sw_ids[subword] = idx
            else:
                subwords[subword] += 1

    print('Computing output...')
    sw_sorted = sorted(subwords, key=lambda x: (len(x), -subwords[x]))
    out = [str(len(sw_sorted)) + ' ' + str(ft.get_dimension()) + '\n']
    for sw in tqdm(sw_sorted, mininterval=1.0):
        tmp = sw
        vec = ft.get_input_vector(sw_ids[sw])
        for num in vec:
            tmp += ' ' + str(num)
        out.append(tmp + '\n')

    print('Writing output...')
    with open('subwords.txt', 'w') as f:
        f.writelines(out)
