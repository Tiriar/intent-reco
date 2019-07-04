"""
Module for evaluating the InferSent embedding model on the STS Benchmark dataset (regression trained on STS train).

The code was mostly dug out from the SentEval project: https://github.com/facebookresearch/SentEval
"""

import io
import copy
import numpy as np

import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from timeit import default_timer as timer

GLOVE_PATH = '../data/glove.840B.300d.txt'
MODEL_PATH = '../data/infersent.allnli.pickle'
DATA_PATH = '../data/stsbenchmark/InferSent_preprocessed/'


class STSBenchmarkEval(object):
    def __init__(self, seed=1111):
        self.seed = seed
        train = load_file(DATA_PATH + 'sts-train.csv')
        dev = load_file(DATA_PATH + 'sts-dev.csv')
        test = load_file(DATA_PATH + 'sts-test.csv')
        self.sick_data = {'train': train, 'dev': dev, 'test': test}

    def do_prepare(self, params):
        samples = self.sick_data['train']['X_A'] + self.sick_data['train']['X_B'] + \
                  self.sick_data['dev']['X_A'] + self.sick_data['dev']['X_B'] + \
                  self.sick_data['test']['X_A'] + self.sick_data['test']['X_B']
        return prepare(params, samples)

    def run(self, params):
        sick_embed = {'train': {}, 'dev': {}, 'test': {}}   # type: dict
        bsize = params.batch_size

        for key in self.sick_data:
            print('Computing embedding for {0}...'.format(key))
            ts = timer()
            # Sort to reduce padding
            sorted_corpus = sorted(zip(self.sick_data[key]['X_A'],
                                       self.sick_data[key]['X_B'],
                                       self.sick_data[key]['y']),
                                   key=lambda z: (len(z[0]), len(z[1]), z[2]))

            self.sick_data[key]['X_A'] = [x for (x, y, z) in sorted_corpus]
            self.sick_data[key]['X_B'] = [y for (x, y, z) in sorted_corpus]
            self.sick_data[key]['y'] = [z for (x, y, z) in sorted_corpus]

            for txt_type in ['X_A', 'X_B']:
                sick_embed[key][txt_type] = []
                for ii in range(0, len(self.sick_data[key]['y']), bsize):
                    batch = self.sick_data[key][txt_type][ii:ii + bsize]
                    embeddings = batcher(params, batch)
                    sick_embed[key][txt_type].append(embeddings)
                sick_embed[key][txt_type] = np.vstack(sick_embed[key][txt_type])
            sick_embed[key]['y'] = np.array(self.sick_data[key]['y'])
            print('Computed {0} embeddings.'.format(key))
            print('Time elapsed:', timer()-ts, 's\n')

        # Train
        train_a = sick_embed['train']['X_A']
        train_b = sick_embed['train']['X_B']
        train_f = np.c_[np.abs(train_a - train_b), train_a * train_b]
        train_y = encode_labels(self.sick_data['train']['y'])

        # Dev
        dev_a = sick_embed['dev']['X_A']
        dev_b = sick_embed['dev']['X_B']
        dev_f = np.c_[np.abs(dev_a - dev_b), dev_a * dev_b]
        dev_y = encode_labels(self.sick_data['dev']['y'])

        # Test
        test_a = sick_embed['test']['X_A']
        test_b = sick_embed['test']['X_B']
        test_f = np.c_[np.abs(test_a - test_b), test_a * test_b]
        test_y = encode_labels(self.sick_data['test']['y'])

        config_classifier = {'seed': self.seed, 'nclasses': 5}
        clf = RelatednessPytorch(train={'X': train_f, 'y': train_y},
                                 valid={'X': dev_f, 'y': dev_y},
                                 test={'X': test_f, 'y': test_y},
                                 devscores=self.sick_data['dev']['y'],
                                 config=config_classifier)
        devpr, yhat = clf.run()

        pr = pearsonr(yhat, self.sick_data['test']['y'])[0]
        sr = spearmanr(yhat, self.sick_data['test']['y'])[0]
        se = mean_squared_error(yhat, self.sick_data['test']['y'])

        return {'devpearson': devpr, 'pearson': pr, 'spearman': sr, 'mse': se,
                'yhat': yhat, 'ndev': len(dev_a), 'ntest': len(test_a)}


class RelatednessPytorch(object):
    def __init__(self, train, valid, test, devscores, config):
        # fix seed
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        assert torch.cuda.is_available(), 'torch.cuda required for Relatedness'
        torch.cuda.manual_seed(config['seed'])

        self.train = train
        self.valid = valid
        self.test = test
        self.devscores = devscores

        self.inputdim = train['X'].shape[1]
        self.nclasses = config['nclasses']
        self.seed = config['seed']
        self.l2reg = 0.
        self.batch_size = 64
        self.maxepoch = 1000
        self.early_stop = True

        self.model = nn.Sequential(nn.Linear(self.inputdim, self.nclasses), nn.Softmax())
        self.loss_fn = nn.MSELoss()

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()

        self.loss_fn.size_average = False
        self.optimizer = optim.Adam(self.model.parameters(), weight_decay=self.l2reg)

        self.nepoch = None

    def run(self):
        self.nepoch = 0
        bestpr = -1
        early_stop_count = 0
        r = np.arange(1, 6)
        stop_train = False

        # Preparing data
        train_x, train_y, dev_x, dev_y, test_x, test_y = prepare_data(
            self.train['X'], self.train['y'],
            self.valid['X'], self.valid['y'],
            self.test['X'], self.test['y'])

        # Training
        print('Training LSTM...')
        ts = timer()
        bestmodel = None
        while not stop_train and self.nepoch <= self.maxepoch:
            self.train_epoch(train_x, train_y, nepoches=50)
            yhat = np.dot(self.predict_proba(dev_x), r)
            pr = pearsonr(yhat, self.devscores)[0]
            # early stop on Pearson
            if pr > bestpr:
                bestpr = pr
                bestmodel = copy.deepcopy(self.model)
            elif self.early_stop:
                if early_stop_count >= 3:
                    stop_train = True
                early_stop_count += 1
        self.model = bestmodel
        print('Time elapsed:', timer()-ts, 's\n')

        print('Predicting scores...')
        ts = timer()
        yhat = np.dot(self.predict_proba(test_x), r)
        print('Time elapsed:', timer()-ts, 's\n')

        return bestpr, yhat

    def train_epoch(self, x, y, nepoches=1):
        self.model.train()
        for _ in range(self.nepoch, self.nepoch + nepoches):
            permutation = np.random.permutation(len(x))
            all_costs = []
            for i in range(0, len(x), self.batch_size):
                # forward
                idx = torch.from_numpy(permutation[i:i + self.batch_size]).long().cuda()
                xbatch = Variable(x.index_select(0, idx))
                ybatch = Variable(y.index_select(0, idx))
                output = self.model(xbatch)
                # loss
                loss = self.loss_fn(output, ybatch)
                all_costs.append(loss.data[0])
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                # Update parameters
                self.optimizer.step()
        self.nepoch += nepoches

    def predict_proba(self, devx):
        self.model.eval()
        probas = []
        for i in range(0, len(devx), self.batch_size):
            xbatch = Variable(devx[i:i + self.batch_size], volatile=True)
            if len(probas) == 0:
                probas = self.model(xbatch).data.cpu().numpy()
            else:
                probas = np.concatenate((probas, self.model(xbatch).data.cpu().numpy()), axis=0)
        return probas


class DotDict(dict):
    """Dot notation access to dictionary attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_file(fpath):
    """Load the STS Benchmark csv format."""
    sick_data = {'X_A': [], 'X_B': [], 'y': []}
    with io.open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip().split('\t')
            sick_data['X_A'].append(text[5].split())
            sick_data['X_B'].append(text[6].split())
            sick_data['y'].append(text[4])
    sick_data['y'] = [float(s) for s in sick_data['y']]
    return sick_data


def encode_labels(labels, nclass=5):
    """Label encoding from Tree LSTM paper (Tai, Socher, Manning)."""
    y = np.zeros((len(labels), nclass)).astype('float32')
    for j, yj in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(yj) + 1:
                y[j, i] = yj - np.floor(yj)
            if i+1 == np.floor(yj):
                y[j, i] = np.floor(yj) - yj + 1
    return y


def prepare_data(train_x, train_y, dev_x, dev_y, test_x, testy):
    """Transform probs to log-probs for KL-divergence."""
    train_x = torch.FloatTensor(train_x).cuda()
    train_y = torch.FloatTensor(train_y).cuda()
    dev_x = torch.FloatTensor(dev_x).cuda()
    dev_y = torch.FloatTensor(dev_y).cuda()
    test_x = torch.FloatTensor(test_x).cuda()
    test_y = torch.FloatTensor(testy).cuda()
    return train_x, train_y, dev_x, dev_y, test_x, test_y


def prepare(params, samples):
    params.infersent.build_vocab([' '.join(s) for s in samples], tokenize=False)


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params.infersent.encode(sentences, bsize=params.batch_size, tokenize=False)
    return embeddings


if __name__ == "__main__":
    print('Loading model...')
    timer_start = timer()
    params_senteval = DotDict({'usepytorch': True, 'classifier': 'LogReg', 'nhid': 0,
                               'batch_size': 128, 'seed': 1111, 'kfold': 5})
    params_senteval.infersent = torch.load(MODEL_PATH, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0'})
    params_senteval.infersent.set_glove_path(GLOVE_PATH)

    evaluation = STSBenchmarkEval(seed=params_senteval.seed)
    evaluation.do_prepare(params_senteval)
    print('Time elapsed:', timer()-timer_start, 's\n')

    results = evaluation.run(params_senteval)
    print(results)
