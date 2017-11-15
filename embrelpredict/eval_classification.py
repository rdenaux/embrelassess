import sys
import vector_operations as vecops
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
import matplotlib.pyplot as plt
import seaborn

class VocabPairLoader():
    def __init__(self, vocabs):
        self.vocabs = vocabs

    def vecOrEmpty(self, word):
        res = self.vocabs[word]
        if not (res.shape[0] == self.vocabs.dim):
            return torch.zeros(self.vocabs.dim)
        else:
            return res


    def avg_vec(self, compound_word):
        vec = self.vocabs[compound_word]
        if (vec.shape[0] == self.vocabs.dim):
            return vec
        else: # word not in vocab
            sum = torch.zeros(self.vocabs.dim)
            words = compound_word.split(' ')
            for w in words:
                sum = sum + self.vecOrEmpty(w)
            return sum / len(words)


    def pairEmbed(self):
        def _pairEmbed(dfrow):
            par = self.avg_vec(dfrow[0])
            chi = self.avg_vec(dfrow[1])
            assert par.shape[0] == self.vocabs.dim
            assert chi.shape[0] == self.vocabs.dim
            return torch.cat([par, chi])
        return _pairEmbed

    def firstEmbed(self):
        def _firstEmbed(dfrow):
            par = self.avg_vec(dfrow[0])
            return par
        return _firstEmbed

    def load_data(self, tsv_file, dfrow_handler):
        """Loads pair classification data from a TSV file.
        By default, we assume each row contains two vocabulary terms followed
        by an integer value for the class of the pair.
        """
        df = pd.read_csv(tsv_file, header=None, sep='\t')
        # print(df.columns.size)
        assert(df.columns.size == 3), 'error'
        # extract categories (no need to one-hot encoding since pytorch takes care of this)
        categories = df.loc[:, 2]
        cat_idxs = torch.LongTensor(categories.values)
        cat_idxs = cat_idxs
        Y = torch.LongTensor(cat_idxs)
        # now extract pairs
        X = torch.stack(df.apply(dfrow_handler, axis=1).values)
        return X, Y

    def load_pair_data(self, tsv_file):
        return self.load_data(tsv_file, dfrow_handler=self.pairEmbed())

    def load_single_data(self, tsv_file):
        return self.load_data(tsv_file, dfrow_handler=self.firstEmbed())

    def _train_validate_test_split(self, tds, train_percent=.9,
                                  validate_percent=.05,
                                  seed=None):
        np.random.seed(seed)
        perm = np.random.permutation(tds.data_tensor.shape[0])
        m = len(tds)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train_ids = perm[:train_end]
        validate_ids = perm[train_end:validate_end]
        test_ids = perm[validate_end:]
        return train_ids, validate_ids, test_ids

    def _split_data(self, tds, train_ids, validate_ids, test_ids, batch_size):
        trainloader = DataLoader(tds, batch_size=batch_size,
                                 sampler=SubsetRandomSampler(train_ids),
                                 num_workers=2)
        validloader = DataLoader(tds, batch_size=batch_size,
                                 sampler=SubsetRandomSampler(validate_ids),
                                 num_workers=2)
        testloader = DataLoader(tds, batch_size=batch_size,
                                sampler=SubsetRandomSampler(test_ids),
                                num_workers=2)
        return trainloader, validloader, testloader

    def split_data(self, X, Y, train_percent=.9, validate_percent=.05, seed=None, batch_size=32):
        tds = TensorDataset(X, Y)
        tr_ids, v_ids, te_ids = self._train_validate_test_split(tds, train_percent, validate_percent, seed)
        print('train %s, validate %s, test %s' % (tr_ids.shape, v_ids.shape, te_ids.shape))
        return self._split_data(tds, tr_ids, v_ids, te_ids, batch_size)


class PairDataLoader():
    def __init__(self, vecs_bin_path, vecs_vocab_path, vecs_dims=300):
        lang = 'en'
        self.vocab, self.vecs = vecops.generateVectors(vecs_bin_path, vecs_vocab_path, vecs_dims, lang)
        self.stoi = dict([(s,i) for i, s in enumerate(self.vocab)])

    def pairEmbed(self):
        def _pairEmbed(dfrow):
            par = self.vecs[self.stoi[str(dfrow[0]).replace('en#', '#').strip()]]
            chi = self.vecs[self.stoi[str(dfrow[1]).replace('en#', '#').strip()]]
            # print(type(par), par.shape) print(type(chi), chi.shape)
            res = np.concatenate([par, chi])
            # print(type(res), res.shape) print(type(res[0]))
            return torch.from_numpy(res.astype(np.float32))
        return _pairEmbed

    def firstEmbed(self):
        def _firstEmbed(dfrow):
            par = self.vecs[self.stoi[str(dfrow[0]).replace('en#', '#').strip()]]
            # chi = vecs[stoi[dfrow[1].replace('en#', '#')]]
            # print(type(par), par.shape)
            # print(type(chi), chi.shape)
            res = par
            # print(type(res), res.shape)
            # print(type(res[0]))
            return torch.from_numpy(res.astype(np.float32))
        return _firstEmbed

    def load_data(self, tsv_file, dfrow_handler):
        """Loads pair classification data from a TSV file.
        By default, we assume each row contains two vocabulary terms followed
        by an integer value for the class of the pair.
        """
        df = pd.read_csv(tsv_file, header=None, sep='\t')
        # print(df.columns.size)
        assert(df.columns.size == 3), 'error'
        # extract categories (no need to one-hot encoding since pytorch takes care of this)
        categories = df.loc[:, 2]
        cat_idxs = torch.LongTensor(categories.values)
        cat_idxs = cat_idxs
        Y = torch.LongTensor(cat_idxs)
        # now extract pairs
        X = torch.stack(df.apply(dfrow_handler, axis=1).values)
        return X, Y


    def generate_random_pair_data(self, target_size):
       """Generates random pairs based on the vocab and generates training data
       """
       num_vecs = len(self.vecs)
       parent_ids = np.random.randint(num_vecs, size=target_size)
       child_ids = np.random.randint(num_vecs, size=target_size)
       X = []
       for i in range(target_size):
           par_vec = self.vecs[parent_ids[i]]
           chi_vec = self.vecs[child_ids[i]]
           res = np.concatenate([par_vec, chi_vec])
           X.append(torch.from_numpy(res.astype(np.float32)))
       X = torch.stack(X)
       Y = torch.LongTensor(np.random.randint(2, size=target_size))
       return X, Y

    def load_pair_data(self, tsv_file):
        return self.load_data(tsv_file, dfrow_handler=self.pairEmbed())

    def load_single_data(self, tsv_file):
        return self.load_data(tsv_file, dfrow_handler=self.firstEmbed())

    def _train_validate_test_split(self, tds, train_percent=.9,
                                  validate_percent=.05,
                                  seed=None):
        np.random.seed(seed)
        perm = np.random.permutation(tds.data_tensor.shape[0])
        m = len(tds)
        train_end = int(train_percent * m)
        validate_end = int(validate_percent * m) + train_end
        train_ids = perm[:train_end]
        validate_ids = perm[train_end:validate_end]
        test_ids = perm[validate_end:]
        return train_ids, validate_ids, test_ids

    def _split_data(self, tds, train_ids, validate_ids, test_ids, batch_size):
        trainloader = DataLoader(tds, batch_size=batch_size,
                                 sampler=SubsetRandomSampler(train_ids),
                                 num_workers=2)
        validloader = DataLoader(tds, batch_size=batch_size,
                                 sampler=SubsetRandomSampler(validate_ids),
                                 num_workers=2)
        testloader = DataLoader(tds, batch_size=batch_size,
                                sampler=SubsetRandomSampler(test_ids),
                                num_workers=2)
        return trainloader, validloader, testloader

    def split_data(self, X, Y, train_percent=.9, validate_percent=.05, seed=None, batch_size=32):
        tds = TensorDataset(X, Y)
        tr_ids, v_ids, te_ids = self._train_validate_test_split(tds, train_percent, validate_percent, seed)
        print('train %s, validate %s, test %s' % (tr_ids.shape, v_ids.shape, te_ids.shape))
        return self._split_data(tds, tr_ids, v_ids, te_ids, batch_size)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, 2, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


class NNBiClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, hidden_node_dropout_ps=None):
        super(NNBiClassifier, self).__init__()
        self.hidden_layer_cnt = len(hidden_dims)
        fcs = []
        dropouts = []
        if hidden_node_dropout_ps:
            assert(len(hidden_node_dropout_ps) == self.hidden_layer_cnt)
        else:
            hidden_node_dropout_ps = [0.5 for x in range(self.hidden_layer_cnt)]
        for hli in range(self.hidden_layer_cnt + 1):
            in_dim = input_dim if hli == 0 else hidden_dims[hli - 1]
            out_dim = 2 if hli == self.hidden_layer_cnt else hidden_dims[hli]
            fcs.append(nn.Linear(in_dim, out_dim))
            if (hli < self.hidden_layer_cnt):
                dropouts.append(nn.Dropout(p=hidden_node_dropout_ps[hli]))
        print('fcs %d, dropouts %d' % (len(fcs), len(dropouts)))
        self.fcs = nn.ModuleList(fcs)
        self.dropouts = nn.ModuleList(dropouts)


    def forward(self, x):
        for hli in range(self.hidden_layer_cnt):
            x = self.dropouts[hli](nn.functional.relu(self.fcs[hli](x)))
        return self.fcs[self.hidden_layer_cnt](x)



class ModelTrainer():
    def __init__(self, model, criterion=None, optimizer=None,
                 scheduler=None, cuda=False):
        # print('Model Trainer for ', model)
        if cuda:
            self.model = model.cuda()
        else:
            self.model = model
        self.model_name = self.model.__class__.__name__
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = nn.CrossEntropyLoss()
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=0.00001)
        if scheduler:
            self.scheduler = scheduler
        else:
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        self.cuda = cuda
        self.epoch = 0
        self.step = 0
        self.columns = ["epoch", "step", "train/loss", "valid/loss", "valid/precision", "valid/recall", "valid/acc"]
        self.df = pd.DataFrame(columns=self.columns)


    def valid_loss(self, loader):
        running_loss = 0.0
        cnt = 0
        for i, data in enumerate(loader, 0):
            cnt += 1
            inputs, labels = data
            if self.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = Variable(inputs), Variable(labels)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.data[0]
        return running_loss/cnt

    def log_msg(self, running_train_loss, vloss, valid_test_result):
        return 'e%d, s%5d] loss %.3f valid_loss %.3f' % (self.epoch + 1, self.step, running_train_loss, vloss)

    def as_training_step_data(self, running_train_loss, vloss, valid_test_result):
        return [self.epoch + 1, self.step, running_train_loss, vloss,
                self.precision(valid_test_result), self.recall(valid_test_result), 
                self.acc(valid_test_result)]

    def train(self, loader, validloader, epochs=2, log_every=250):
        training_data = []
        self.model.train(True)
        #pbar = tqdm(total=epochs * len(loader), file=sys.stdout)
        pbar = tqdm(range(epochs))
        #for epoch in range(epochs):  # loop over the dataset multiple times
        for epoch in pbar:
            running_loss = 0.0
            cnt = 0
            self.optimizer.zero_grad()  # zero the parameter gradients
            for i, data in enumerate(loader, 0):
                # get the inputs
                inputs, labels = data
                if self.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                self.step += 1

                # print statistics
                running_loss += loss.data[0]
                cnt += 1
                if i % log_every == 0:  # log
                    vloss = self.valid_loss(validloader)
                    vtest_result = self.test(validloader)
                    self.scheduler.step(vloss)
                    data = self.as_training_step_data(running_loss / cnt, vloss, vtest_result)
                    training_data.append(data)
                    #pbar.set_description(self.log_msg(running_loss / cnt, vloss, vtest_result))
                    running_loss = 0.0
                    cnt = 0

            if cnt != 0:
                vloss = self.valid_loss(validloader)
                vtest_result = self.test(validloader)
                data = self.as_training_step_data(running_loss / cnt, vloss, vtest_result)
                training_data.append(data)
                #pbar.set_description(self.log_msg(running_loss/ cnt, vloss, vtest_result))
            self.epoch += 1

        print('Finished %s epochs of training' % epochs)
        #pbar.close()
        tdf = pd.DataFrame(training_data, columns=self.columns)
        self.df = self.df.append(tdf)
        return tdf

    def test_random(self, loader):
        """Test the examples from a loader against a random predictor
        """
        correct = 0
        tp = 0
        fp = 0
        fn = 0
        total = 0
        for data in loader:
            inputs, labels = data
            #print('data shape %s' % labels.shape)
            total += labels.size(0)
            predicted = torch.LongTensor(np.random.randint(2, size=labels.size(0)))
            tp += (predicted + labels == 0).sum()
            fp += (predicted - labels == -1).sum()
            fn += (predicted - labels == 1).sum()
            correct += (predicted == labels).sum()
        result = {"model": "randpredict", "total_examples": total, "threshold": 1.0, "examples_above_threshold": total, 
                  "correct": correct, "tp": tp, "fp": fp, "fn": fn}
        self.print_test_result(result)
        return result

    def print_test_result(self, result):
       above_thresh = result['examples_above_threshold']
       total = result['total_examples']
       threshold = result['threshold']
       model_name = result['model']
       if above_thresh > 0:
           print('Precision, Recall and Accuracy of %s on %d(%.2f%%) of the %d test examples above %.2f confidence: %d %d %d %%' %
               (model_name, above_thresh, 100 * (above_thresh/total), total, threshold,
                100 * self.precision(result), 100 * self.recall(result), 100 * self.acc(result)))
       else:
           print('%d(%.2f%%) of the %d test examples above %.2f confidence' %
               (above_thresh, 100 * (above_thresh/total), total, threshold))

    def precision(self, test_result):
       tp = test_result['tp']
       fp = test_result['fp']
       if tp + fp == 0:
           return 0
       else:
           return tp / (tp + fp)

    def recall(self, test_result):
       tp = test_result['tp']
       fn = test_result['fn']
       if tp + fn == 0:
           return 0
       else:
           return tp / (tp + fn)

    def acc(self, test_result):
       correct = test_result['correct']
       total = test_result['examples_above_threshold']
       return correct / total


    def test(self, loader, threshold=0.0):
        """Tests the net on a loader test set."""
        correct = 0
        tp = 0
        fp = 0
        fn = 0
        total = 0
        above_thresh = 0
        softmax = nn.LogSoftmax()
        self.model.train(False)
        for data in loader:
            inputs, labels = data
            if self.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(Variable(inputs))
            sout = softmax(outputs)
            vals, predicted = torch.max(sout.data, 1)
            confidence = vals + 1.0
            # print(sout.data[0], confidence[0], predicted[0], labels[0])
            mask = confidence.ge(threshold)
            total += labels.size(0)
            above_thresh += mask.sum()

            masked_predicted = torch.masked_select(predicted, mask)
            masked_labels = torch.masked_select(labels, mask)
            tp += (masked_predicted + masked_labels == 0).sum()
            fp += (masked_predicted - masked_labels == -1).sum()
            fn += (masked_predicted - masked_labels == 1).sum()
            correct += (masked_predicted == masked_labels).sum()
        result = {"model": self.model_name, "total_examples": total, "threshold": threshold,
                  "examples_above_threshold": above_thresh, "correct": correct, "tp": tp, "fp": fp, "fn": fn}
        return result

    def test_df(self, loader):
        result = []
        for threshold in np.arange(0.0, 1.0, 0.05):
            result.append(self.test(loader, threshold=threshold))
        return pd.DataFrame(result, columns=["model", "total_examples", "threshold",
             "examples_above_threshold", "correct", "tp", "fp", "fn"])


class Plotter():
    def __init__(self):
        print('Plotter')
        self.colors = seaborn.color_palette()


    def plot_learning_curve(self, df_training, model_name):
        df = df_training
        row_min = df.min()
        row_max = df.max()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.plot(df['step'], df['train/loss'], '-',
            markersize=1, color=self.colors[0], alpha=.5,
            label='train loss')
        plt.plot(df['step'], df['valid/loss'], '-',
            markersize=1, color=self.colors[1], alpha=.5,
            label='valid loss')
        plt.xlim((0, row_max['step']))
        plt.ylim((min(row_min['train/loss'], row_min['valid/loss']),
            max(row_max['train/loss'], row_max['valid/loss'])))
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('learning curve %s' % model_name)
        plt.legend()

    def plot_valid_acc(self, df_training, model_name):
        df = df_training
        row_min = df.min()
        row_max = df.max()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df['step'], df['valid/precision'], '-',
            markersize=1, color=self.colors[0], alpha=.5,
            label='precision')
        plt.plot(df['step'], df['valid/recall'], '-',
            markersize=1, color=self.colors[1], alpha=.5,
            label='recall')
        plt.plot(df['step'], df['valid/acc'], '-',
            markersize=1, color=self.colors[2], alpha=.5,
            label='accuracy')
        plt.xlim((0, row_max['step']))
        plt.ylim(0.0, 1.0)
        plt.xlabel('step')
        plt.ylabel('percent')
        plt.legend()
        plt.title('Validation results %s ' % model_name)

    def expand(self, df_test):
        df = df_test
        df['precision'] = df['tp'] / (df['tp'] + df['fp'])
        df['recall'] = df['tp'] / (df['tp'] + df['fn'])
        df['acc'] = df['correct'] / df['examples_above_threshold']
        df['coverage'] = df['examples_above_threshold']/df['total_examples']
        return df 

    def plot_test_df(self, df_test, model_name):
        df = self.expand(df_test)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df['threshold'], df['precision'], '-',
            markersize=1, color=self.colors[0], alpha=.5,
            label='precision')
        plt.plot(df['threshold'], df['recall'], '-',
            markersize=1, color=self.colors[1], alpha=.5,
            label='recall')
        plt.plot(df['threshold'], df['acc'], '-',
            markersize=1, color=self.colors[2], alpha=.5,
            label='accuracy')
        plt.plot(df['threshold'], df['coverage'], '-',
            markersize=1, color=self.colors[3], alpha=.5,
            label='coverage')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('threshold')
        plt.ylabel('percent')
        plt.legend()
        plt.title('Test results %s ' % model_name)


    def plot_learning(self, df_training, df_test, model_name, n_row=2, n_col=2, figsize=(10,6), dpi=300):
        plt.figure(figsize=figsize, dpi=dpi)

        # learning curve
        plt.subplot(n_row, n_col, 1)
        self.plot_learning_curve(df_training, model_name)

        # validation p-r-acc
        plt.subplot(n_row, n_col, 2)
        self.plot_valid_acc(df_training, model_name)

        # test p-r-acc
        plt.subplot(n_row, n_col, 3)
        self.plot_test_df(df_test, model_name)

        fig = plt.gcf()
        fig.tight_layout()

        return plt 
