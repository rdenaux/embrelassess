import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np


class ModelTrainer():
    """Provides methods for training and evaluating a binary classifier model
    """

    def __init__(self, model, criterion=None, optimizer=None,
                 scheduler=None, cuda=False):
        """Creates a ModelTrainer

        Args:
        model a pytorch Module, assumed to be a binary classifier
        criterion a pytorch criterion, if None provided will use nn.CrossEntropyLoss
        optimizer a pytorch optimizer, if None provided will use optim.Adam
        scheduler a pytorch scheduler, if None provided will use ReduceLROnPlateau
        cuda (boolean): whether to use GPU optimization
        """
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

        param_cnt = 0
        for p in model.parameters():
            param_cnt += 1

        if optimizer:
            self.optimizer = optimizer
        elif param_cnt > 0:
            self.optimizer = optim.Adam(model.parameters(), lr=0.00001)
        else:
            self.optimizer = optimizer

        if scheduler:
            self.scheduler = scheduler
        elif self.optimizer:
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')
        else:
            self.scheduler = scheduler
        self.cuda = cuda
        self.epoch = 0
        self.step = 0
        self.columns = ["epoch", "step", "train/loss", "valid/loss",
                        "valid/precision", "valid/recall", "valid/acc",
                        "valid/f1"]
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
            running_loss += loss.item() # pytorch < 0.4: loss.data[0]
        return running_loss/cnt

    def log_msg(self, running_train_loss, vloss, valid_test_result):
        return 'e%d, s%5d] loss %.3f valid_loss %.3f' % (
            self.epoch + 1, self.step, running_train_loss, vloss)

    def as_training_step_data(self, running_train_loss, vloss,
                              valid_test_result):
        return [self.epoch + 1, self.step, running_train_loss, vloss,
                self.precision(valid_test_result),
                self.recall(valid_test_result),
                self.acc(valid_test_result), self.f1(valid_test_result)]

    def train(self, loader, validloader, epochs_list=range(2), log_every=250,
              input_disturber=None):
        training_data = []
        if not self.optimizer:
            print('Skipping training since no optimizer')
            tdf = pd.DataFrame(training_data, columns=self.columns)
            self.df = self.df.append(tdf)
            return tdf

        self.model.train(True)
        # pbar = tqdm(range(epochs))
        for epoch in epochs_list:
            running_loss = 0.0
            cnt = 0
            self.optimizer.zero_grad()  # zero the parameter gradients
            for i, data in enumerate(loader, 0):
                # get the inputs
                inputs, labels = data
                if input_disturber:
                    inputs = input_disturber(inputs)
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
                running_loss += loss.item() # pytorch < 0.4: loss.data[0]
                cnt += 1
                if i % log_every == 0:  # log
                    vloss = self.valid_loss(validloader)
                    vtest_result = self.test(validloader)
                    self.scheduler.step(vloss)
                    data = self.as_training_step_data(running_loss / cnt,
                                                      vloss, vtest_result)
                    training_data.append(data)
                    running_loss = 0.0
                    cnt = 0

            if cnt != 0:
                vloss = self.valid_loss(validloader)
                vtest_result = self.test(validloader)
                data = self.as_training_step_data(running_loss / cnt,
                                                  vloss, vtest_result)
                training_data.append(data)
            self.epoch += 1

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
            # print('data shape %s' % labels.shape)
            total += labels.size(0)
            predicted = torch.LongTensor(
                np.random.randint(2, size=labels.size(0)))
            tp += (predicted + labels == 2).sum()
            fp += (predicted - labels == 1).sum()
            fn += (predicted - labels == -1).sum()
            correct += (predicted == labels).sum()
        result = {"model": "randpredict", "total_examples": total,
                  "threshold": 1.0, "examples_above_threshold": total,
                  #"correct": correct, "tp": tp, "fp": fp, "fn": fn}
                  "correct": correct.item(), "tp": tp.item(), "fp": fp.item(), "fn": fn.item()}
        self.print_test_result(result)
        return result

    def print_test_result(self, result):
        above_thresh = result['examples_above_threshold']
        total = result['total_examples']
        threshold = result['threshold']
        model_name = result['model']
        if above_thresh > 0:
            praf = 'prec, rec, acc, f1: %d %d %d %d %%' % (
                100 * self.precision(result), 100 * self.recall(result),
                   100 * self.acc(result), 100 * self.f1(result))
            print('test result', result)
            print('For %s on %d(%.2f%%) of %d examples > %.2f confidence, %s' %
                  (model_name, above_thresh, 100 * (above_thresh/total),
                   total, threshold, praf))
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

    def f1(self, test_result):
        tp = test_result['tp']
        fp = test_result['fp']
        fn = test_result['fn']
        denom = (2*tp + fp + fn)
        if denom == 0:
            return 0
        else:
            return 2*tp / denom

    def acc(self, test_result):
        correct = test_result['correct']
        total = test_result['examples_above_threshold']
        return correct / total

    def test(self, loader, threshold=0.0, debug=False):
        """Tests the net on a loader test set for a given threshold.

        Args:
          loader a pytorch data loader providing the test set
          threshold to test against (a float from 0.0 to 1.0)
          debug boolean that triggers extra logging

        Returns:
          dictionary with:
            model the name of this ModelTrainer's model,
            total_examples int total examples in loader
            threshold float the input threshold
            examples_above_threshold int
            correct int sum of tp and true negatives
            tp int number of true positives
            fp int number of false positives
            fn int number of false negatives
        """
        correct = 0
        tp = 0
        fp = 0
        fn = 0
        total = 0
        above_thresh = 0
        softmax = nn.LogSoftmax(dim=1)
        self.model.train(False)
        for data in loader:
            inputs, labels = data
            if self.cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(Variable(inputs))
            sout = softmax(outputs)
            vals, predicted = torch.max(sout.data, dim=1)
            confidence = vals + 1.0
            mask = confidence.ge(threshold)
            above_thresh += mask.sum()

            masked_predicted = torch.masked_select(predicted, mask)
            masked_labels = torch.masked_select(labels, mask)
            tp += (masked_predicted + masked_labels == 2).sum()
            fp += (masked_predicted - masked_labels == 1).sum()
            fn += (masked_predicted - masked_labels == -1).sum()
            correct += (masked_predicted == masked_labels).sum()  # tp + tn
            if debug and total == 0:
                print(sout.data[0], confidence[0], predicted[0])
            total += labels.size(0)
        result = {"model": self.model_name, "total_examples": total,
                  "threshold": threshold,
                  "examples_above_threshold": above_thresh.item(), "correct": correct.item(),
                  "tp": tp.item(), "fp": fp.item(), "fn": fn.item()}
        if debug:
            print("test result", result)
        return result

    def test_df(self, loader, debug=False):
        result = []
        for threshold in np.arange(0.0, 1.0, 0.05):
            result.append(self.test(loader, threshold=threshold, debug=debug))
        return pd.DataFrame(result, columns=[
            "model", "total_examples",
            "threshold", "examples_above_threshold", "correct",
            "tp", "fp", "fn"])
