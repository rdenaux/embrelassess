import torch
import torch.nn as nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, 2, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


class DummyBiClassifier(nn.Module):
    def __init__(self, input_dim, predef=[0.01, 0.99]):
        super(DummyBiClassifier, self).__init__()
        print('Dummy Bi Classifier')
        assert(len(predef) == 2)
        self.out = torch.Tensor(predef)

    def forward(self, x):
        if x.is_cuda:
            out = self.out.cuda()
        else:
            out = self.out
        return out.expand([x.size()[0], 2])


class NNBiClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, hidden_node_dropout_ps=None):
        super(NNBiClassifier, self).__init__()
        self.hidden_layer_cnt = len(hidden_dims)
        fcs = []
        dropouts = []
        if hidden_node_dropout_ps:
            assert(len(hidden_node_dropout_ps) == self.hidden_layer_cnt)
        else:
            hidden_node_dropout_ps = [.5 for x in range(self.hidden_layer_cnt)]
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
