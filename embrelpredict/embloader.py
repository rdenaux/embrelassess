import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import embrelpredict.vecops as vecops
import pandas as pd
import numpy as np
import logging
import os
import io
import array


def simple_syns(val):
    """normalises an input syncon name by stripping the language

    Use this method when loading Vecsigrafo embeddings to avoid having
    to specify the language every time, simply refer to syncons by
    using the '#' prefix.
    """
    return str(val).replace('en#', '#').strip()


class SwivelAsTorchTextVector(object):
    """torchtext.Vectors compatible object for Swivel embeddings
    """

    def __init__(self, vecs_bin_path, vecs_vocab_path,
                 vecs_dims=300,
                 unk_init=torch.FloatTensor.zero_,
                 vocab_map=lambda x: x):
        """Creates a SwivelAsTorchTextVector from bin and vocab files

        Args:
          vecs_bin_path a .bin file produced by Swivel
          vecs_vocab_path a vocab.txt file produced by Swivel
              this should be aligned to the vectors in the bin file
          unk_init tensor initializer for words out of vocab
          vocab_map maps original tokens to new tokens at loading
           time. This can be useful to simplify token names or to
           avoid clashes when loading multiple embedding spaces.
        """
        self.vocab, self.vecs = vecops.read_swivel_vecs(
            vecs_bin_path, vecs_vocab_path, vecs_dims)
        self.stoi = dict([(vocab_map(s), i) for i, s in
                          enumerate(self.vocab)])
        self.dim = vecs_dims
        self.vectors = torch.FloatTensor(self.vecs)
        self.unk_init = unk_init

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.FloatTensor(1, self.dim))


class RandomVectors(object):
    """torchtext.Vecrtors compatible object with random vectors for a given vocabulary
    """
    def __init__(self, vocab_path,
                 unk_init=torch.Tensor.uniform_,
                 dim=None):
        """Arguments:
               vocab_path: path of the vocab file, this may be a file with a token per
                   line, or a TSV where the first column contains the token names.
               unk_init (callback): by default, initalize word vectors
                   to random uniform vectors between 0 and 1; can be any function that
                   takes in a Tensor and returns a Tensor of the same size
         """
        self.logger = logging.getLogger(__name__)
        self.unk_init = unk_init
        assert(dim)  # a dimension must be defined
        self.load(vocab_path, dim)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))

    def load(self, vocab_path, dim=None):
        path = os.path.join(vocab_path)
        # path_pt = path + '.pt'
        if not os.path.isfile(path):
            raise RuntimeError('no vectors found at {}'.format(path))

        # str call is necessary for Python 2/3 compatibility, since
        # argument must be Python 2 str (Python 3 bytes) or
        # Python 3 str (Python 2 unicode)
        itos, vectors = [], array.array(str('d'))

        # Try to read the whole file with utf-8 encoding.
        with io.open(path, encoding="utf8") as f:
            lines = [line for line in f]

        self.logger.info("Loading vectors from {}".format(path))
        for line in lines:
            # Explicitly splitting on "\t" is important, so we don't
            # get rid of Unicode non-breaking spaces in the vectors.
            entries = line.rstrip().split("\t")
            word = entries[0]
            tens = torch.Tensor(dim).uniform_(to=2) - 1.0
            entries = tens.tolist()

            vectors.extend(float(x) for x in entries)
            itos.append(word)
        self.itos = itos
        self.stoi = {word: i for i, word in enumerate(itos)}
        self.vectors = torch.Tensor(vectors).view(-1, dim)
        self.dim = dim


class VecPairLoader():
    """Generates DataLoaders from a word embedding space and a relation file

    On one hand we have word embeddings which need to be loaded. On the other
    hand we have TSV word relation files, which provide pairs of words which
    belong to some category (as well as negative pairs). For training PyTorch
    models, we need to map the words to their embeddings to generate
    TensorDatasets,  which in practice are used during training as DataLoaders.
    This class provides methods for performing these operations.

    The embeddings are given by a PyTorch.text Vectors instance.
    """
    def __init__(self, torch_vecs):
        """Creates a VecPairLoader based on a torchtext.vocab.Vectors

        Args:
          torch_vecs a torchtext.vocab.Vectors instance (or compatible)
        """
        assert(torch_vecs)
        self.vecs = torch_vecs

    def vecOrEmpty(self, word):
        """Returns the vector for the word if in vocab, or a zero vector

        Returns:
          vector of the word if in vocab, or zero vector
          found int 0 if not found, or 1 if found
        """
        res = self.vecs[word]
        if not (res.shape[0] == self.vecs.dim):
            return torch.zeros(self.vecs.dim), 0
        else:
            return res, 1

    def avg_vec(self, compound_word):
        """Returns a vector for a possibly compound_word

        If the compound_word is in the vocab, simply returns that vector.
        Otherwise, splits the compound word and returns the average of
        the individual words.

        Returns:
          vector for the compound_word
          tok_count number of subtokens derived from compound_word
          toks_found how many of the subtokens were in the vocabulary
        """
        compound_word = str(compound_word)
        vec = self.vecs[compound_word]
        if (vec.shape[0] == self.vecs.dim):
            return vec, 1, 1
        else:  # word not in vocab
            sum = torch.zeros(self.vecs.dim)
            words = compound_word.split(' ')
            tok_count = len(words)
            toks_found = 0
            for w in words:
                w_vec, w_found = self.vecOrEmpty(w)
                sum = sum + w_vec
                toks_found = toks_found + w_found
            return sum / len(words), tok_count, toks_found

    def pairEmbed(self):
        def _pairEmbed(dfrow):
            par, pt_cnt, pt_fnd = self.avg_vec(dfrow[0])
            chi, ct_cnt, ct_fnd = self.avg_vec(dfrow[1])
            assert par.shape[0] == self.vecs.dim
            assert chi.shape[0] == self.vecs.dim
            return torch.cat([par, chi]), pt_cnt + ct_cnt, pt_fnd + ct_fnd
        return _pairEmbed

    def firstEmbed(self):
        def _firstEmbed(dfrow):
            return self.avg_vec(dfrow[0])
        return _firstEmbed

    def load_data(self, tsv_file, dfrow_handler):
        """Loads pair classification data from a TSV file.
        By default, we assume each row contains two vocabulary terms followed
        by an integer value for the class of the pair.

        Returns:
          X FloatTensor of n by x*dim for the input pairs
          Y LongTensor of n elements
          n the number of pairs, i.e. size of the dataset
          tok_count number of tokens used to provide the embeddings
            minimum value is n*x, but could be higher due to compound words
          tok_found number of tokens in the vocabulary
            maximum value is tok_count, but can be lower if the
            tsv_file contains words ouf of the vocabulary
        """
        df = pd.read_csv(tsv_file, header=None, sep='\t')
        # print(df.columns.size)
        assert(df.columns.size >= 3), 'error'
        # extract categories (pytorch takes care of 1-hot encoding)
        categories = df.loc[:, 2]
        cat_idxs = torch.LongTensor(categories.values)
        cat_idxs = cat_idxs
        Y = torch.LongTensor(cat_idxs)
        # now extract pairs
        vec_cnt_fnds = df.apply(dfrow_handler, axis=1)
        vecs = vec_cnt_fnds.apply(lambda triple: triple[0])
        cnts = vec_cnt_fnds.apply(lambda triple: triple[1])
        fnds = vec_cnt_fnds.apply(lambda triple: triple[2])
        X = torch.stack(vecs.values)
        return X, Y, df.shape[0], sum(cnts), sum(fnds)

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

    def split_data(self, X, Y, train_percent=.9,
                   validate_percent=.05, seed=None, batch_size=32):
        tds = TensorDataset(X, Y)
        tr_ids, v_ids, te_ids = self._train_validate_test_split(
            tds, train_percent, validate_percent, seed)
        print('train %s, validate %s, test %s' %
              (tr_ids.shape, v_ids.shape, te_ids.shape))
        return self._split_data(tds, tr_ids, v_ids, te_ids, batch_size)

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
        return X, Y, target_size, target_size * 2, target_size * 2
