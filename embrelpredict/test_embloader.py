import embloader
import unittest
import os.path as osp

import numpy as np
import torch


class PairDataLoaderTest(unittest.TestCase):
    def test_create_missing_bin(self):
        with self.assertRaises(FileNotFoundError):
            embloader.PairDataLoader('', '', 300)

    def test_create_ok(self):
        base = '../tests/data/kcap-basic-vec/'
        pdl = embloader.PairDataLoader(base + 'vecs.bin',
                                       base+'vocab.txt', 300)
        self.assertTrue(pdl)

    def test_load_pair_data(self):
        base = '../tests/data/kcap-basic-vec/'
        pdl = embloader.PairDataLoader(base + 'vecs.bin',
                                       base+'vocab.txt', 300)
        X, Y = pdl.load_pair_data(base+'pair.tsv')
        self.assertEqual(6, X.shape[0])
        self.assertEqual(600, X.shape[1])
        self.assertEqual(6, Y.shape[0])

    def test_load_single_data(self):
        base = '../tests/data/kcap-basic-vec/'
        pdl = embloader.PairDataLoader(base + 'vecs.bin',
                                       base+'vocab.txt', 300)
        X, Y = pdl.load_single_data(base+'pair.tsv')
        self.assertEqual(6, X.shape[0])
        self.assertEqual(300, X.shape[1])
        self.assertEqual(6, Y.shape[0])

    def test_split_data(self):
        base = '../tests/data/kcap-basic-vec/'
        pdl = embloader.PairDataLoader(base + 'vecs.bin',
                                       base+'vocab.txt', 300)
        X, Y = pdl.load_pair_data(base+'pair.tsv')
        tr, va, te = pdl.split_data(X, Y, train_percent=.5,
                                    validate_percent=0.17,
                                    seed=3, batch_size=1)
        self.assertIsNotNone(tr)
        self.assertEqual(3, len(tr))
        self.assertIsNotNone(va)
        self.assertEqual(1, len(va))
        self.assertIsNotNone(te)
        self.assertEqual(2, len(te))


class TextVectors(object):
    def __init__(self, stoi, vectors, dim, unk_init=torch.Tensor.zero_):
        self.stoi = stoi
        self.vectors = vectors
        self.dim = dim
        self.unk_init = unk_init

    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1, self.dim))


class VecPairLoaderTest(unittest.TestCase):

    def setUp_manual(self):
        words = ['are', 'on', 'can', 'knowledge', 'using',
                 'between', 'number', 'two', 'patterns',
                 'algorithm', 'human', 'classification']
        stoi = {word: i for i, word in enumerate(words)}

        dim = 300
        rnd_vecs = np.random.rand(len(words), dim)
        vectors = torch.FloatTensor(rnd_vecs)
        self.v = TextVectors(stoi, vectors, dim)

    def setUp(self):
        base = '../tests/data/kcap-basic-vec/'
        dim = 300
        self.v = embloader.SwivelAsTorchTextVector(
            base+'vecs.bin',
            base+'vocab.txt',
            dim)

    def test_create_missing_vectors(self):
        with self.assertRaises(AssertionError):
            embloader.VecPairLoader(None)

    def test_create_ok(self):
        vpl = embloader.VecPairLoader(self.v)
        self.assertTrue(vpl)

    def test_load_pair_data(self):
        base = '../tests/data/kcap-basic-vec/'

        pdl = embloader.VecPairLoader(self.v)
        X, Y, n, tc, tf = pdl.load_pair_data(base+'pair.tsv')
        self.assertEqual(6, X.shape[0])
        self.assertEqual(6, n)
        self.assertEqual(n*2, tc)  # all pairs are single words
        self.assertEqual(n*2, tf)  # all pairs in vocab
        self.assertEqual(600, X.shape[1])
        self.assertEqual(6, Y.shape[0])

    def test_load_pair_missing_word(self):
        base = '../tests/data/kcap-basic-vec/'

        pdl = embloader.VecPairLoader(self.v)
        X, Y, n, tc, tf = pdl.load_pair_data(base+'pair2.tsv')
        self.assertEqual(7, X.shape[0])
        self.assertEqual(7, n)
        self.assertEqual(n*2, tc)  # all pairs are single words
        self.assertEqual(13, tf)  # one word out of vocab
        self.assertEqual(600, X.shape[1])
        self.assertEqual(7, Y.shape[0])

    def test_load_single_data(self):
        base = '../tests/data/kcap-basic-vec/'

        pdl = embloader.VecPairLoader(self.v)
        X, Y, n, tc, tf = pdl.load_single_data(base+'pair.tsv')
        self.assertEqual(6, X.shape[0])
        self.assertEqual(300, X.shape[1])
        self.assertEqual(6, Y.shape[0])

    def test_split_data(self):
        base = '../tests/data/kcap-basic-vec/'

        pdl = embloader.VecPairLoader(self.v)
        X, Y, _, _, _ = pdl.load_pair_data(base+'pair.tsv')
        tr, va, te = pdl.split_data(X, Y, train_percent=.5,
                                    validate_percent=0.17,
                                    seed=3, batch_size=1)
        self.assertIsNotNone(tr)
        self.assertEqual(3, len(tr))
        self.assertIsNotNone(va)
        self.assertEqual(1, len(va))
        self.assertIsNotNone(te)
        self.assertEqual(2, len(te))


if __name__ == '__main__':
    unittest.main()
