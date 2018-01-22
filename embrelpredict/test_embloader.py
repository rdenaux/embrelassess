import embloader
import unittest


class SwivelAsTorchTextVector(unittest.TestCase):

    def test_create(self):
        base = '../tests/data/kcap-basic-vec/'
        dim = 300
        vecs = embloader.SwivelAsTorchTextVector(
            base+'vecs.bin',
            base+'vocab.txt',
            dim)
        self.assertIsNotNone(vecs)

    def test_no_vocab_map(self):
        base = '../tests/data/kcap-wsd-vec/'
        dim = 300
        vecs = embloader.SwivelAsTorchTextVector(
            base+'vecs.bin',
            base+'vocab.txt',
            dim)
        syntok = 'en#77408'
        self.assertTrue(syntok in vecs.stoi)
        self.assertEqual(1, len(vecs[syntok].shape))
        self.assertAlmostEqual(0.189, vecs[syntok][0], places=3)
        simple_syntok = '#77408'
        self.assertFalse(simple_syntok in vecs.stoi)
        self.assertEqual(2, len(vecs[simple_syntok].shape))
        vec = vecs[simple_syntok][0]
        # print('vec', type(vec), vec.shape)
        self.assertAlmostEqual(0.000, vec[0], places=3)

    def test_vocab_map(self):
        base = '../tests/data/kcap-wsd-vec/'
        dim = 300
        vecs = embloader.SwivelAsTorchTextVector(
            base+'vecs.bin',
            base+'vocab.txt',
            dim,
            vocab_map=embloader.simple_syns)
        syntok = 'en#77408'
        self.assertFalse(syntok in vecs.stoi)
        self.assertEqual(2, len(vecs[syntok].shape))
        self.assertAlmostEqual(0.000, vecs[syntok][0][0], places=3)
        simple_syntok = '#77408'
        self.assertTrue(simple_syntok in vecs.stoi)
        self.assertEqual(1, len(vecs[simple_syntok].shape))
        vec = vecs[simple_syntok]
        # print('vec', type(vec), vec.shape)
        self.assertAlmostEqual(0.189, vec[0], places=3)


class VecPairLoaderTest(unittest.TestCase):

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
