import unittest
import vecops


class VecopsTest(unittest.TestCase):

    def test_read_swivel_vecs(self):
        base = '../tests/data/kcap-basic-vec/'
        dim = 300
        vocab, vecs = vecops.read_swivel_vecs(
            base + 'vecs.bin',
            base + 'vocab.txt',
            dim)
        self.assertIsNotNone(vocab)
        self.assertIsNotNone(vecs)
        self.assertEqual(5632, len(vocab))
        self.assertEqual((5632, 300), vecs.shape)


if __name__ == '__main__':
    unittest.main()
