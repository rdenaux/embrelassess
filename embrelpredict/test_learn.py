from context import learn
import unittest
import os.path as osp
import torch


class MyTest(unittest.TestCase):

    def test__epochs_from_trainset_size(self):
        self.assertEquals(
            learn._epochs_from_trainset_size(20), 1)
        self.assertEquals(
            learn._epochs_from_trainset_size(200), 1)
        self.assertEquals(
            learn._epochs_from_trainset_size(2000), 1)
        self.assertEquals(
            learn._epochs_from_trainset_size(20000), 1)
        self.assertEquals(
            learn._epochs_from_trainset_size(200000), 1)

    def test__build_model(self):
        self.assertIsInstance(learn._build_model('nn1'), torch.Module)
        self.assertIsInstance(learn._build_model('nn2'), torch.Module)
        self.assertIsInstance(learn._build_model('nn3'), torch.Module)
        self.assertIsInstance(learn._build_model('logreg'), torch.Module)
        self.assertIsInstance(learn._build_model('alwaysT'), torch.Module)
        self.assertIsInstance(learn._build_model('alwaysF'), torch.Module)
        with self.assertRaises(Exception):
            learn._build_model('other')
            learn._build_model()

    def test_learn_rel(self):
        relpath = osp.join('data', 'rels')
        # rel_df_row =  # load df from relpath and select one
        # data_loaders = { 'vecsi5k': }  # load test (random?) vector loader
        # learnrel_result = learn.learn_rel(relpath, rel_df_row, data_loaders)


if __name__ == '__main__':
    unittest.main()
