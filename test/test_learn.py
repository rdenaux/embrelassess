import embrelpredict.learn as learn
import embrelpredict.embloader as embloader
import unittest
import os
import os.path as osp
import torch
import datetime
import filecmp
import difflib

from .common.test_markers import slow
from .common.embrelpred_test_case import EmbrelpredTestCase


class LearnTest(EmbrelpredTestCase):

    def test__epochs_from_trainset_size(self):
        self.assertEquals(
            learn._epochs_from_trainset_size(20), 96)
        self.assertEquals(
            learn._epochs_from_trainset_size(200), 48)
        self.assertEquals(
            learn._epochs_from_trainset_size(2000), 24)
        self.assertEquals(
            learn._epochs_from_trainset_size(20000), 12)
        self.assertEquals(
            learn._epochs_from_trainset_size(200000), 6)

    def test__build_model(self):
        print(learn._build_model('nn1'))
        self.assertIsInstance(learn._build_model('nn1'), torch.nn.Module)
        self.assertIsInstance(learn._build_model('nn2'), torch.nn.Module)
        self.assertIsInstance(learn._build_model('nn3'), torch.nn.Module)
        self.assertIsInstance(learn._build_model(), torch.nn.Module)
        self.assertIsInstance(learn._build_model('logreg'), torch.nn.Module)
        self.assertIsInstance(learn._build_model('alwaysT'), torch.nn.Module)
        self.assertIsInstance(learn._build_model('alwaysF'), torch.nn.Module)
        with self.assertRaises(Exception):
            learn._build_model('other')

    def test_load_rels_meta(self):
        relpath = osp.join('test', 'data', 'wnet-rels')
        rel_df = learn.load_rels_meta(relpath)
        self.assertIsNotNone(rel_df)
        self.assertEqual(3, len(rel_df))
        self.assertListEqual(['cnt', 'file', 'name', 'type'], list(rel_df.columns))
        self.assertListEqual(['lem2lem', 'lem2lem', 'lem2lem'],
                             list(rel_df['type']))
        self.assertListEqual([719, 81, 369], list(rel_df['cnt']))
        self.assertListEqual(['cause', 'participle_of', 'substance_meronym'],
                             list(rel_df['name']))
        self.assertListEqual(['lem2lem_cause__719.txt',
                              'lem2lem_participle_of__81.txt',
                              'lem2lem_substance_meronym__369.txt'],
                             list(rel_df['file']))

    @slow
    def test_learn_rel(self):
        def vecpath_to_loader(vecpath, dim=300):
            vecs = embloader.SwivelAsTorchTextVector(vecpath+'vecs.bin',
                                                     vecpath+'vocab.txt', dim)
            return embloader.VecPairLoader(vecs)

        relpath = osp.join('test', 'data', 'wnet-rels')
        basic_path = 'test/data/kcap-basic-vec/'
        wsd_path = 'test/data/kcap-wsd-vec/'
        data_loaders = {'kcap17_basic': vecpath_to_loader(basic_path),
                        'kcap17_wsd': vecpath_to_loader(wsd_path)}

        rels_meta = learn.load_rels_meta(relpath)
        rel_meta = rels_meta.loc[0]
        n_runs = 2
        epochs = 2
        models = ['nn2', 'nn3']
        learn_result = learn.learn_rel(relpath, rel_meta,
                                       data_loaders,
                                       models=models,
                                       epochs_from_trainset_size_fn=lambda x: epochs,
                                       n_runs=n_runs)

        # Uncomment to generate learn_ress folder that can be used in io tests
        # odir = osp.join(relpath, os.pardir, 'learn-ress',
        #                datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        # learn.store_learn_result(odir, learn_result)

        self._assert_expected_learn_result(learn_result, n_runs, models, epochs)

    def test_load_learn_result(self):
        learn_result = learn.load_learn_result(osp.join('test', 'data',
                                                        'learn-ress', '20180123-155938'),
                                               'lem2lem', 'cause')
        self._assert_expected_learn_result(learn_result, n_runs=2,
                                           models=['nn2', 'nn3'], epochs=5)

    def test_roundtrip_io_learn_result(self):
        idir = osp.join('test', 'data', 'learn-ress', '20180123-155938')
        learn_result = learn.load_learn_result(idir, 'lem2lem', 'cause')
        odir = osp.join(self.test_dir, 'test-rountrip-io-lr')
        learn.store_learn_result(odir, learn_result)
        dcmp = filecmp.dircmp(idir, odir)
        self._assert_same_dircmp(dcmp)

    def _assert_same_dircmp(self, dcmp):
        self.assertEqual(0, len(dcmp.left_only),
                         '%s only on %s' % (dcmp.left_only, dcmp.left))
        self.assertEqual(0, len(dcmp.right_only),
                         '%s only on %s' % (dcmp.right_only, dcmp.right))
        if len(dcmp.diff_files) > 0:
            # differences can occur due to float rounding, keep in check
            self._assert_diff_similar(dcmp, min_ratio=0.8)
        for subdcmp in dcmp.subdirs:
            self._assert_same_dircmp(dcmp.subdirs[subdcmp])

    def _assert_diff_similar(self, dcmp, min_ratio=0.8):
        for name in dcmp.diff_files:
            with open(osp.join(dcmp.left, name), "r") as file:
                left = file.readlines()
            with open(osp.join(dcmp.right, name), "r") as file:
                right = file.readlines()
            matcher = difflib.SequenceMatcher(None, left, right)
            self.assertGreater(matcher.ratio(), min_ratio,
                               "File %s in \n%s\nand\n%s are not even similar %.3f" %
                               (name, dcmp.left, dcmp.right, matcher.ratio()))

    def _assert_expected_learn_result(self, learn_result, n_runs, models, epochs):
        self.assertSetEqual(set(['rel_name', 'rel_type', 'pos_exs', 'emb_model_results']),
                            set(learn_result))
        self.assertEqual('lem2lem', learn_result['rel_type'])
        self.assertEqual('cause', learn_result['rel_name'])
        self.assertEqual(719, learn_result['pos_exs'])
        self.assertListEqual(['kcap17_basic', 'kcap17_wsd'],
                             list(learn_result['emb_model_results']))
        basic_results = learn_result['emb_model_results']['kcap17_basic']
        self.assertEqual(int(n_runs * len(models)), len(basic_results))
        basic_result = basic_results[0]
        self.assertSetEqual(set(['model', 'i', 'emb', 'epochs', 'pos_exs',
                                 'dataset_size', 'dataset_tok_cnt', 'dataset_tok_found',
                                 'trainer_df', 'pretrain_test_result', 'test_df',
                                 'test_random_result']),
                            set(basic_result))
        self.assertEqual('nn2', basic_result['model'])
        self.assertEqual(0, basic_result['i'])
        self.assertEqual('kcap17_basic', basic_result['emb'])
        self.assertEqual(epochs, basic_result['epochs'])
        self.assertEqual(719, basic_result['pos_exs'])
        trainer_df = basic_result['trainer_df']
        self.assertSetEqual(set(['epoch', 'step', 'train/loss', 'valid/loss',
                                 'valid/precision', 'valid/recall',
                                 'valid/acc', 'valid/f1']),
                            set(trainer_df.columns))
        self.assertEqual(epochs * 2, len(trainer_df))
        pretrain_test_result = basic_result['pretrain_test_result']
        # model initialization is random, so values differ, cannot use assertDictEqual
        test_fields = ['model', 'total_examples', 'threshold',
                       'examples_above_threshold', 'correct', 'tp', 'fp', 'fn']
        self.assertSetEqual(set(test_fields),
                            set(pretrain_test_result))
        test_df = basic_result['test_df']
        self.assertEqual(20, len(test_df))
        self.assertSetEqual(set(test_fields),
                            set(test_df.columns))
        # random results differ per run, cannot use assertDictEqual
        self.assertSetEqual(set(test_fields),
                            set(basic_result['test_random_result']))


if __name__ == '__main__':
    unittest.main()
