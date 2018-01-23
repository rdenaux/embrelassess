import embrelpredict.learn as learn
import embrelpredict.embloader as embloader
import unittest
import os.path as osp
import torch


class LearnTest(unittest.TestCase):

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
        n_runs = 3
        learn_result = learn.learn_rel(relpath, rel_meta,
                                       data_loaders,
                                       models=['nn2'],
                                       n_runs=n_runs)

        self.assertListEqual(['rel_name', 'rel_type', 'pos_exs', 'emb_model_results'],
                             list(learn_result))
        self.assertEqual('lem2lem', learn_result['rel_type'])
        self.assertEqual('cause', learn_result['rel_name'])
        self.assertEqual(719, learn_result['pos_exs'])
        self.assertListEqual(['kcap17_basic', 'kcap17_wsd'],
                             list(learn_result['emb_model_results']))
        basic_results = learn_result['emb_model_results']['kcap17_basic']
        print('basic_results', type('basic_results'))
        self.assertEqual(n_runs, len(basic_results))
        basic_result = basic_results[0]
        self.assertListEqual(['model', 'i', 'emb', 'epochs', 'pos_exs', 'trainer_df',
                              'pretrain_test_result', 'test_df', 'test_random_result'],
                             list(basic_result))
        self.assertEqual('nn2', basic_result['model'])
        self.assertEqual(0, basic_result['i'])
        self.assertEqual('kcap17_basic', basic_result['emb'])
        self.assertEqual(24, basic_result['epochs'])
        self.assertEqual(719, basic_result['pos_exs'])
        trainer_df = basic_result['trainer_df']
        self.assertListEqual(['epoch', 'step', 'train/loss', 'valid/loss',
                              'valid/precision', 'valid/recall', 'valid/acc', 'valid/f1'],
                             list(trainer_df.columns))
        self.assertEqual(48, len(trainer_df))
        pretrain_test_result = basic_result['pretrain_test_result']
        # model initialization is random, so values differ, cannot use assertDictEqual
        test_fields = ['model', 'total_examples', 'threshold',
                       'examples_above_threshold', 'correct', 'tp', 'fp', 'fn']
        self.assertListEqual(test_fields,
                             list(pretrain_test_result))
        test_df = basic_result['test_df']
        self.assertEqual(20, len(test_df))
        self.assertListEqual(test_fields,
                             list(test_df.columns))
        # random results differ per run, cannot use assertDictEqual
        self.assertListEqual(test_fields,
                             list(basic_result['test_random_result']))


if __name__ == '__main__':
    unittest.main()
