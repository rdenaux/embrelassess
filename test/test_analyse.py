import embrelpredict.learn as learn
import embrelpredict.analyse as analyse
import os.path as osp

from .common.embrelpred_test_case import EmbrelpredTestCase

exp_agg_fields = ['model', 'datapoints']
for metric in ['acc', 'f1', 'precision', 'recall']:
    for agg_val in ['avg', 'std', 'min', 'max']:
        exp_agg_fields.append('%s_%s' % (metric, agg_val))


class AnalyseTest(EmbrelpredTestCase):
    def setUp(self):
        EmbrelpredTestCase.setUp(self)
        self.learn_result = learn.load_learn_result(
            osp.join('test', 'data',
                     'learn-ress', '20180123-155938'),
            'lem2lem', 'cause')

    def test_aggregate_runs(self):
        lr_aggs = analyse.aggregate_runs(self.learn_result)
        # 2 embs * 2 models * 2 result_types
        self.assertEqual(8, len(lr_aggs))
        exp_lr_agg_fields = exp_agg_fields + ['rel_type', 'rel_name',
                                              'emb', 'result_type']
        self.assertSetEqual(set(exp_lr_agg_fields), set(lr_aggs[0]))


class EmbeddingModelResultsTest(EmbrelpredTestCase):

    def setUp(self):
        EmbrelpredTestCase.setUp(self)
        learn_result = learn.load_learn_result(
            osp.join('test', 'data',
                     'learn-ress', '20180123-155938'),
            'lem2lem', 'cause')
        basic_results = learn_result['emb_model_results']['kcap17_basic']
        self.emress = analyse.EmbeddingModelResults(basic_results)

    def test_models(self):
        self.assertSetEqual(set(['nn2', 'nn3']), self.emress.models())

    def test_calc_test_aggregates(self):
        test_aggs = self.emress.calc_test_aggregates()
        self.assertEqual(2, len(test_aggs))

        test_agg = None
        for ta in test_aggs:
            if ta['model'] == 'nn3':
                test_agg = ta

        self.assertSetEqual(set(exp_agg_fields), set(test_agg))
        self.assertEqual(4, test_agg['datapoints'])
        self.assertEqual('nn3', test_agg['model'])
        # self.assertDictEqual({}, test_agg)
        self.assertAlmostEqual(0.575, test_agg['acc_avg'], places=3)
        self.assertAlmostEqual(0.051, test_agg['acc_std'], places=3)
        self.assertAlmostEqual(0.630, test_agg['acc_max'], places=3)
        self.assertAlmostEqual(0.493, test_agg['acc_min'], places=3)
        self.assertAlmostEqual(0.553, test_agg['f1_avg'], places=3)
        self.assertAlmostEqual(0.573, test_agg['precision_avg'], places=3)
        self.assertAlmostEqual(0.55, test_agg['recall_avg'], places=3)

    def test_calc_randpredict_aggregates(self):
        rand_aggs = self.emress.calc_randpredict_aggregates()
        self.assertEqual(2, len(rand_aggs))

        rand_agg = None
        agg_models = []
        for ta in rand_aggs:
            agg_models.append(ta['model'])
            if ta['model'] == 'nn3':
                rand_agg = ta

        self.assertSetEqual(set(['nn2', 'nn3']), set(agg_models))
        self.assertSetEqual(set(exp_agg_fields), set(rand_agg))
        self.assertEqual(4, rand_agg['datapoints'])
        # self.assertDictEqual({}, rand_agg)
        self.assertEqual('nn3', rand_agg['model'])
        self.assertAlmostEqual(0.466, rand_agg['acc_avg'], places=3)
        self.assertAlmostEqual(0.051, rand_agg['acc_std'], places=3)
        self.assertAlmostEqual(0.521, rand_agg['acc_max'], places=3)
        self.assertAlmostEqual(0.384, rand_agg['acc_min'], places=3)
        self.assertAlmostEqual(0.451, rand_agg['f1_avg'], places=3)
        self.assertAlmostEqual(0.444, rand_agg['precision_avg'], places=3)
        self.assertAlmostEqual(0.457, rand_agg['recall_avg'], places=3)

    def test_calc_pretrain_aggregates(self):
        aggs = self.emress.calc_pretrain_aggregates()
        self.assertEqual(2, len(aggs))

        agg = None
        agg_models = []
        for ta in aggs:
            agg_models.append(ta['model'])
            if ta['model'] == 'nn3':
                agg = ta

        self.assertSetEqual(set(['nn2', 'nn3']), set(agg_models))
        self.assertSetEqual(set(exp_agg_fields), set(agg))
        self.assertEqual(4, agg['datapoints'])
        # self.assertDictEqual({}, agg)
        self.assertEqual('nn3', agg['model'])
        self.assertAlmostEqual(0.432, agg['acc_avg'], places=3)
        self.assertAlmostEqual(0.041, agg['acc_std'], places=3)
        self.assertAlmostEqual(0.479, agg['acc_max'], places=3)
        self.assertAlmostEqual(0.370, agg['acc_min'], places=3)
        self.assertAlmostEqual(0.537, agg['f1_avg'], places=3)
        self.assertAlmostEqual(0.439, agg['precision_avg'], places=3)
        self.assertAlmostEqual(0.714, agg['recall_avg'], places=3)
