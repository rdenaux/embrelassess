import matplotlib.pyplot as plt
import seaborn
import numpy as np

# The 'analyse' module provides methods for analysing embedding relation
# learn_results


def expand_test_result_df(df_test):
    """Adds columns to a DataFrame with test results

    Args:
    df_test DataFrame as produced by trainer.ModelTrainer, i.e. with columns
      'tp', 'fp', 'fn', 'correct', 'total_examples' and 'examples_above_threshold'

    Returns:
    the input DataFrame with additional columns for 'precision', 'recall',
      'acc'uracy, 'f1' measure and 'coverage' percentage.
    """
    df = df_test
    df['precision'] = df['tp'] / (df['tp'] + df['fp'])
    df['recall'] = df['tp'] / (df['tp'] + df['fn'])
    df['acc'] = df['correct'] / df['examples_above_threshold']
    df['f1'] = 2*df['tp'] / (2*df['tp'] + df['fp'] + df['fn'])
    df['coverage'] = df['examples_above_threshold']/df['total_examples']
    return df


def aggregate_runs(learn_results):
    rel_name = learn_results['rel_name']
    rel_type = learn_results['rel_type']
    emb_model_results = learn_results['emb_model_results']
    result = []
    for emb in emb_model_results:
        emb_results = emb_model_results[emb]
        emress = EmbeddingModelResults(emb_results)
        basic_agg = {
            'rel_type': rel_type,
            'rel_name': rel_name,
            'emb': emb,
        }
        for test_agg in emress.calc_test_aggregates():
            ba = {**basic_agg, **test_agg}
            ba['result_type'] = 'test'
            result.append(ba)

        for rand_agg in emress.calc_randpredict_aggregates():
            ra = {**basic_agg, **rand_agg}
            ra['result_type'] = 'random'
            result.append(ra)
    return result


class EmbeddingModelResults():
    def __init__(self, modress):
        """Provides methods for aggregating embedding model results

        Args:
        modress a list of embedding-model results
        """
        self.modress = modress

    def calc_test_aggregates(self):
        return self._calc_aggregates(self._test_val)

    def calc_randpredict_aggregates(self):
        return self._calc_aggregates(self._rand_val)

    def calc_pretrain_aggregates(self):
        return self._calc_aggregates(self._pretrain_val)

    def models(self):
        return set(self.extract_vals(self._res_val('model')))

    def _calc_aggregates(self, metric_modres_to_val_fn):
        result = []
        for model in self.models():
            agg = {}
            n = None
            for metric in ['acc', 'f1', 'precision', 'recall']:
                metrics = self.extract_vals(metric_modres_to_val_fn(metric))
                if not n:
                    n = len(metrics)
                else:
                    assert(n == len(metrics))
                agg['%s_avg' % metric] = np.mean(metrics)
                agg['%s_std' % metric] = np.std(metrics)
                agg['%s_min' % metric] = np.min(metrics)
                agg['%s_max' % metric] = np.max(metrics)
            agg['model'] = model
            agg['datapoints'] = n
            result.append(agg)
        return result

    def _testres(self, modres):
        # plotter.expand needed to calculate 'acc', 'f1', etc. from
        # raw tp, fp, fn, correct counts
        # better to perform this as a separate step somewhere else...
        df = modres['test_df']
        no_threshold = df[df['threshold'] == 0.0]  # df.loc[0]
        # print('no_threshold_testres:', no_threshold)
        return expand_test_result_df(no_threshold)

    def _randres(self, modres):
        return expand_test_result_df(modres['test_random_result'])

    def _pretrain_res(self, modres):
        return expand_test_result_df(modres['pretrain_test_result'])

    def _test_val(self, key):
        return lambda modres: self._testres(modres)[key]

    def _rand_val(self, key):
        return lambda modres: self._randres(modres)[key]

    def _pretrain_val(self, key):
        return lambda modres: self._pretrain_res(modres)[key]

    def _res_val(self, key):
        return lambda modres: modres[key]

    def extract_vals(self, value_extractor):
        """returns a list of values for a given list of model results"""
        result = []
        for model_result in self.modress:
            result.append(value_extractor(model_result))
        return result

    def extract_val(self, value_extractor):
        vals = set(self.extract_vals(value_extractor))
        if len(vals) == 1:
            return vals.pop()
        elif len(vals) == 0:
            raise Exception('No value using ' + value_extractor)
        else:
            print('Multiple values for ' + value_extractor)
            return vals.pop()


def summarise_rel_models(learn_rel_results, plotter):
    """Summarises the data gathered while learning a relation

    Since the learn_rel_results contains multiple runs/model results for the
    relation, this method provides a method for aggregating the results of
    the different runs. Allowing us to calculate the mean and stdev metrics.
    Also, since various models may have been tried in learn_rel, this
    method performs a selection to choose to best performing model.

    Args:
      learn_rel_results object as output by method learn_rel
      plotter object of type embrelpred.analyse.Plotter

    Returns:
      dictionary summarising the best model and the base model
    """
    rel_name = learn_rel_results['rel_name']
    rel_type = learn_rel_results['rel_type']
    pos_exs = learn_rel_results['pos_exs']
    empty_result = {
        "rel_name": rel_name, "rel_type": rel_type, "epochs": 0,
        "best_acc": 0, "best_f1": 0, "best_prec": 0, "best_rec": 0,
        "base_acc": 0.5, "base_f1": 0.5, "base_prec": 0.5, "base_rec": 0.5,
        "best_model": "None", "best_model_type": "None",
        "pos_exs": pos_exs}

    def get_testres(model_result):
        # plotter.expand needed to calculate 'acc', 'f1', etc. from
        # raw tp, fp, fn, correct counts
        # better to perform this as a separate step somewhere else...
        return expand_test_result_df(model_result['test_df'].loc[0])

    def get_randres(model_result):
        return model_result['test_random_result']

    def get_test_accuracy(model_result):
        return get_testres(model_result)['acc']

    def get_test_f1(model_result):
        return get_testres(model_result)['f1']

    def get_test_precision(model_result):
        return get_testres(model_result)['precision']

    def get_test_recall(model_result):
        return get_testres(model_result)['recall']

    def get_base_accuracy(model_result):
        return get_randres(model_result)['acc']

    def get_base_f1(model_results):
        return get_randres(model_results)['f1']

    def get_base_prec(model_results):
        return get_randres(model_results)['precision']

    def get_base_rec(model_results):
        return get_randres(model_results)['recall']

    def get_model(model_result):
        return model_result['model']

    def extract_vals(model_results, value_extractor):
        """returns a list of values for a given list of model results"""
        result = []
        for model_result in model_results:
            result.append(value_extractor(model_result))
        return result

    def extract_val(model_results, value_extractor):
        result = None
        for model_result in model_results:
            result = value_extractor(model_result)
        return result

    winner_model = None  # winner agg_model_results
    emb_agg_results = {}
    emb_model_results = learn_rel_results['emb_model_results']
    for emb_name in learn_rel_results:
        model_results = emb_model_results[emb_name]
        model = extract_val(model_results, get_model)
        test_accs = extract_vals(model_results, get_test_accuracy)
        test_f1s = extract_vals(model_results, get_test_f1)
        test_precs = extract_vals(model_results, get_test_precision)
        test_recs = extract_vals(model_results, get_test_recall)
        ba_accs = extract_vals(model_results, get_base_accuracy)
        ba_f1s = extract_vals(model_results, get_base_f1)
        ba_precs = extract_vals(model_results, get_base_prec)
        ba_recs = extract_vals(model_results, get_base_rec)
        agg_model_results = {
            "model": model,
            "avg_acc": np.mean(test_accs),      "std_acc": np.std(test_accs),
            "avg_f1": np.mean(test_f1s),        "std_f1": np.std(test_f1s),
            "avg_prec": np.mean(test_precs),    "std_prec": np.std(test_precs),
            "avg_rec": np.mean(test_recs),      "std_rec": np.std(test_recs),
            "base_avg_acc": np.mean(ba_accs), "base_std_acc": np.std(ba_accs),
            "base_avg_f1": np.mean(ba_f1s),   "base_std_f1": np.std(ba_f1s),
            "base_avg_prec": np.mean(ba_precs),
            "base_std_prec": np.std(ba_precs),
            "base_avg_rec": np.mean(ba_recs), "base_std_rec": np.std(ba_recs),
            "results": model_results}
        agg_models_results = emb_agg_results.get(emb_name, [])
        agg_models_results.append(agg_model_results)
        emb_agg_results[emb_name] = agg_models_results
        print(
            '%s acc %.3f+-%.3f f1 %.3f+-%.3f prec %.3f+-%.3f rec %.3f+-%.3f' %
            (
                model,
                agg_model_results['avg_acc'], agg_model_results['std_acc'],
                agg_model_results['avg_f1'], agg_model_results['std_f1'],
                agg_model_results['avg_prec'], agg_model_results['std_prec'],
                agg_model_results['avg_rec'], agg_model_results['std_rec']
            )
        )

        def model_summary(model, name=None):
            if not name:
                name = model['model']
            return '%s (avg_acc %.2f, avg_f1 %.2f)' % (
                name, model['avg_acc'], model['avg_f1'])

        if not winner_model:
            winner_model = agg_model_results
        elif winner_model['avg_acc'] > agg_model_results['avg_acc']:
            print('Previous model %s is, on average, better than %s' %
                  (model_summary(winner_model),
                   model_summary(agg_model_results, name=model)))
        else:
            print('Previous model %s was, on average, worse than %s' %
                  (model_summary(winner_model),
                   model_summary(agg_model_results, name=model)))
            winner_model = agg_model_results

    if not winner_model:
        return empty_result

    def select_best_result(winner_model):
        result = None
        for model_result in winner_model['results']:
            if not result:
                result = model_result
            else:
                if get_test_accuracy(result) > get_test_accuracy(model_result):
                    result = result
                else:
                    result = model_result
        return result

    best_result = select_best_result(winner_model)

    # only place where plotter is used, maybe better to separate
    # plotting from aggregation of data
    plt = plotter.plot_learning(winner_model['results']['trainer_df'],
                                best_result['test_df'],
                                winner_model['results']['model'])
    plt.show()

    base_test_result = winner_model['test_random_result']
    row = best_result['test_df'].loc[0]
    result = {"rel_name": rel_name, "rel_type": rel_type, "epochs": epochs,
              "best_acc": row['acc'], "best_f1": row['f1'],
              "best_prec": row['precision'], "best_rec": row['recall'],
              "base_acc": winner_trainer.acc(base_test_result),
              "base_f1": winner_trainer.f1(base_test_result),
              "base_prec": winner_trainer.precision(base_test_result),
              "base_rec": winner_trainer.recall(base_test_result),
              "best_model": winner_model['model'], "best_model_type": winner_model['model'],
              "pos_exs": cnt}

    for agg_model_results in agg_models_results:
        model = agg_model_results['model']
        result['%s_avg_acc' % model] = agg_model_results['avg_acc']
        result['%s_std_acc' % model] = agg_model_results['std_acc']
        result['%s_avg_f1' % model] = agg_model_results['avg_f1']
        result['%s_std_f1' % model] = agg_model_results['std_f1']

    return result


class Plotter():
    def __init__(self):
        print('Plotter')
        self.colors = seaborn.color_palette()

    def plot_learning_curve(self, df_training, model_name):
        df = df_training
        row_min = df.min()
        row_max = df.max()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

        plt.plot(df['step'], df['train/loss'], '-',
                 markersize=1, color=self.colors[0], alpha=.5,
                 label='train loss')
        plt.plot(df['step'], df['valid/loss'], '-',
                 markersize=1, color=self.colors[1], alpha=.5,
                 label='valid loss')
        plt.xlim((0, row_max['step']))
        plt.ylim((min(row_min['train/loss'], row_min['valid/loss']),
                  max(row_max['train/loss'], row_max['valid/loss'])))
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('learning curve %s' % model_name)
        plt.legend()

    def plot_valid_acc(self, df_training, model_name):
        df = df_training
        # row_min = df.min()
        row_max = df.max()

        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df['step'], df['valid/precision'], '-',
                 markersize=1, color=self.colors[0], alpha=.5,
                 label='precision')
        plt.plot(df['step'], df['valid/recall'], '-',
                 markersize=1, color=self.colors[1], alpha=.5,
                 label='recall')
        plt.plot(df['step'], df['valid/acc'], '-',
                 markersize=1, color=self.colors[2], alpha=.5,
                 label='accuracy')
        plt.plot(df['step'], df['valid/f1'], '-',
                 markersize=1, color=self.colors[3], alpha=.5,
                 label='f1')
        plt.xlim((0, row_max['step']))
        plt.ylim(0.0, 1.0)
        plt.xlabel('step')
        plt.ylabel('percent')
        plt.legend()
        plt.title('Validation results %s ' % model_name)

    def plot_test_df(self, df_test, model_name):
        df = expand_test_result_df(df_test)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.plot(df['threshold'], df['precision'], '-',
                 markersize=1, color=self.colors[0], alpha=.5,
                 label='precision')
        plt.plot(df['threshold'], df['recall'], '-',
                 markersize=1, color=self.colors[1], alpha=.5,
                 label='recall')
        plt.plot(df['threshold'], df['acc'], '-',
                 markersize=1, color=self.colors[2], alpha=.5,
                 label='accuracy')
        plt.plot(df['threshold'], df['f1'], '-',
                 markersize=1, color=self.colors[4], alpha=.5,
                 label='f1')
        plt.plot(df['threshold'], df['coverage'], '-',
                 markersize=1, color=self.colors[3], alpha=.5,
                 label='coverage')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.05)
        plt.xlabel('threshold')
        plt.ylabel('percent')
        plt.legend()
        plt.title('Test results %s ' % model_name)

    def plot_learning(self, df_training, df_test, model_name,
                      n_row=2, n_col=2, figsize=(10, 6), dpi=300):
        plt.figure(figsize=figsize, dpi=dpi)

        # learning curve
        plt.subplot(n_row, n_col, 1)
        self.plot_learning_curve(df_training, model_name)

        # validation p-r-acc
        plt.subplot(n_row, n_col, 2)
        self.plot_valid_acc(df_training, model_name)

        # test p-r-acc
        plt.subplot(n_row, n_col, 3)
        self.plot_test_df(df_test, model_name)

        fig = plt.gcf()
        fig.tight_layout()

        return plt
