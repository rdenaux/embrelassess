import torch
import eval_classification as eval
import numpy as np
import sys
import logging
import os.path as osp
import math
import itertools

logger = logging.getLogger(__name__)


def _epochs_from_trainset_size(trainset_size):
    """Returns a sensible number of epochs for a given trainset_size

    Args:
       trainset_size the size of the training set (i.e. examples in 1 epoch)

    Returns:
       integer number of suggested epochs to train for
    """
    log10 = math.log10(trainset_size/2)
    order = round(log10 - 1)
    inv_order = 5 - order
    factor = math.pow(2, inv_order)
    base = 3
    return round(factor * base)


def _build_model(name='nn3', indim=600):
    """Build a pytorch binary classifier for a given input dimension

    Args:
        name one of 'nn1', 'nn2', 'nn3', 'logreg', 'alwaysT', 'alwaysF'

    Returns:
      a pytorch binary classifier model
    """
    if name == 'logreg':
        my_model = eval.LogisticRegression(indim)
    elif name == 'nn1':
        nn1 = {"layer_dims": [indim], "dropouts": [0.5]}
        my_model = eval.NNBiClassifier(indim, nn1['layer_dims'],
                                       nn1['dropouts'])
    elif name == 'nn2':
        if indim == 600:
            nn2 = {"layer_dims": [750, 400], "dropouts": [0.5, 0.5]}
        elif indim == 300:
            nn2 = {"layer_dims": [400, 150], "dropouts": [0.5, 0.5]}
        else:
            raise Exception('Unexpected input dimension %d' % indim)
        my_model = eval.NNBiClassifier(indim, nn2['layer_dims'],
                                       nn2['dropouts'])
    elif name == 'nn3':
        if indim == 600:
            nn3 = {"layer_dims": [750, 500, 250],
                   "dropouts": [0.5, 0.5, 0.5]}
        elif indim == 300:
            nn3 = {"layer_dims": [400, 200, 100],
                   "dropouts": [0.5, 0.5, 0.5]}
        else:
            raise Exception('Unexpected input dimension %d' % indim)
        my_model = eval.NNBiClassifier(indim, nn3['layer_dims'],
                                       nn3['dropouts'])
    elif name == 'alwaysT':
        my_model = eval.DummyBiClassifier(indim, predef=[0.01, 0.99])
    elif name == 'alwaysF':
        my_model = eval.DummyBiClassifier(indim, predef=[0.99, 0.01])
    else:
        raise Exception('Unknown model name %d' % name)
    return my_model


def learn_rel(relpath, rel_df_row, data_loaders,
              single_rel_types=[],
              epochs_from_trainset_size_fn=_epochs_from_trainset_size,
              rel_filter=None, models=['logreg', 'nn2', 'nn3'], n_runs=5,
              train_input_disturber=None,
              debug_test_df=False,
              cuda=False):
    """ Train binary classifier models to learn a relation given a dataset

    Args:
       relpath path to the relation tsv files
       rel_df_row dataframe row with metadata about the relation to learn
       data_loaders dictionary of data_loader objects responsible for loading
                    and splitting the dataset
       single_rel_types list of rel type names which are not pairs, but
                    single words
       epochs_from_trainset_size_fn function from trainset size to number
                    of epochs
       rel_filter filter for the rel_df_row to skip unwanted relations
       models list of model names to train
       n_runs times to train each model (to get average and stdv)
       train_input_disturber function to disturb an input batch

    Returns:
      An object with data summarising the learning result. It includes the
      rel_name, rel_type, number of epochs trained, number of positive samples
      for the relation and metrics for various models: base, best, and the
      specified models. Metrics include (average and stdv for) accuracy, f1,
      precision and recall.
    """
    # models = ['logreg', 'nn1', 'nn2', 'nn3']
    # print("\n\n\n", rel_df_row['file'])
    cnt = rel_df_row['cnt']
    rel_name = rel_df_row['name']
    rel_type = rel_df_row['type']
    empty_result = {"rel_name": rel_name, "rel_type": rel_type,
                    "pos_exs": cnt,
                    "emb_model_results": {}}
    if cnt < 75:
        print(rel_name, rel_type, 'too few examples')
        return empty_result

    if rel_filter and not rel_filter(rel_df_row):
        print(rel_name, rel_type, 'not in rel_name filter')
        return empty_result

    emb_model_results = {}  # dict from 'emb name' to a list of model_results
    print('Training each model %d times...' % n_runs)
    for model, loader_name, run in itertools.product(
            models, data_loaders, range(n_runs)):
        print("run %d on model %s with vectors %s" %
              (run, model, loader_name))
        data_loader = data_loaders[loader_name]
        fpath = osp.join(relpath, rel_df_row['file'])
        # load embeddings and labels
        if rel_type == 'rnd2rnd':
            X, Y = data_loader.generate_random_pair_data(target_size=cnt*2)
        elif rel_type in single_rel_types:
            X, Y = data_loader.load_single_data(fpath)
        else:
            X, Y = data_loader.load_pair_data(fpath)

        indim = X.shape[1]

        msg = 'Expecting binary classifier but found max %d min %d' % (
            torch.max(Y), torch.min(Y))
        assert torch.max(Y) == 1 and torch.min(Y) == 0, msg

        print("\n\n\n", rel_df_row['file'])
        epochs = epochs_from_trainset_size_fn(X.shape[0])  # from full dataset
        trainloader, validloader, testloader = data_loader.split_data(
            X, Y, seed=41)
        my_model = _build_model(model, indim)

        try:
            trainer = eval.ModelTrainer(my_model, cuda=cuda)
            pretrain_test_result = trainer.test(testloader)
            trainer.train(trainloader, validloader, epochs=epochs,
                          input_disturber=train_input_disturber)
            test_df = trainer.test_df(testloader, debug=debug_test_df)

            test_random_result = trainer.test_random(testloader)
            model_result = {"model": model, "i": run, "emb": loader_name,
                            "epochs": epochs, "pos_exs": cnt,
                            # "trainer": trainer,
                            "trainer_df": trainer.df,  # to plot learning curve
                            "pretrain_test_result": pretrain_test_result,
                            "test_df": test_df,
                            "test_random_result": test_random_result}
            model_results = emb_model_results.get(loader_name, [])
            model_results.append(model_result)
            emb_model_results[loader_name] = model_results
        except:
            print("Unexpected error executing %s:" % model, sys.exc_info()[0])
            raise

        # del trainer # cannot delete the trainer as we need it later on
        del my_model
        del trainloader
        del validloader
        del testloader

    # TODO: write test results to csv files and model checkpoints?
    result = {"rel_name": rel_name, "rel_type": rel_type,
              "pos_exs": cnt,
              "emb_model_results": emb_model_results}
    return result


def summarise_rel_models(learn_rel_results, plotter):
    """Summarises the data gathered while learning a relation

    Since the learn_rel_results contains multiple runs/model results for the
    relation, this method provides a method for aggregating the results of
    the different runs. Allowing us to calculate the mean and stdev metrics.
    Also, since various models may have been tried in learn_rel, this
    method performs a selection to choose to best performing model.

    Args:
      learn_rel_results object as output by method learn_rel
      plotter object of type eval_classification.Plotter

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
        return plotter.expand(model_result['test_df'].loc[0])

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
        "best_acc": row['acc'], "best_f1": row['f1'], "best_prec": row['precision'], "best_rec": row['recall'],
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
        
    del trainer
    del my_model
    del trainloader
    del validloader
    del testloader
    return result
        
