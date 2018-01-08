import torch
import eval_classification as eval
import numpy as np
import sys

def learn_rel(relpath, rel_df_row, plotter, data_loader,
              single_rel_types=[],
              rel_filter=None, models=['logreg', 'nn2', 'nn3'], n_crossval=5,
              train_input_disturber=None):
    """ Train binary classifier models to learn a relation given a dataset
    
    Args:
       relpath path to the relation tsv files
       rel_df_row dataframe row with metadata about the relation to learn
       plotter Plotter object that generates summary plots for models
       data_loader object responsible for loading and splitting the dataset
       single_rel_types list of rel type names which are not pairs, but single words
       rel_filter filter for the rel_df_row to skip unwanted relations
       models list of model names to train 
       n_crossval times to train each model (to get average and stdv)
       train_input_disturber function to disturb an input batch 

    Returns:
      An object with data summarising the learning result. It includes the
      rel_name, rel_type, number of epochs trained, number of positive samples 
      for the relation and metrics for various models: base, best, and the
      specified models. Metrics include (average and stdv for) accuracy, f1,
      precision and recall.
    """
    # models = ['logreg', 'nn1', 'nn2', 'nn3']
    #print("\n\n\n", rel_df_row['file'])
    cnt = rel_df_row['cnt']
    rel_name = rel_df_row['name']
    rel_type = rel_df_row['type']
    empty_result = {"rel_name": rel_name, "rel_type": rel_type, "epochs": 0, 
            "best_acc": 0, "best_f1": 0, "best_prec": 0, "best_rec": 0, 
            "base_acc": 0.5, 
            "base_f1": 0.5,
            "base_prec": 0.5,
            "base_rec": 0.5,
            "best_model": "None", "best_model_type": "None",
            "pos_exs": cnt}
    if cnt < 75:
        print(rel_name, rel_type, 'too few examples')
        return empty_result
    
    if rel_type == 'rnd2rnd':
        X, Y = data_loader.generate_random_pair_data(target_size=cnt*2)
    elif rel_type in single_rel_types:
        X, Y = data_loader.load_single_data(osp.join(relpath, rel_df_row['file']))
    else:
        X, Y = data_loader.load_pair_data(osp.join(relpath, rel_df_row['file']))

    indim = X.shape[1]
    
    if indim == 300:
        nn3 = {"layer_dims": [400, 200, 100], "dropouts": [0.5, 0.5, 0.5]}
        nn2 = {"layer_dims": [400, 150], "dropouts": [0.5, 0.5]}
        nn1 = {"layer_dims": [300], "dropouts": [0.5]}
    elif indim == 600:
        nn3 = {"layer_dims": [750, 500, 250], "dropouts": [0.5, 0.5, 0.5]}
        nn2 = {"layer_dims": [750, 400],      "dropouts": [0.5, 0.5]}
        nn1 = {"layer_dims": [600],           "dropouts": [0.5]}
    else:
        raise Exception('Unexpected input dimension %d' % indim)
            
    assert torch.max(Y) == 1 and torch.min(Y) == 0, 'Expecting binary classifier but found max %d min %d ' % (torch.max(Y), torch.min(Y))
           
    if rel_filter and not rel_filter(rel_df_row):
        print(rel_name, rel_type, 'not in rel_name filter')
        return empty_result
    
    print("\n\n\n", rel_df_row['file'])
    epochs = epochs_from_ex_cnt(X.shape[0]/2)
    trainloader, validloader, testloader = data_loader.split_data(X, Y, seed=41)
    
    def get_test_accuracy(model_result):
        return model_result['test_df'].loc[0]['acc']
    
    def get_test_f1(model_result):
        return model_result['test_df'].loc[0]['f1']

    def get_test_precision(model_result):
        return model_result['test_df'].loc[0]['precision']

    def get_test_recall(model_result):
        return model_result['test_df'].loc[0]['recall']
    
    winner_model = None
    agg_models_results = []
    for model in models:
        print("model", model)
        model_results = []
        for run in range(n_crossval):
            if model == 'logreg':
                my_model = eval.LogisticRegression(indim)
            elif model == 'nn1':
                my_model = eval.NNBiClassifier(indim, nn1['layer_dims'], nn1['dropouts'])
            elif model == 'nn2':
                my_model = eval.NNBiClassifier(indim, nn2['layer_dims'], nn2['dropouts'])
            elif model == 'nn3':
                my_model = eval.NNBiClassifier(indim, nn3['layer_dims'], nn3['dropouts'])
            elif model == 'alwaysT':
                my_model = eval.DummyBiClassifier(indim, predef=[0.01, 0.99])
            elif model == 'alwaysF':
                my_model = eval.DummyBiClassifier(indim, predef=[0.99, 0.01])
            else:
                my_model = eval.NNBiClassifier(indim, nn2['layer_dims'], nn2['dropouts'])
            
            if run == 0:
                print('Training %s\n %d times...' % (my_model, n_crossval))
                
            try:
                trainer = eval.ModelTrainer(my_model, cuda=cuda)
                pretrain_test_result = trainer.test(testloader)
                trainer.train(trainloader, validloader, epochs=epochs, input_disturber=train_input_disturber)
                test_df = trainer.test_df(testloader, debug=True)
                plotter.expand(test_df)
                model_result = {"model": model, "i": run, 
                                "trainer": trainer, "pretrain_test_result": pretrain_test_result, "test_df": test_df}
                model_results.append(model_result)
                #plt = plotter.plot_learning(trainer.df, logreg_test_df, trainer.model_name)
            except:
                print("Unexpected error executing %s:" % model, sys.exc_info()[0])
                del trainer
                del my_model
                raise



        def extract_vals(model_results, value_extractor):
            """returns a list of values for a given list of model results"""
            result = []
            for model_result in model_results:
                result.append(value_extractor(model_result))
            return result
            
        test_accs = extract_vals(model_results, get_test_accuracy)
        test_f1s = extract_vals(model_results, get_test_f1)
        test_precision = extract_vals(model_results, get_test_precision)
        test_recall = extract_vals(model_results, get_test_recall)
        agg_model_results = {"model": model,
                         "avg_acc": np.mean(test_accs),        "std_acc": np.std(test_accs),
                         "avg_f1": np.mean(test_f1s),          "std_f1": np.std(test_f1s),
                         "avg_prec": np.mean(test_precision), "std_prec": np.std(test_precision),
                         "avg_rec": np.mean(test_recall),      "std_rec": np.std(test_recall),
                         "results": model_results}
        agg_models_results.append(agg_model_results)
        print('model %s acc %.3f+-%.3f f1 %.3f+-%.3f prec %.3f+-%.3f rec %.3f+-%.3f' % 
              (model,
               agg_model_results['avg_acc'], agg_model_results['std_acc'],
               agg_model_results['avg_f1'], agg_model_results['std_f1'],
               agg_model_results['avg_prec'], agg_model_results['std_prec'],
               agg_model_results['avg_rec'], agg_model_results['std_rec'],
              ))
        
        if not winner_model:
            winner_model = agg_model_results
        elif winner_model['avg_acc'] > agg_model_results['avg_acc']:
            print('Previous model %s (avg_acc %.2f, avg_f1 %.2f) is, on average, better than %s (avg_acc %.2f, avg_f1 %.2f)' % 
                  (winner_model['model'], winner_model['avg_acc'], winner_model['avg_f1'],
                   model, agg_model_results['avg_acc'], agg_model_results['avg_f1']))
        else:
            print('Previous model %s (avg_acc %.2f, avg_f1 %.2f) was, on average, worse than %s (avg_acc %.2f, avg_f1 %.2f)' % 
                  (winner_model['model'], winner_model['avg_acc'], winner_model['avg_f1'],
                   model, agg_model_results['avg_acc'], winner_model['avg_f1']))
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
    winner_trainer = best_result['trainer']
        
    plt = plotter.plot_learning(winner_trainer.df, best_result['test_df'], winner_trainer.model_name)
    plt.show()
    
    base_test_result = winner_trainer.test_random(testloader)
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
        
