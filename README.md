# Assessing Relational Knowledge captured by embeddings using a Knowledge Graph
This repo allows you to assess how well word (or concept) embeddings capture 
relational knowledge, by using a Knowledge Graph (KG) as a ground truth.

In short, this library will help you train and evaluate binary classifiers using 
embeddings to predicting word-pair relations.  Although intended mainly for 
word-pair relations, it can also be used to predict word categories.

The full pipeline is explained in detail in:

  Ronald Denaux and Jose Manuel Gomez-Perez. 
  *Assessing the Lexico-Semantic Relational Knowledge Captured by Word and Concept Embeddings* 
  Accepted at [K-Cap 2019](http://www.k-cap.org/2019/accepted-papers/index.html). [arxiv pre-print](https://arxiv.org/abs/1909.11042v1)

## How to use
After you clone this repo, make sure you have all dependencies required (see `requirements.txt`). You can use `pip` or `conda`. We recommend to use a machine with a GPU in order to train the models much faster.

    conda create -n embrelassess
    conda activate embrelassess
    conda install pytorch=0.4.1 cuda90 -c pytorch
    conda install pytest pandas matplotlib seaborn

You need:
 - One or more embedding spaces. You can use publicly available
   embeddings like FastText or GloVe, or you can use your own.
 - One or more word-pair TSV files. Each file provides examples and
   counter examples of a specific relation (e.g. hypernymy,
   synonymy). Each line in these files must contain:
   - a source word
   - a target word
   - 1 or 0 indicating whether it is an example or a counter example
   - optionally a comment or identifier, especially useful for
     indicating the source of (counter) examples.
   You can generate example word-pairs derived from WordNet by using the `wnet-rel-pair-extractor` subproject.


### Training models

Training the models can be split into three steps:
1. load generated datasets
2. load embeddings to be evaluated
3. learn and train models 

The first step can be achieved by calling:

    import embrelassess.learn as learn
    import os.path as osp
    
    rel_path = osp.join('/your/path/to/generated_dataset/')
    rels_df = learn.load_rels_meta(rel_path)
    
The second step depends on which embeddings you want to load, but the following can give you an idea:

    import embrelassess.embloader as embloader
    import torchtext
    from torchtext.vocab import Vectors, FastText, GloVe
    
    vocab_path = osp.join('/your/path/to/vocab.txt')
    rnd_vecs = embloader.RandomVectors(vocab_path, dim=300)
    
    vec_cache = '/home/rdenaux/nbs/expsys/.vector_cache/'
    glove_en = GloVe(name='840B', cache=vec_cache)
    ft_en = FastText(language='en', cache=vec_cache)
    
    holE_wnet_en = embloader.TSVVectors('/data/models/kge/en_wnet_3.1-HolE-500e.vec')
    
The final step is to learn the models, here you can specify which embeddings to use, which model architecture, how many models should be learned for each relation, which relations to learn models for, folder where the models and their results should be stored, etc.:

    import embrelassess.learn as learn
    
    data_loaders = {
        'glove_cc_en':    embloader.VecPairLoader(glove_en),
        'ft_wikip_en':    embloader.VecPairLoader(ft_en),
        #'vecsi_wiki_en': embloader.VecPairLoader(vecsi_wiki_en),
        #'vecsi_un_en':   embloader.VecPairLoader(vecsi_un_en),
        'rand_en':        embloader.VecPairLoader(rnd_vecs),
        'holE_wnet_en':   embloader.VecPairLoader(holE_wnet_en)
    }
    models=['nn2', 'nn3']
    n_runs=3
    my_rels = ['verb_group', 'entailment']
    def only_with_names(rel_name_whitelist):
        return lambda df_row: df_row['name'] in rel_name_whitelist

    odir='/your/path/to/experiment/modress-190923/'
    learn_results = learn.learn_rels(rel_path, rels_df, data_loaders,
                                 models=models, n_runs=n_runs, 
                                 rel_filter=only_with_names(my_rels),
                                 train_input_disturber_for_vec=learn.pair_disturber_for_vectors,
                                 odir_path=odir,
                                 cuda=True)
                                 
    # the following is no longer needed (now stored as part of learn_rels)
    for learn_result in learn_results:
        learn.store_learn_result(odir, learn_result)
    
This will generate a directory structure in the `odir` with the following structure:

`(odir)/(rel_type)/(rel_name)/(emb_id)/(arch_id)/run_(number)/`

e.g.:

`odir/em2lem/entailment/ft_wikip_en/nn2/run_01/`

Inside each of these folders, you'll find the files:
* `meta.json`, `model.params`, `raw.model` which you can use to load the final model and params that were trained
* `pretrain-test.json` contains baseline test results for the model prior to any training (i.e. with the initial random parameters)
* `randomvec-test.json` contains another baseline test result based on random predictions
* `test.tsv` contains test results for the trained model (the table contains results for different threshold values)
* `train.tsv` contains metrics gathered during the training process (epochs, loss, validation metrics)

*In the next few weeks we will provide an example jupyter notebooks to show how to do this for a sample dataset based on WordNet.*

### Analysing learning results

Once you have learned and stored the models and test metrics, you can analyse them. We suggest to load the test results in a pandas DataFrame for easier processing and analysis:

    import embrelpredict.analyse as analyse
    import embrelassess.learn as learn
    
    lr_read = learn.load_learn_results(learn_results_path)
    
    aggs = []
    for learn_result in lr_read:
        rel_aggs = analyse.aggregate_runs(learn_result)
        print('found', len(rel_aggs), 'for relation', learn_result['rel_name'])
        aggs = aggs + rel_aggs
        
    aggs_df = pd.DataFrame(aggs)

*In the next few weeks we will provide an example jupyter notebook to show how to replicate the analysis described in [our paper](https://arxiv.org/abs/1909.11042v1)*


## Development

If you want to modify or contribute to this project, this section
provides an overview of the code and how to test it.

### Architecture

The code is distributed in various modules in the `embrelassess`
folder. Roughly from low-level to high-level:

 * `vecops`: provides methods for loading embeddings from local files
 * `model`: defines various Pytorch binary classification Modules
 * `embloader`: helps to convert word-pair tsv files into PyTorch
   datasets by looking up the corresponging word embeddings. This
   conversion is required for training the models.
 * `train`: provides methods to automate the training of a model
 * `learn`: provides methods to train various models, based on
   different embeddings and store the learning results in a
   standardized way. This essentially takes care of generating and
   storing the training and baseline data.
 * `analyse`: provides methods for displaying and analysing learning
   results generated previously.
 

### Testing
To run the unit tests simply run

    pytest
    
This executes most of the unit tests, except some which are slower. To run all the tests, run 

    pytest --runslow
    
