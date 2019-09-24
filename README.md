# Assessing Relational Knowledge captured by embeddings using a Knowledge Graph
This repo allows you to assess how well word (or concept) embeddings capture 
relational knowledge, by using a Knowledge Graph (KG) as a ground truth.

In short, this library will help you train and evaluate binary classifiers using 
embeddings to predicting word-pair relations.  Although intended mainly for 
word-pair relations, it can also be used to predict word categories.

The full pipeline is explained in detail in:

  Ronald Denaux and Jose Manuel Gomez-Perez. 
  *Assessing the Lexico-Semantic Relational Knowledge Captured by Word and Concept Embeddings* 
  Accepted at [K-Cap 2019](http://www.k-cap.org/2019/accepted-papers/index.html).

## How to use
After you clone this repo, make sure you have all dependencies required (see `requirements.txt`). You can use `pip` or `conda`. We recommend to use a machine with a GPU in order to train the models much faster.

    conda create -n embrelassess
    conda activate embrelassess
    conda install pytorch=0.4.1 cuda90 -c pytorch
    conda install pytest pandas matplotlib seaborn
    conda install pandas


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
   We provide example word-pairs derived from WordNet.

### Training models

TODO

### Analysing learning results

TODO

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
    
