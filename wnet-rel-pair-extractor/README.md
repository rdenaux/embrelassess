# WordNet Relation Pair Extractor
Java project to generate balanced datasets based on WordNet and a seed vocabulary.

## Build
Use maven, it should fetch all dependencies and build an uber-jar

  mvn install 
  
## Test/execute
We include a sample vocabulary, extracted from the first thousand lines from the English Europarl v7 dataset, which you can use to test the functionality. Once you have built an uber-jar:

  java -jar .\wnet-relpair-extractor-0.1.1.jar --vocab ../src/test/resources/corpus/europarlv7_en_1k/tlgs_wnscd/vocab.txt
  
After a short while, this should create a folder `wn_vocabrels` next to the `vocab.txt` file with large number of generated datasets. 

### Supported vocab format
The input `vocab` should be a simple text file with a vocabulary term in each line.

Lemmas must have prefix `lem_`, e.g. `lem_be`.

Synsets must be represented as `wn31_(POS_name)#(offset)`. E.g.:

    lem_be
    wn31_VERB#2610777

Any lines not adhering to these two formats will be ignored.

## Genererated datasets

The name of the generated files follows the following pattern:

  (rel_type)_(rel_name)__(size).txt
  
Where `rel_type` is either:
* `lem2lem` relation between two lemmas 
* `lem2pos` relation between a lemma and a part of speech
* `lem2syn` relation between a lemma and a synset
* `syn2pos` 
* `syn2syn`

`rel_name` is a relation name from WordNet

`size` is the number of **positive** pairs extracted from WordNet, the file will usually have about twice that number of examples, since it will include generated negative samples.

Each generated file has a TSV (tab-separated value) format. E.g. the first few lines for the generated `lem2lem_also_see__137.txt` looks as follows:
    
    even	smooth	1	positive
    legitimate	necessary	0	[NegSwitched]
    pass	deliver	1	positive
    have	fund	0	[NegSwitched]
    like	same	1	positive
    significant	valid	0	[NegSwitched]
    
So, the columns are:
1. the first word/synset in the pair 
2. the second word/synset/POS in the pair
3. whether this is a positive (1) or negative (0) sample for the relation
4. the last column provides further information about how the pair was derived from WordNet, this is especially informative about negative pairs. This will usually be `NegSwitched` (derived from the positive pairs for the relation), but in some cases this may be `Compatible-rel` (taken from a  different relation, but with the same `rel_type`).

