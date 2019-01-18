This repository was created as a part of a diploma thesis 
[Semantic Sentence Similarity for Intent Recognition Task](https://dspace.cvut.cz/bitstream/handle/10467/77029/F3-DP-2018-Brich-Tomas-Semantic_Sentence_Similarity_for_Intent_Recognition_Task.pdf?sequence=-1&isAllowed=y).
 While all the included code works and is well documented, as of now, 
 it might be hard for anyone to actually use it, but feel free to try. 
 It is my intention to convert it into a proper Python library in the future.

# intent-reco
Template-based intent recognition system built on word embedding models.

## Installation instructions:

## Prerequisites:
Install Python 3.6 and higher
```commandline
sudo python3 python3-pip python3-dev build-essential
```
Install Python package manager `pipenv`
```commandline
pip3 install --user pipenv
```

Create and enter the virtual environment
```commandline
pipenv shell
```

Install dependencies (including development):
```commandline
pipenv install --dev
```

The intent recognition system is dependent on the used embedding model. 
These models are loaded using the wrappers in  ```embeddings``` directory.

Currently supported embedding algorithms:
* InferSent: https://github.com/facebookresearch/InferSent
* sent2vec: https://github.com/epfml/sent2vec
* GloVe: https://github.com/maciejkula/glove-python or SpaCy package implementation
* word2vec: gensim package implementation
* FastText: https://github.com/facebookresearch/fastText or gensim package implementation
* StarSpace: https://github.com/facebookresearch/StarSpace

Depending on the embedding algorithm used, it might be needed to install its implementation. You will find the installation instructions for each algorithm on the respective repository.

## Model compression

Module ```model_compression.py``` includes functions for compressing embedding models. It is able to compress the models by using different versions of vocabulary pruning and by using vector quantization.

The vector quantization is based on the LBG clustering algorithm, which is implemented in module ```lbg.py```.

## Intent recognition

The resulting intent recognition system is implemented in ```intent_query.py```. As of now, it loads an embedding model and a set of intent templates from a JSON file (the templates are further quantized). The user can then write sentences in the command line and the algorithm will output the matched intent, together with the respective template and its cosine similarity to the input sentence.

Examples of an embedding model and a JSON template file can be found in the ```data``` directory.
