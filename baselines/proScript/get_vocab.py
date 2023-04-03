# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import csv
import json
from tqdm import tqdm
from nltk.tokenize import word_tokenize


def save_vocab(datafile):
    # tuple: (input_hypothesis, output_graph_DOT, output_dsl_graph_DOT)
    file_tuples = [tuple(sample[0].split('\t')) for sample in list(csv.reader(open(datafile),
                                                                              delimiter="\n"))]
    vocab = set()
    for sample_tuple in tqdm(file_tuples):
        inp, out1, _ = sample_tuple
        # vocab = vocab.union(set(word_tokenize(inp)))
        vocab = vocab.union(set(word_tokenize(out1)))
    with open('my_vocab.json', 'w') as jsonfile:
        json.dump(list(vocab), jsonfile)
