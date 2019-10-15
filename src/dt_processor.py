# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pickle
from collections import defaultdict
import math

from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize

import pdb

class DataProcessor(object):

    def __init__(self, args):

        self.caseless = args.caseless
        self.raw_train = self._build_data(args.train_file, args.caseless)
        self.raw_dev = self._build_data(args.dev_file, args.caseless)
        self.raw_test = self._build_data(args.test_file, args.caseless)


    def build_vocab(self):
        """Build the vocabulary for sentence pairs in the training set, vocab outside this will be considered as <unk>
        """

        q_vocab = defaultdict(int)
        q_vocab["<pad>"] = len(q_vocab)
        q_vocab["<unk>"] = len(q_vocab)

        d_vocab = defaultdict(int)
        d_vocab["<pad>"] = len(d_vocab)
        d_vocab["<unk>"] = len(d_vocab)

        for line in self.raw_train:
            _, q_text, d_text = line[0], line[1], line[2]
            for qtk in q_text:
                tok = qtk.lower() if self.caseless else qtk
                if tok not in q_vocab:
                    q_vocab[tok] = len(q_vocab)
            for dtk in d_text:
                tok = dtk.lower() if self.caseless else dtk
                if tok not in d_vocab:
                    d_vocab[tok] = len(d_vocab)
        return q_vocab, d_vocab

    def load_vocab(self, args):
        """
        load language vocabulary from files with given size
        """

        q_vocab = defaultdict(int)
        q_vocab["<pad>"] = len(q_vocab)
        q_vocab["<unk>"] = len(q_vocab)

        d_vocab = defaultdict(int)
        d_vocab["<pad>"] = len(d_vocab)
        d_vocab["<unk>"] = len(d_vocab)

        for i, w in enumerate(open(args.q_vocab_path, "r")):
            if i < args.vocab_size:
                w = w.lower() if self.caseless else w
                if w not in q_vocab:
                    q_vocab[w.strip()] = len(q_vocab)
        for i, w in enumerate(open(args.d_vocab_path, "r")):
            if i < args.vocab_size:
                w = w.lower() if self.caseless else w
                if w not in d_vocab:
                    d_vocab[w.strip()] = len(d_vocab)
        return q_vocab, d_vocab

    def _build_data(self, data_path, caseless):
        """Build the token dataset for sentence pairs, each data point is represented by (relevance score, query_text, doc_text)
        """

        data_ls = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                rel, q_text_s, d_text_s = line.strip('\n').split('\t')
                q_text, d_text = word_tokenize(q_text_s), word_tokenize(d_text_s)

                if caseless:
                    q_text = [t.lower() for t in q_text]
                    d_text = [t.lower() for t in d_text]

                data_ls.append([rel, q_text, d_text])

        return data_ls

    def generate_train_batch(self, batch_size, is_shuffle=False):
        """Generate the batch of raw training dataset
        """

        data_ls = self.raw_train
        if is_shuffle:
            random.shuffle(data_ls)

        ttl_bn = int(math.ceil(len(data_ls)*1.0/batch_size))

        for bn in range(ttl_bn):

            yield data_ls[(bn*batch_size):(bn+1)*batch_size]

    def generate_dev_batch(self, batch_size):
        """Generate the batch of raw dev dataset
        """

        data_ls = self.raw_dev

        ttl_bn = int(math.ceil(len(data_ls)*1.0/batch_size))

        for bn in range(ttl_bn):

            yield data_ls[(bn*batch_size):(bn+1)*batch_size]

    def generate_test_batch(self, batch_size):
        """Generate the batch of raw test dataset
        """

        data_ls = self.raw_test

        ttl_bn = int(math.ceil(len(data_ls)*1.0/batch_size))

        for bn in range(ttl_bn):

            yield data_ls[(bn*batch_size):(bn+1)*batch_size]

    def generate_eval_index(self):

        n_qd_pairs = [0]
        index = []
        count = -1

        data_ls = self.raw_dev
        for i, line in enumerate(data_ls):
            count += 1
            rel = int(line[0])
            if rel == 2:
                if i != 0:
                    n_qd_pairs.append(count)
                    index.append(rels_vector)
                    rels_vector = [rel]
                else:
                    rels_vector = [rel]

            else:
                rels_vector.append(rel)
        index.append(rels_vector)
        n_qd_pairs.append(len(data_ls))
        return n_qd_pairs, index