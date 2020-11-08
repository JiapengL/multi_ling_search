# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pickle
from collections import defaultdict
import math
import random
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import pdb
from itertools import compress



class DataProcessor(object):

    def __init__(self, args):

        self.caseless = args.caseless
        self.truncate = args.truncate
        if args.build_data:
            print("loading train_data ...")
            self.raw_train = self._build_data(args.train_file, args.caseless)
            print("loading dev_data ...")
            self.raw_dev = self._build_data(args.dev_file, args.caseless)
            print("loading test_data ...")
            self.raw_test = self._build_data(args.test_file, args.caseless)
            if args.vat or args.self:
                print("loading unlabelled train_data ...")
                self.raw_train_ul = self._build_data(args.train_ul_file, args.caseless)

        else:
            print("loading train_data ...")
            self.raw_train = self._load_data(args.train_file)
            print("loading dev_data ...")
            self.raw_dev = self._load_data(args.dev_file)
            print("loading test_data ...")
            self.raw_test = self._load_data(args.test_file)
            if args.vat or args.self:
                print("loading unlabelled train_data ...")
                self.raw_train_ul = self._load_data(args.train_ul_file)
            if args.train_eval:
                print("loading train_data_2 ...")
                self.raw_train_2 = self._load_data(args.train_file)

    def build_vocab(self):
        """Build the vocabulary for sentence pairs in the training set, vocab outside this will be considered as <unk>
        """

        q_vocab = defaultdict(int)
        q_vocab["<pad>"] = len(q_vocab)
        q_vocab["<UNK>"] = len(q_vocab)

        d_vocab = defaultdict(int)
        d_vocab["<pad>"] = len(d_vocab)
        d_vocab["<UNK>"] = len(d_vocab)

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
        d_vocab = defaultdict(int)

        with open(args.q_vocab_path, "rb") as f_q:
            voc_q = pickle.load(f_q)
        for i, w in enumerate(voc_q):
            if i < args.vocab_size:
                w = w.lower() if self.caseless else w
                if w not in q_vocab:
                    q_vocab[w.strip()] = len(q_vocab)

        with open(args.d_vocab_path, "rb") as f_d:
            voc_d = pickle.load(f_d)
        for i, w in enumerate(voc_d):
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


    def _load_data(self, data_path):
        """Build the token dataset for sentence pairs, each data point is represented by (relevance score, query_text, doc_text)
        """

        with open(data_path, "rb") as f:
            data_ls = pickle.load(f)

        return data_ls



    def generate_batch(self, batch_size, is_shuffle=False, dataset="None"):
        """Generate the batch of dataset
        """

        if dataset == "train":
            data_ls = self.raw_train
        elif dataset == "dev":
            data_ls = self.raw_dev
        elif dataset == "test":
            data_ls = self.raw_test
        elif dataset == "unlabel":
            data_ls = self.raw_train_ul
        elif dataset == "train_2":
            data_ls = self.raw_train_2

        if is_shuffle:
            random.shuffle(data_ls)

        ttl_bn = int(math.ceil(len(data_ls)*1.0/batch_size))

        for bn in range(ttl_bn):

            yield data_ls[(bn*batch_size):(bn+1)*batch_size]



    def generate_ul_batch(self, batch_size, prediction, label=None, is_shuffle=False):
        """Generate the batch of dataset
        """
        if self.truncate == 0:
            data_ls = self.raw_train_ul
        else:
            data_ls = [data for data, l in zip(self.raw_train_ul, label) if l]


        if is_shuffle:
            assert len(prediction) == len(data_ls)
            pred = prediction.tolist()
            data_pred = list(zip(pred, data_ls))
            random.shuffle(data_pred)
            prediction, data_ls = zip(*data_pred)


        ttl_bn = int(math.ceil(len(data_ls)*1.0/batch_size))

        for bn in range(ttl_bn):

            yield prediction[(bn*batch_size):(bn+1)*batch_size], data_ls[(bn*batch_size):(bn+1)*batch_size]




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
        """
        Generate the batch of raw test dataset
        """

        data_ls = self.raw_test

        ttl_bn = int(math.ceil(len(data_ls)*1.0/batch_size))

        for bn in range(ttl_bn):

            yield data_ls[(bn*batch_size):(bn+1)*batch_size]


    def generate_data_index(self, set):
        """
        generate the index for rel == 2
        """

        n_qd_pairs = [0]
        index = []
        count = -1
        if set == "dev":
            data_ls = self.raw_dev
        elif set == "test":
            data_ls = self.raw_test
        elif set == "train":
            data_ls = self.raw_train
        elif set == "train_2":
            data_ls = self.raw_train_2

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