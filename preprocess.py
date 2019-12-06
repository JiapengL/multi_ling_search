import sys

sys.path.insert(0, '../transformers')
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import math
import random
import time
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import pdb
import torch
from torch import cuda
import pickle

from src.dssm_model import SimpleDSSM, DeepDSSM
from src.dt_processor import DataProcessor
from src.utils import numerize
import pdb


class DataProcessor(object):

    def __init__(self, args):

        self.caseless = args.caseless

        if args.build_data:
            print("loading train_data ...")
            self.raw_train = self._build_data(args.train_file, args.caseless)
            print("loading dev_data ...")
            self.raw_dev = self._build_data(args.dev_file, args.caseless)
            #print("loading test_data ...")
            #self.raw_test = self._build_data(args.test_file, args.caseless)
        else:
            print("loading train_data ...")
            self.raw_train = self._load_data(args.train_file)
            print("loading dev_data ...")
            self.raw_dev = self._load_data(args.dev_file)
            #print("loading test_data ...")
            #self.raw_test = self._load_data(args.test_file)


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


    def generate_train_batch(self, batch_size, is_shuffle=False):
        """Generate the batch of raw training dataset
        """

        data_ls = self.raw_train

        if is_shuffle:
            random.shuffle(data_ls)

        ttl_bn = int(math.ceil(len(data_ls)*1.0/batch_size))

        for bn in range(ttl_bn):

            yield data_ls[(bn*batch_size):(bn+1)*batch_size]



def pad_convert(sent, vocab, max_l, pad_type="back"):
    """
    padding embeddings for non-equal length of sentences
    """

    num_st = [vocab[t] if t in vocab else vocab["<UNK>"] for t in sent]

    if pad_type=="front":
        num_st = [vocab["<PAD>"]]*(max_l-len(sent)) + num_st
    elif pad_type=="both":
        if (max_l-len(sent))%2 == 0:
            pad_l = (max_l-len(sent))/2
            num_st = [vocab["<PAD>"]]*pad_l + num_st + [vocab["<PAD>"]]*pad_l
        else:
            pad_l, pad_r = (max_l-len(sent))/2, (max_l-len(sent))/2+1
            num_st = [vocab["<PAD>"]]*pad_l + num_st + [vocab["<PAD>"]]*pad_r
    else:
        num_st = num_st + [vocab["<PAD>"]]*(max_l-len(sent))

    return num_st


def numerize(batch, q_vocab, d_vocab, pad_type="back"):
    """
    numerize query and docs to embeddings
    return: relevance score, query embedding, doc embedding
    """

    labels = [int(pair[0]) for pair in batch]
    _len = len(labels)
    batch_max_q_len = max(map(lambda st: len(st), [pair[1] for pair in batch]))
    batch_max_d_len = max(map(lambda st: len(st), [pair[2] for pair in batch]))

    q_arr = q_vocab["<PAD>"] * np.ones([_len, batch_max_q_len])
    d_arr = d_vocab["<PAD>"] * np.ones([_len, ])


    q_x = [pad_convert(pair[1], q_vocab, batch_max_q_len, pad_type) for pair in batch]


    d_x = [pad_convert(pair[2], d_vocab, batch_max_d_len, pad_type) for pair in batch]

 #   if windom_size != 1:

    return torch.tensor(labels).long(), torch.tensor(q_x).long(), torch.tensor(d_x).long()





def run(args):
    random.seed(123)
    np.random.seed(123)
    torch.cuda.manual_seed(123)
    torch.manual_seed(123)

    time1 = time.time()
    #use gpu is args.use_gpu is True
    if args.use_gpu:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device is:", device)

    # data setup
    data_processor = DataProcessor(args)

    # load vocabulary
    if args.create_vocabulary:
        vocab_q, vocab_d = data_processor.build_vocab()
    else:
        vocab_q, vocab_d = data_processor.load_vocab(args)

    print('The length of query vocabulary is ', len(vocab_q))
    print('The length of doc vocabulary is ', len(vocab_d))
    #vocab_q, vocab_d = data_processor.build_vocab()
    time2 = time.time()
    print('Loading running time is:', time2 - time1)

    # model setup
    if args.deep:
        model = DeepDSSM(args, len(vocab_q), len(vocab_d))
    else:
        model = SimpleDSSM(args, len(vocab_q), len(vocab_d))


    if args.use_gpu:
        model.to(device)
        print(model)

    # load embedding
    if args.q_extn_embedding:
        model.load_embeddings(args, vocab_q, "query")

    if args.d_extn_embedding:
        model.load_embeddings(args, vocab_d, "document")
    i = 0
    hf = h5py.File('/home/liu1769/scratch/data_eng__french50000/train.hdf5', 'w')

    for batch in data_processor.generate_train_batch(args.train_batchsize, args.is_shuffle):
        rels, qs, ds = numerize(batch, vocab_q, vocab_d)
        #pdb.set_trace()
        hf.create_dataset(str(i)+'_label', data=rels)
        hf.create_dataset(str(i)+'_qu', data=qs)
        hf.create_dataset(str(i)+'_doc', data=ds)
        if i%10000 == 0:
            print(i)
        i += 1

    hf.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    # data

    parser.add_argument('--build_data', dest='build_data', action='store_true', help='if True, build data list from csv file')
    parser.add_argument('--len_token', dest='len_token', type=int, default='200', help='the length of tokenize for each sentence')

    parser.add_argument('--train_file', dest='train_file', type=str, default='/home/liu1769/scratch/data_eng__french50000/train.pkl', help='path to training file')
    parser.add_argument('--dev_file', dest='dev_file', type=str, default='/home/liu1769/scratch/data_eng__french50000/dev_small.pkl', help='path to development file')
    parser.add_argument('--test_file', dest='test_file', type=str, default='/home/liu1769/scratch/data_eng__french50000/test.pkl', help='path to test file')

    """
    parser.add_argument('--train_file', dest='train_file', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/toy_eng_data/train.csv', help='path to training file')
    parser.add_argument('--dev_file', dest='dev_file', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/toy_eng_data/dev.csv', help='path to development file')
    parser.add_argument('--test_file', dest='test_file', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/toy_eng_data/test.csv', help='path to test file')
    """
    # embeddings

    parser.add_argument('--create_vocabulary', action='store_true', help='')
    parser.add_argument('--vocab_size', dest='vocab_size', type=int, default='100000', help='size of vocabulary')

    """
    parser.add_argument('--q_vocab_path', dest='q_vocab_path', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/vocab_en.pkl', help='path to vocabulary for queries')
    parser.add_argument('--d_vocab_path', dest='d_vocab_path', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/vocab_fr.pkl', help='path to vocabulary embedding for documents')
    parser.add_argument('--q_extn_embedding', dest='q_extn_embedding', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/polyglot_en_dict.pkl', help='path to pre-trained embedding for queries')
    parser.add_argument('--d_extn_embedding', dest='d_extn_embedding', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/polyglot_fr_dict.pkl', help='path to pre-trained embedding for documents')

    """
    parser.add_argument('--q_vocab_path', dest='q_vocab_path', type=str, default='/home/liu1769/multi_ling_search/word_embed/vocab_en.pkl', help='path to vocabulary for queries')
    parser.add_argument('--d_vocab_path', dest='d_vocab_path', type=str, default='/home/liu1769/multi_ling_search/word_embed/vocab_fr.pkl', help='path to vocabulary embedding for documents')
    parser.add_argument('--q_extn_embedding', dest='q_extn_embedding', type=str, default='/home/liu1769/multi_ling_search/word_embed/polyglot_en_dict.pkl', help='path to pre-trained embedding for queries')
    parser.add_argument('--d_extn_embedding', dest='d_extn_embedding', type=str, default='/home/liu1769/multi_ling_search/word_embed/polyglot_fr_dict.pkl', help='path to pre-trained embedding for documents')


    # training
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='whether to use gpu')
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='number of epochs to run')
    parser.add_argument('--train_batchsize', dest='train_batchsize', type=int, default=32, help='training minibatch size')
    parser.add_argument('--test_batchsize', dest='test_batchsize', type=int, default=32, help='testing minibatch size')
    parser.add_argument('--is_shuffle', dest='is_shuffle', action='store_true', help='whether to shuffle training set each epoch')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default="adam")
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--clip_grad', dest='clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--caseless', dest='caseless', action='store_true', help='caseless or not')
    parser.add_argument('--fine_tune', dest='fine_tune', action='store_true', help='whether to fine tune the word embedding or not')

    # neural model
    parser.add_argument('--deep', action='store_true', help='')
    parser.add_argument('--n_hdim', dest='n_hdim', type=int, default=200, help='dimension of hidden layer')
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=64, help='dimension of the embedding')

    # loss function
    parser.add_argument('--theta', dest='theta', nargs='*', type=int, default=[-0.5, 0.5], help='thresholds of loss function ')

    # save
    parser.add_argument('--model_path', dest='model_path', type=str, default='')
    args = parser.parse_args()
    run(args)

