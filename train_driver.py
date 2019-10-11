# coding: utf-8
import os
import random
import numpy as np
import argparse
from datetime import datetime
import pickle, json

HOME = os.getenv("HOME")

import torch
import torch.nn as nn
import torch.nn.functional  as F

from src.dssm_model import SimpleDSSM, DeepDSSM
from src.dt_processor import DataProcessor
from src.utils import numerize

import pdb


def run(args):
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    # data setup
    data_processor = DataProcessor(args)
    vocab_q, vocab_d = data_processor.build_vocab()
    print('The length of query vocabulary is ', len(vocab_q))
    print('The length of doc vocabulary is ', len(vocab_d))
    # model setup
    if args.deep:
        model = DeepDSSM(args, len(vocab_q), len(vocab_d))
    else:
        model = SimpleDSSM(args, len(vocab_q), len(vocab_d))

    # load embedding
    if args.q_extn_embedding:
        model.load_embeddings(args, vocab_q, "query")

    if args.d_extn_embedding:
        model.load_embeddings(args, vocab_d, "document")

    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    dev_set = data_processor.raw_dev
    rels_dev, qs_dev, ds_dev = numerize(dev_set, vocab_q, vocab_d)
    n_qd_pairs, index = data_processor.generate_eval_index()
    # print(n_qd_pairs, len(index[0]), len(dev_set))
    # print(dev_set[0])
"""
    for epoch in range(1, 1+args.epochs):
        model.train()
        loss_total = []
        for batch in data_processor.generate_train_batch(args.train_batchsize, args.is_shuffle):

            rels, qs, ds = numerize(batch, vocab_q, vocab_d)
            sims = model.forward(qs, ds, rels)

            model.zero_grad()

            loss = model.cal_loss(sims, rels)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            loss_total.append(loss.data)

            # pdb.set_trace()
        print('The training loss at Epoch ', epoch, 'is ', np.mean(loss_total))

        # evaluation on dev set
        model.eval()
        sims_dev = model.forward(qs_dev, ds_dev, rels_dev)
        loss_dev = model.cal_loss(sims_dev, rels_dev)
        print('The evaluation loss at Epoch ', epoch, 'is ', loss_dev)
        # todo: evaluate on the dev set
        # todo: decay learning rate

    # todo: evaluate on the final test set
"""


if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    # data
    parser.add_argument('--train_file', dest='train_file', type=str, default='toy_eng_data/train.csv', help='path to training file')
    parser.add_argument('--dev_file', dest='dev_file', type=str, default='toy_eng_data/dev.csv', help='path to development file')
    parser.add_argument('--test_file', dest='test_file', type=str, default='toy_eng_data/test.csv', help='path to test file')

    # embeddings
    parser.add_argument('--q_extn_embedding', dest='q_extn_embedding', type=str, default='', help='path to pre-trained embedding for queries')
    parser.add_argument('--d_extn_embedding', dest='d_extn_embedding', type=str, default='', help='path to pre-trained embedding for documents')

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


