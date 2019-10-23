# coding: utf-8
import os
import random
import numpy as np
import argparse
from datetime import datetime
import pickle, json
import time

HOME = os.getenv("HOME")

import torch
import torch.nn as nn
import torch.nn.functional  as F

from src.dssm_model import SimpleDSSM, DeepDSSM
from src.dt_processor import DataProcessor
from src.utils import numerize
import src.evaluator as eval
import pdb


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

    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    """
    n_qd_pairs, index = data_processor.generate_eval_index()
    print('The index of relevant qd_pairs are', n_qd_pairs)
    print('The length of dev_set is ', len(dev_set))
    print('The number of queries in dev_set is', len(n_qd_pairs))
    #print('The length of docs for the first query is ', index[10:15])
    print(dev_set[0])
    """

    # load dev data
    dev_set = data_processor.raw_dev
    qd_index, rels_index = data_processor.generate_eval_index()
    rels_dev, qs_dev, ds_dev = numerize(dev_set, vocab_q, vocab_d)
    if args.use_gpu:
        rels_dev, qs_dev, ds_dev = rels_dev.to(device), qs_dev.to(device), ds_dev.to(device)

    #training and evaluating
    for epoch in range(1, 1+args.epochs):

        model.train()
        loss_total = []
        i = 0
        for batch in data_processor.generate_train_batch(args.train_batchsize, args.is_shuffle):
            rels, qs, ds = numerize(batch, vocab_q, vocab_d)

            if args.use_gpu:
                rels, qs, ds = rels.to(device), qs.to(device), ds.to(device)
#            print(batch[1], len(batch[1][1]))
#            print(qs[1], qs.shape)
#            qs_input, ds_input = model.forward(qs, ds, rels)
#            print(qs_input, ds_input)
            sims = model.forward(qs, ds, rels)
            model.zero_grad()

            if i%10000 == 0:
                print(i)

            i += 1
            loss = model.cal_loss(sims, rels)
            loss.backward()

            #nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            loss_total.append(loss.data)
            # pdb.set_trace()
        #print('The first 100 training loss', loss_total[:20])
        print('The last 100 train loss', loss_total[-20:])
        print('The training loss at Epoch ', epoch, 'is ', torch.mean(torch.stack(loss_total)).item())
        print('is nan:', torch.sum(torch.isnan(torch.stack(loss_total))))


        # evaluation on dev set
        model.eval()

        sims_dev = model.forward(qs_dev, ds_dev, rels_dev)
        precision = eval.precision_at_k(sims_dev, qd_index, rels_index, 5)
        print('the precision @ 5 is', np.mean(precision))
        dcg = eval.dcg_at_k(sims_dev, qd_index, rels_index, 5)
        print('the dcg is', np.mean(dcg))
        ndcg = eval.ndcg_at_k(sims_dev, qd_index, rels_index, 5)
        print('the ndcg is', np.mean(ndcg))

        loss_dev = model.cal_loss(sims_dev, rels_dev)
        print('The evaluation loss at Epoch ', epoch, 'is ', loss_dev.data.item())
        #print('The similarity of dev is ', sims_dev)

        # todo: decay learning rate

    # todo: evaluate on the final test set



if __name__ == '__main__':
    parser=argparse.ArgumentParser()

    # data

    parser.add_argument('--build_data', dest='build_data', action='store_true', help='if True, build data list from csv file')

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


