# coding: utf-8
import os
import random
import numpy as np
import argparse
from datetime import datetime
import pickle, json
import time
import h5py

HOME = os.getenv("HOME")

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dssm_model import SimpleDSSM, DeepDSSM, LSTM
from src.adversarial import Adv_Simple, Adv_Deep
from src.dt_processor import DataProcessor
from src.utils import numerize
from src.vat_semi import VAT_Simple
import src.evaluator as eval

import pdb
from GPUtil import showUtilization as gpu_usage
from operator import itemgetter



def run(args):
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    print('The theta is ', args.theta)
    print('The m is ', args.m)
    print('whether use adv is ', args.use_adv)

    time1 = time.time()
    # use gpu is args.use_gpu is True
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
    # vocab_q, vocab_d = data_processor.build_vocab()
    time2 = time.time()
    print('Loading running time is:', time2 - time1)

    # model setup
    if args.lstm:
        model = LSTM(args, len(vocab_q), len(vocab_d))
    else:
        if args.use_adv:
            if args.deep:
                model = Adv_Deep(args, len(vocab_q), len(vocab_d))
            else:
                model = Adv_Simple(args, len(vocab_q), len(vocab_d))

        elif args.deep:
            model = DeepDSSM(args, len(vocab_q), len(vocab_d))
        else:
            model = SimpleDSSM(args, len(vocab_q), len(vocab_d))

    # whether use gpu
    if args.use_gpu:
        model.to(device)
        print(model)
    else:
        print(model)

    if args.vat:
        reg_fn = VAT_Simple(model, args, len(vocab_q), len(vocab_d))

    # load embedding
    if args.q_extn_embedding:
        model.load_embeddings(args, vocab_q, "query")
    if args.d_extn_embedding:
        model.load_embeddings(args, vocab_d, "document")

    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.9)

    # generate data index
    dev_qd_index, dev_rels_index = data_processor.generate_data_index("dev")
    test_qd_index, test_rels_index = data_processor.generate_data_index("test")

    best_ndcg = 0
    best_precision_nn = 0
    best_precision_2 = 0

    # training and evaluating
    for epoch in range(1, 1 + args.epochs):

        model.train()
        loss_total = []
        loss_pure = []
        i = 0   # training iterations
        for batch in data_processor.generate_train_batch(args.train_batchsize, args.is_shuffle):
            #  numerize labeled data
            rels, qs, ds = numerize(batch, vocab_q, vocab_d)
            if args.use_gpu:
                rels, qs, ds = rels.to(device), qs.to(device), ds.to(device)

            model.zero_grad()
            sims_l = model(qs, ds)
            loss = model.cal_loss(sims_l, rels)

            if args.vat:
                # sample and numerize unlabeled data

                batch_indices_ul = torch.LongTensor(
                    np.random.choice(len(data_processor.raw_train_ul), args.train_batchsize, replace=False))
                batch_ul = itemgetter(*batch_indices_ul)(data_processor.raw_train_ul)
                _, qs_ul, ds_ul = numerize(batch_ul, vocab_q, vocab_d)

                if args.use_gpu:
                    qs_ul, ds_ul = qs_ul.to(device), ds_ul.to(device)

                sims_ul = model(qs_ul, ds_ul)
                vat_loss = reg_fn(qs_ul, ds_ul, sims_ul)
                loss += 0.1*vat_loss
                loss_pure.append(vat_loss.item())

                """
                # sims_ul = model(qs, ds)
                vat_loss = reg_fn(qs, ds, sims_l)
                loss += 0.5*vat_loss
                """

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            loss_total.append(loss.item())

            if i % 10000 == 0:
                print(i)

            i += 1

        print('The training loss at Epoch ', epoch, 'is {:05.5f}'.format(np.mean(loss_total)))
        if args.vat:
            print('The Pure training loss at Epoch ', epoch, 'is {:05.5f}'.format(np.mean(loss_total)))

        # evaluation on dev set
        model.eval()
        loss_dev = []
        sims_dev = torch.tensor([]).to(device) if args.use_gpu else torch.tensor([])
        for dev_batch in data_processor.generate_dev_batch(args.train_batchsize):

            rels_dev, qs_dev, ds_dev = numerize(dev_batch, vocab_q, vocab_d)
            if args.use_gpu:
                rels_dev, qs_dev, ds_dev = rels_dev.to(device), qs_dev.to(device), ds_dev.to(device)

            sims = model(qs_dev, ds_dev)
            sims_dev = torch.cat([sims_dev, sims.data])

            loss = model.cal_loss(sims, rels_dev)
            loss_dev.append(loss.item())

        # evaluation
        dev_precision_nn, dev_precision_2 = eval.precision_at_k(sims_dev, dev_qd_index, dev_rels_index, 5)
        dev_ndcg = eval.ndcg_at_k(sims_dev, dev_qd_index, dev_rels_index, 5)

        print('Epoch {}: dev loss: {:05.5f}; dev ndcg  @ 5: {:05.3f}; dev precision_nn  @ 5: {:05.3f}; dev precision_2  @ 5: {:05.3f}'
              .format(epoch, np.mean(loss_dev), dev_ndcg, dev_precision_nn, dev_precision_2))
        # prediction matrix
        """
        prediction = eval.predict(sims_dev, args.theta)
        pre_dict = eval.prediction_matrix(prediction, dev_rels_index)
        print('The prediction matrix is following, rows are rels and columns are prediction:')
        print(pre_dict)
        
        if epoch > 10:
            scheduler.step()
        """


        # test
        if epoch > 5:
            if dev_ndcg > best_ndcg:
                patience_count = 0
                best_ndcg = dev_ndcg
                best_precision_nn = dev_precision_nn
                best_precision_2 = dev_precision_2

                # evaluate on test set

                model.eval()
                loss_test = []
                sims_test = torch.tensor([]).to(device) if args.use_gpu else torch.tensor([])
                for test_batch in data_processor.generate_test_batch(args.train_batchsize):

                    rels_test, qs_test, ds_test = numerize(test_batch, vocab_q, vocab_d)
                    if args.use_gpu:
                        rels_test, qs_test, ds_test = rels_test.to(device), qs_test.to(device), ds_test.to(device)

                    sims = model(qs_test, ds_test)
                    sims_test = torch.cat([sims_test, sims.data])

                    loss = model.cal_loss(sims, rels_test)
                    loss_test.append(loss.item())

                test_precision_nn, test_precision_2 = eval.precision_at_k(sims_test, test_qd_index, test_rels_index, 5)
                test_ndcg = eval.ndcg_at_k(sims_test, test_qd_index, test_rels_index, 5)
                print('Epoch {}: test ndcg  @ 5: {:05.3f}; test precision_nn  @ 5: {:05.3f}; test precision_2  @ 5: {:05.3f}'.format(epoch, test_ndcg, test_precision_nn, test_precision_2))


            else:
                patience_count += 1

            if patience_count >= args.patience:
                break

    print('best dev ndcg @ 5: {:05.3f}; best dev precision_nn @ 5: {:05.3f}; best dev precision_2 @ 5: {:05.3f}'.format(best_ndcg, best_precision_nn, best_precision_2))
    print('best test ndcg @ 5: {:05.3f}; best dev precision_nn @ 5: {:05.3f}; best dev precision_2 @ 5: {:05.3f}'.format(test_ndcg, test_precision_nn, test_precision_2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data

    parser.add_argument('--build_data', dest='build_data', action='store_true',
                        help='if True, build data list from csv file')
    parser.add_argument('--len_token', dest='len_token', type=int, default='400',
                        help='the length of tokenize for each sentence')


    parser.add_argument('--train_file', dest='train_file', type=str,
                        default='/home/liu1769/scratch/data_eng__french/train_l_1000.pkl', help='path to training file')
    parser.add_argument('--train_ul_file', dest='train_ul_file', type=str,
                        default='/home/liu1769/scratch/data_eng__french/train_ul_1000.pkl', help='path to unlabelled training file')

    parser.add_argument('--dev_file', dest='dev_file', type=str,
                        default='/home/liu1769/scratch/data_eng__french/dev_2000.pkl', help='path to development file')
    parser.add_argument('--test_file', dest='test_file', type=str,
                        default='/home/liu1769/scratch/data_eng__french/test_2000.pkl', help='path to test file')

    parser.add_argument('--q_vocab_path', dest='q_vocab_path', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/vocab_en.pkl',
                        help='path to vocabulary for queries')
    parser.add_argument('--d_vocab_path', dest='d_vocab_path', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/vocab_fr.pkl',
                        help='path to vocabulary embedding for documents')
    parser.add_argument('--q_extn_embedding', dest='q_extn_embedding', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/polyglot_en_dict.pkl',
                        help='path to pre-trained embedding for queries')
    parser.add_argument('--d_extn_embedding', dest='d_extn_embedding', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/polyglot_fr_dict.pkl',
                        help='path to pre-trained embedding for documents')


    """
    parser.add_argument('--train_file', dest='train_file', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/toy_eng_data/train_l.pkl', help='path to training file')
    parser.add_argument('--train_ul_file', dest='train_ul_file', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/toy_eng_data/train_ul.pkl', help='path to unlabelled training file')
    parser.add_argument('--dev_file', dest='dev_file', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/toy_eng_data/dev.pkl', help='path to development file')
    parser.add_argument('--test_file', dest='test_file', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/toy_eng_data/test.pkl', help='path to test file')

    parser.add_argument('--q_vocab_path', dest='q_vocab_path', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/vocab_en.pkl', help='path to vocabulary for queries')
    parser.add_argument('--d_vocab_path', dest='d_vocab_path', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/vocab_fr.pkl', help='path to vocabulary embedding for documents')
    parser.add_argument('--q_extn_embedding', dest='q_extn_embedding', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/polyglot_en_dict.pkl', help='path to pre-trained embedding for queries')
    parser.add_argument('--d_extn_embedding', dest='d_extn_embedding', type=str, default='/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/polyglot_fr_dict.pkl', help='path to pre-trained embedding for documents')
    """
    # embeddings

    parser.add_argument('--create_vocabulary', action='store_true', help='')
    parser.add_argument('--vocab_size', dest='vocab_size', type=int, default='100000', help='size of vocabulary')

    # training
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='whether to use gpu')
    parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='number of epochs to run')
    parser.add_argument('--train_batchsize', dest='train_batchsize', type=int, default=128,
                        help='training minibatch size')
    parser.add_argument('--test_batchsize', dest='test_batchsize', type=int, default=128, help='testing minibatch size')
    parser.add_argument('--is_shuffle', dest='is_shuffle', action='store_true',
                        help='whether to shuffle training set each epoch')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default="adam")
    parser.add_argument('--lr', dest='lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--clip_grad', dest='clip_grad', type=float, default=5.0, help='clip grad at')
    parser.add_argument('--caseless', dest='caseless', action='store_true', help='caseless or not')
    parser.add_argument('--fine_tune', dest='fine_tune', action='store_true',
                        help='whether to fine tune the word embedding or not')
    parser.add_argument('--dropout', dest='dropout', action='store_true', help='whether use dropout in embedding layer')
    parser.add_argument('--patience', type=int, default=50, help='patience for early stop')

    # neural model
    parser.add_argument('--deep', action='store_true', help='')
    parser.add_argument('--lstm', action='store_true', help='whether use LSTM')

    parser.add_argument('--n_hdim', dest='n_hdim', type=int, default=200, help='dimension of hidden layer')
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=64, help='dimension of the embedding')

    # loss function
    parser.add_argument('--theta', dest='theta', nargs='*', type=float, default=[0.2, 0.9],
                        help='thresholds of loss function ')
    parser.add_argument('--sps_theta', dest='sps_theta', nargs='*', type=float, default=[0.2, 0.55, 0.9],
                        help='thresholds of loss function in Semantic Product Search paper')
    parser.add_argument('--use_sps', dest='use_sps', action='store_true',
                        help='whether use the loss from Semantic Product Search paper')
    parser.add_argument('--m', dest='m', type=int, default=1,
                        help='the order of the loss from Semantic Product Search paper')
    parser.add_argument('--use_cross', dest='use_cross', action='store_true',
                        help='whether use the cross entropy loss')
    # save
    parser.add_argument('--model_path', dest='model_path', type=str, default='')

    # adversarial training
    parser.add_argument('--vat', dest='vat', action='store_true',
                        help='whether use semi-supervised virtual adversarial training')
    parser.add_argument('--use_adv', dest='use_adv', action='store_true',
                        help='whether use supervised adversarial training')
    parser.add_argument('--adv_iter', dest='adv_iter', type=int, default=1,
                        help='maximum iteration when generating adversarial examples')
    parser.add_argument('--n_power', dest='n_power', type=int, default=1,
                        help='power iterations in virtual adversarial training')
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.01,
                        help='movement multiplier per iteration when generating adversarial examples')

    args = parser.parse_args()
    run(args)


