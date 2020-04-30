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
from src.train_function import train_model, eval_model, test_model, eval_cross_model,  test_cross_model
import src.evaluator as eval

import pdb
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
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu")
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

    best_f1_nn = 0
    best_total_accuracy = 0
    best_class_accuracy = list(0. for i in range(3))
    best_class_precision = list(0. for i in range(3))
    best_class_f1 = list(0. for i in range(3))

    # training and evaluating without unlabelled data
    train_flag = False
    patience_count = 0
    for epoch in range(1, 1 + args.epochs):


        # model training
        train_model(args, model, optimizer, data_processor, vocab_q, vocab_d, device, epoch)
        # model evaluation
        dev_total_accuracy, dev_class_accuracy, dev_class_precision, dev_class_f1, dev_f1_nn = eval_cross_model(args, model, data_processor, vocab_q, vocab_d, device, epoch, dev_qd_index, dev_rels_index)

        # test
        if epoch > 5 or args.epochs < 6:
            if dev_f1_nn > best_f1_nn:
                train_flag = True
                patience_count = 0
                best_f1_nn = dev_f1_nn
                best_total_accuracy = dev_total_accuracy
                best_class_accuracy = dev_class_accuracy
                best_class_precision = dev_class_precision
                best_class_f1 = dev_class_f1


                # evaluate on test set
                test_total_accuracy, test_class_accuracy, test_class_precision, test_class_f1, test_f1_nn = test_cross_model(args, model, data_processor, vocab_q, vocab_d, device, epoch, test_qd_index, test_rels_index)

            else:
                patience_count += 1
            if patience_count >= args.patience:
                break

    if train_flag:
        print(
            'Epoch {}: best_dev total_accuracy: {:05.3f}; best_dev class_2_f1: {:05.3f}; best_dev class_1_f1: {:05.3f}; best_dev_class_nn_f1:  {:05.3f}'
                .format(epoch, best_total_accuracy, best_class_f1[2], best_class_f1[1], best_f1_nn))

        print(
            'Epoch {}: best_test total_accuracy: {:05.3f}; best_test class_2_f1: {:05.3f}; best_test class_1_f1: {:05.3f}; best_test_class_nn_f1:  {:05.3f}'
            .format(epoch, test_total_accuracy, test_class_f1[2], test_class_f1[1], test_f1_nn))

        #torch.save(best_sims, "laplace_11")
        #flat_index = [item for sublist in dev_rels_index for item in sublist]
        #torch.save(flat_index, "baseline_index")
    else:
        print('No enough regular train')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--build_data', dest='build_data', action='store_true',
                        help='if True, build data list from csv file')
    parser.add_argument('--len_token', dest='len_token', type=int, default='400',
                        help='the length of tokenize for each sentence')

    parser.add_argument('--train_file', dest='train_file', type=str,
                        default='/home/liu1769/scratch/data_eng__swahili/train_15.pkl', help='path to training file')
    parser.add_argument('--train_ul_file', dest='train_ul_file', type=str,
                        default='/home/liu1769/scratch/data_eng__french/train_ul_100.pkl',
                        help='path to unlabelled training file')

    parser.add_argument('--dev_file', dest='dev_file', type=str,
                        default='/home/liu1769/scratch/data_eng__swahili/dev_15.pkl', help='path to development file')
    parser.add_argument('--test_file', dest='test_file', type=str,
                        default='/home/liu1769/scratch/data_eng__swahili/test_15.pkl', help='path to test file')

    parser.add_argument('--q_vocab_path', dest='q_vocab_path', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/vocab_en.pkl',
                        help='path to vocabulary for queries')
    parser.add_argument('--d_vocab_path', dest='d_vocab_path', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/vocab_sw.pkl',
                        help='path to vocabulary embedding for documents')
    parser.add_argument('--q_extn_embedding', dest='q_extn_embedding', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/polyglot_en_dict.pkl',
                        help='path to pre-trained embedding for queries')
    parser.add_argument('--d_extn_embedding', dest='d_extn_embedding', type=str,
                        default='/home/liu1769/multi_ling_search/word_embed/polyglot_sw_dict.pkl',
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
    parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='number of epochs to run')
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
    parser.add_argument('--batchnorm', dest='batchnorm', action='store_true', help='whether use batchnorm')

    # neural model
    parser.add_argument('--deep', action='store_true', help='')
    parser.add_argument('--lstm', action='store_true', help='whether use LSTM')
    parser.add_argument('--n_hdim', dest='n_hdim', type=int, default=200, help='dimension of hidden layer')
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=64, help='dimension of the embedding')

    # self training
    parser.add_argument('--self', dest='self', action='store_true', help='whether do self training')
    parser.add_argument('--self_epochs_in', dest='self_epochs_in', type=int, default=1, help='number of epochs in the inner loop to run during self training')
    parser.add_argument('--self_epochs_out', dest='self_epochs_out', type=int, default=5, help='number of epochs in the outer loop to run during self training')
    parser.add_argument('--truncate', dest='truncate', type=float, default=0, help='whether to add thresholding during self training')

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
    parser.add_argument('--leaky_loss', dest='leaky_loss', action='store_true',
                        help='whether use the leaky loss for self training')
    parser.add_argument('--im_loss', dest='im_loss', action='store_true',
                        help='whether use the immediate threshold loss for self training')
    parser.add_argument('--laplace', dest='laplace', action='store_true',
                        help='whether use the laplace loss for self training')
    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.05,
                        help='epsilon for leaky loss function')
    parser.add_argument('--combine', dest='combine', action='store_true',
                        help='whether use the combine loss for self training')
    parser.add_argument('--pol', dest='pol', action='store_true',
                        help='whether use the proportional odds model with laplace')
    parser.add_argument('--pos', dest='pos', action='store_true',
                        help='whether use the proportional odds model with sigmoid')
    parser.add_argument('--pos_trun', dest='pos_trun', action='store_true',
                        help='whether use the proportional odds model with sigmoid')
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


