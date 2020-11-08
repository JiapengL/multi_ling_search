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


def train_model(args, model, optimizer, data_processor, vocab_q, vocab_d, device, epoch, adv_stage=False):
    """
    model training
    """

    model.train()
    loss_total = []
    i = 0
    for batch in data_processor.generate_batch(args.train_batchsize,  args.is_shuffle, dataset='train'):
        #  numerize labeled data
        rels, qs, ds = numerize(batch, vocab_q, vocab_d)
        if args.use_gpu:
            rels, qs, ds = rels.to(device), qs.to(device), ds.to(device)

        model.zero_grad()
        sims_l = model(qs, ds)
        loss = model.cal_loss(sims_l, rels)

        if adv_stage:
            loss.backward(retain_graph=True)
            q_grads = torch.autograd.grad(loss, model.qs_emb, retain_graph=True)[0]
            d_grads = torch.autograd.grad(loss, model.ds_emb, retain_graph=True)[0]

            sims = model(qs, ds, q_grads, d_grads)
            loss_adv = model.cal_loss(sims, rels)
            loss += 0.5 * loss_adv

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        loss_total.append(loss.item())

        i += 1
    if epoch % 5 == 0:
        print('The training loss at Epoch ', epoch, 'is {:05.5f}'.format(np.mean(loss_total)))





def train_ul_model(args, model, prediction, optimizer, data_processor, vocab_q, vocab_d, device, epoch, label=None):
    """
    model training with unlabeled data
    label indicates whether the data will be kept when using truncating
    """

    model.train()
    loss_ul_total = []
    i = 0

    for rels, batch in data_processor.generate_ul_batch(args.train_batchsize, prediction, label, args.is_shuffle):
        #  numerize labeled data
        _, qs, ds = numerize(batch, vocab_q, vocab_d)
        if args.is_shuffle:
            rels = torch.IntTensor(rels)
        if args.use_gpu:
            rels, qs, ds = rels.to(device), qs.to(device), ds.to(device)
        model.zero_grad()
        sims_l = model(qs, ds)
        loss = model.cal_loss(sims_l, rels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()
        loss_ul_total.append(loss.item())

        i += 1

    if epoch % 5 == 0:
        print('The training loss of unlabeled data at Epoch ', epoch, 'is {:05.5f}'.format(np.mean(loss_ul_total)))







def eval_model(args, model, data_processor, vocab_q, vocab_d, device, epoch, dev_qd_index, dev_rels_index):
    """
    model evaluation
    """

    model.eval()
    loss_dev = []
    sims_dev = torch.tensor([]).to(device) if args.use_gpu else torch.tensor([])
    for dev_batch in data_processor.generate_batch(args.train_batchsize, is_shuffle=False, dataset="dev"):

        rels_dev, qs_dev, ds_dev = numerize(dev_batch, vocab_q, vocab_d)
        if args.use_gpu:
            rels_dev, qs_dev, ds_dev = rels_dev.to(device), qs_dev.to(device), ds_dev.to(device)

        sims = model(qs_dev, ds_dev)
        sims_dev = torch.cat([sims_dev, sims.data])

        loss = model.cal_loss(sims, rels_dev)
        loss_dev.append(loss.item())

    # evaluation
    dev_map, dev_mrr_nn, dev_mrr_2 = eval.map_mrr(sims_dev, dev_qd_index, dev_rels_index)
    dev_precision_nn, dev_precision_2, dev_precision_21 = eval.precision_at_k(sims_dev, dev_qd_index, dev_rels_index, 5)
    dev_ndcg = eval.ndcg_at_k(sims_dev, dev_qd_index, dev_rels_index, 5)

    if epoch % 5 == 0:
        print(
            'Epoch {}: dev loss: {:05.5f}; dev precision_2  @ 1: {:05.3f}; dev precision_2  @ 5: {:05.3f}; dev precision_nn  @ 5: {:05.3f}; dev ndcg  @ 5: {:05.3f}'
                .format(epoch, np.mean(loss_dev), dev_precision_21, dev_precision_2, dev_precision_nn, dev_ndcg))
        print(
            'Epoch {}: dev map: {:05.3f}; dev mrr_nn: {:05.3f}; dev mrr_2: {:05.3f}'
                .format(epoch, dev_map, dev_mrr_nn, dev_mrr_2))
        # prediction matrix
        """
        prediction = eval.predict(sims_dev, args.theta)
        pre_dict = eval.prediction_matrix(prediction, dev_rels_index)
        print('The prediction matrix is following, rows are rels and columns are prediction:')
        print(pre_dict)
        """
    """
    if epoch > 10:
        scheduler.step()
    """
    return dev_precision_nn, dev_precision_2, dev_precision_21, dev_ndcg, dev_map, dev_mrr_nn, dev_mrr_2, sims_dev











def train_model_result(args, model, data_processor, vocab_q, vocab_d, device, epoch, qd_index, rels_index):
    """
    model evaluation
    """

    model.eval()
    loss_dev = []
    sims_dev = torch.tensor([]).to(device) if args.use_gpu else torch.tensor([])
    for dev_batch in data_processor.generate_batch(args.train_batchsize, is_shuffle=False, dataset="train_2"):

        rels_dev, qs_dev, ds_dev = numerize(dev_batch, vocab_q, vocab_d)
        if args.use_gpu:
            rels_dev, qs_dev, ds_dev = rels_dev.to(device), qs_dev.to(device), ds_dev.to(device)

        sims = model(qs_dev, ds_dev)
        sims_dev = torch.cat([sims_dev, sims.data])

        loss = model.cal_loss(sims, rels_dev)
        loss_dev.append(loss.item())

    # evaluation
    dev_map, dev_mrr_nn, dev_mrr_2 = eval.map_mrr(sims_dev, qd_index, rels_index)
    dev_precision_nn, dev_precision_2, dev_precision_21 = eval.precision_at_k(sims_dev, qd_index, rels_index, 5)
    dev_ndcg = eval.ndcg_at_k(sims_dev, qd_index, rels_index, 5)

    if epoch % 5 == 0:
        print(
            'Epoch {}: train loss: {:05.5f}; train precision_2  @ 1: {:05.3f}; train precision_2  @ 5: {:05.3f}; train precision_nn  @ 5: {:05.3f}; train ndcg  @ 5: {:05.3f}'
                .format(epoch, np.mean(loss_dev), dev_precision_21, dev_precision_2, dev_precision_nn, dev_ndcg))
        print(
            'Epoch {}: train map: {:05.3f}; train mrr_nn: {:05.3f}; train mrr_2: {:05.3f}'
                .format(epoch, dev_map, dev_mrr_nn, dev_mrr_2))
        # prediction matrix
    return  sims_dev






def test_model(args, model, data_processor, vocab_q, vocab_d, device, epoch, test_qd_index, test_rels_index):
    """
    model testing
    """

    model.eval()
    loss_test = []
    sims_test = torch.tensor([]).to(device) if args.use_gpu else torch.tensor([])
    for test_batch in data_processor.generate_batch(args.train_batchsize, is_shuffle=False, dataset="test"):

        rels_test, qs_test, ds_test = numerize(test_batch, vocab_q, vocab_d)
        if args.use_gpu:
            rels_test, qs_test, ds_test = rels_test.to(device), qs_test.to(device), ds_test.to(device)

        sims = model(qs_test, ds_test)
        sims_test = torch.cat([sims_test, sims.data])

        loss = model.cal_loss(sims, rels_test)
        loss_test.append(loss.item())


    test_map, test_mrr_nn, test_mrr_2 = eval.map_mrr(sims_test, test_qd_index, test_rels_index)
    test_precision_nn, test_precision_2, test_precision_21 = eval.precision_at_k(sims_test, test_qd_index, test_rels_index, 5)
    test_ndcg = eval.ndcg_at_k(sims_test, test_qd_index, test_rels_index, 5)
    if epoch % 5 == 0:
        print(
            'Epoch {}: test loss: {:05.5f}; test precision_2  @ 1: {:05.3f}; test precision_2  @ 5: {:05.3f}; test precision_nn  @ 5: {:05.3f}; test ndcg  @ 5: {:05.3f}'
                .format(epoch, np.mean(loss_test), test_precision_21, test_precision_2, test_precision_nn, test_ndcg))
        print(
            'Epoch {}: test map: {:05.3f}; test mrr_nn: {:05.3f}; test mrr_2: {:05.3f}'
                .format(epoch, test_map, test_mrr_nn, test_mrr_2))
    return test_precision_nn, test_precision_2, test_precision_21, test_ndcg, test_map, test_mrr_nn, test_mrr_2







def eval_cross_model(args, model, data_processor, vocab_q, vocab_d, device, epoch, dev_qd_index, dev_rels_index):
    """
    model evaluation
    """

    model.eval()
    loss_dev = []
    sims_dev = torch.tensor([]).to(device) if args.use_gpu else torch.tensor([])
    for dev_batch in data_processor.generate_batch(args.train_batchsize, is_shuffle=False, dataset="dev"):

        rels_dev, qs_dev, ds_dev = numerize(dev_batch, vocab_q, vocab_d)
        if args.use_gpu:
            rels_dev, qs_dev, ds_dev = rels_dev.to(device), qs_dev.to(device), ds_dev.to(device)

        sims = model(qs_dev, ds_dev)
        sims_dev = torch.cat([sims_dev, sims.data])
        loss = model.cal_loss(sims, rels_dev)
        loss_dev.append(loss.item())

    if args.use_cross:
        pred = eval.predict_cross(sims_dev)
    else:
        pred = eval.predict(sims_dev, args.theta)

    dev_label = torch.LongTensor([item for sublist in dev_rels_index for item in sublist]).to(device)   # evaluation
    class_total, class_correct, class_pred = eval.prediction_accuracy(pred, dev_label)
    total_accuracy = sum(class_correct)/sum(class_total)
    class_accuracy = list(0. for i in range(3))
    class_precision = list(0. for i in range(3))
    class_f1 = list(0. for i in range(3))
    for i in range(3):
        class_accuracy[i] = class_correct[i]/class_total[i]
        class_precision[i] = class_correct[i]/(class_pred[i]+0.001)
        class_f1[i] = 2 * class_accuracy[i] * class_precision[i]/(class_accuracy[i] + class_precision[i] + 0.001)
    f1_nn = (class_f1[1] + class_f1[2])/2

    if epoch % 5 == 0:
        print(
            'Epoch {}: dev loss: {:05.5f}; dev total_accuracy: {:05.3f}; dev class_2_f1: {:05.3f}; dev class_1_f1: {:05.3f}; dev_class_nn_f1:  {:05.3f}'
            .format(epoch, np.mean(loss_dev), total_accuracy, class_f1[2], class_f1[1], f1_nn))

    return total_accuracy, class_accuracy, class_precision, class_f1, f1_nn




def test_cross_model(args, model, data_processor, vocab_q, vocab_d, device, epoch, test_qd_index, test_rels_index):
    """
    model evaluation
    """

    model.eval()
    loss_test = []
    sims_test = torch.tensor([]).to(device) if args.use_gpu else torch.tensor([])
    for test_batch in data_processor.generate_batch(args.train_batchsize, is_shuffle=False, dataset="test"):

        rels_test, qs_test, ds_test = numerize(test_batch, vocab_q, vocab_d)
        if args.use_gpu:
            rels_test, qs_test, ds_test = rels_test.to(device), qs_test.to(device), ds_test.to(device)

        sims = model(qs_test, ds_test)
        sims_test = torch.cat([sims_test, sims.data])

        loss = model.cal_loss(sims, rels_test)
        loss_test.append(loss.item())

    if args.use_cross:
        pred = eval.predict_cross(sims_test)
    else:
        pred = eval.predict(sims_test, args.theta)

    test_label = torch.LongTensor([item for sublist in test_rels_index for item in sublist]).to(device)   # evaluation

    class_total, class_correct, class_pred = eval.prediction_accuracy(pred, test_label)
    total_accuracy = sum(class_correct)/sum(class_total)
    class_accuracy = list(0. for i in range(3))
    class_precision = list(0. for i in range(3))
    class_f1 = list(0. for i in range(3))
    for i in range(3):
        class_accuracy[i] = class_correct[i]/class_total[i]
        class_precision[i] = class_correct[i]/(class_pred[i]+0.001)
        class_f1[i] = 2 * class_accuracy[i] * class_precision[i]/(class_accuracy[i] + class_precision[i] + 0.001)
    f1_nn = (class_f1[1] + class_f1[2])/2

    if epoch % 5 == 0:
        print(
            'Epoch {}: test loss: {:05.5f}; test total_accuracy: {:05.3f}; test class_2_f1: {:05.3f}; test class_1_f1: {:05.3f}; test_class_nn_f1:  {:05.3f}'
            .format(epoch, np.mean(loss_test), total_accuracy, class_f1[2], class_f1[1], f1_nn))

    return total_accuracy, class_accuracy, class_precision, class_f1, f1_nn










