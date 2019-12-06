import sys
import pickle as pkl

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from src.dssm_model import SimpleDSSM, DeepDSSM


class Adv_Simple(SimpleDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(Adv_Simple, self).__init__(args, q_vocab_size, d_vocab_size)

        # layers for query
        self.use_adv = args.use_adv
        self.alpha = args.alpha

    def forward(self, qs, ds, q_grads=None, d_grads=None):

        qs_input = self.q_word_embeds.forward(qs)
        ds_input = self.d_word_embeds.forward(ds)

        self.qs_input = qs_input
        self.ds_input = ds_input

        adv_flag = self.training and self.use_adv

        if adv_flag and q_grads is not None and d_grads is not None:
            qs_input += self.alpha * torch.sign(q_grads.data)
            ds_input += self.alpha * torch.sign(d_grads.data)

        q_rep = torch.tanh(torch.mean(qs_input, dim=1))
        d_rep = torch.tanh(torch.mean(ds_input, dim=1))

        sims = self.cal_sim(q_rep, d_rep)

        return sims



class Adv_Deep(DeepDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(Adv_Deep, self).__init__(args, q_vocab_size, d_vocab_size)

        # layers for query
        self.use_adv = args.use_adv
        self.alpha = args.alpha

    def forward(self, qs, ds, q_grads=None, d_grads=None):

        qs_input = self.q_word_embeds.forward(qs)
        ds_input = self.q_word_embeds.forward(ds)

        self.qs_input = qs_input
        self.ds_input = ds_input

        adv_flag = self.training and self.use_adv

        if adv_flag and q_grads is not None and d_grads is not None:
            qs_input += self.alpha * torch.sign(q_grads.data)
            ds_input += self.alpha * torch.sign(d_grads.data)

        qs_ngram, ds_ngram = self.generate_n_gram(qs_input), self.generate_n_gram(ds_input)

        qs_conv, ds_conv = torch.tanh(self.q_conv(qs_ngram)), torch.tanh(self.q_conv(ds_ngram))

        qs_maxp, ds_maxp = self.max_pooling(qs_conv), self.max_pooling(ds_conv)

        qs_drop, ds_drop = self.drop_out(qs_maxp),  self.drop_out(ds_maxp)

        qs_sem, ds_sem = self.q_fc(qs_drop), self.q_fc(ds_drop)

        sims = self.cal_sim(qs_sem, ds_sem)

        return sims

