"""Two class margin based model
Author: Cosmo Zhang
"""

# -*- coding: utf-8 -*-
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional  as F


class SimpleDSSM(nn.Module):
    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(SimpleDSSM, self).__init__()

        self.n_hdim = args.n_hdim
        self.embed_dim = args.embed_dim
        self.args = args

        # Internal structures

        # embedding of q
        self.q_word_embeds = nn.Embedding(q_vocab_size, self.embed_dim)
        # embedding of d
        self.d_word_embeds = nn.Embedding(d_vocab_size, self.embed_dim)

    def cal_sim(self, x, y):
        """
        Input:
            x, bs * q_rep_dim
            y, bs * d_rep_dim

        """

        norm_x = F.normalize(x, p=2, dim=1)
        norm_y = F.normalize(y, p=2, dim=1)

        ret = torch.einsum('bs,bs->b', norm_x, norm_y)

        return ret

    def init_parameters(self):
        pass

    def load_embeddings(self, args, vocab, _type):
        # load query language embeding and doc language embedding as model parameters

        if _type == "query":
            vec_path = args.q_embed_file
        elif _type == "document":
            vec_path = args.d_embed_file
        else:
            sys.exit("Unknow embedding type")

        temp_tensor = torch.randn(len(vocab), args.embedding_dim)

        print("loading " +_type + " embeddings...")
        with open(vec_path, "r") as fi:
            for n, line in enumerate(fi.readlines()):
                # 1st line contains stats
                if n == 0:
                    continue
                line_list = line.strip().split(" ", 1)
                word = line_list[0].lower() if args.caseless else line_list[0]
                if word in vocab:
                    value = line.strip().split(" ")[1::]
                    vec =  np.fromstring(value, dtype=float, sep=' ')
                    temp_tensor[vocab[word]] = nn.Parameter(torch.from_numpy(vec))

        if _type == "query":
            self.q_word_embeds = nn.Embedding.from_pretrained(temp_tensor)
        elif _type == "document":
            self.d_word_embeds = nn.Embedding.from_pretrained(temp_tensor)
        print("Successfully loaded embeddings.")

    def forward(self, qs, ds, rels):

        qs_input = self.q_word_embeds.forward(qs)
        ds_input = self.d_word_embeds.forward(ds)

        q_rep = torch.mean(qs_input, dim=1)
        d_rep = torch.mean(ds_input, dim=1)

        sims = self.cal_sim(q_rep, d_rep)

        return sims


    def predict(self, qs, ds):
        pass


    def cal_loss(self):
        pass

class DeepDSSM(SimpleDSSM):

    def __init__(self, args):
        super(DeepDSSM, self).__init__()
