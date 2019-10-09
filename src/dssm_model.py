"""Two class margin based model
Author: Cosmo Zhang
"""

# -*- coding: utf-8 -*-
import sys

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDSSM(nn.Module):
    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(SimpleDSSM, self).__init__()

        self.n_hdim = args.n_hdim
        self.embed_dim = args.embed_dim
        self.theta = args.theta
        self.args = args

        # Internal structures

        # embedding of q, the input of embedding should be index tensor
        self.q_word_embeds = nn.Embedding(q_vocab_size, self.embed_dim)
        # embedding of d
        self.d_word_embeds = nn.Embedding(d_vocab_size, self.embed_dim)

    def init_parameters(self):
        pass


    def load_vocab(self, path, size):
        """
        load language vocabulary from files with given size
        """
        vocab = {}
        for i, w in enumerate(open(path, "r")):
            if i < size:
                vocab[w.strip()] = 1
        return vocab


    def load_embeddings(self, args, vocab, _type):
        """
        load query language embedding and doc language embedding as model parameters

        args:
            args: argparser object
            vocab: vocab dictionary
            _type: queries or documents
        Return:
            None
        """
        if args.extn_embedding:
             with open(args.q_extn_embedding, "rb") as f_q, \
                  open(args.d_extn_embedding, "rb") as f_d:
                self.q_word_embeds = pickle.load(f_q,  encoding='latin1')
                self.q_word_embeds = pickle.load(f_d, encoding='latin1')

        else:
            if _type == "query":
                vec_path = args.q_embed_file
            elif _type == "document":
                vec_path = args.d_embed_file
            else:
                sys.exit("Unknown embedding type")

            temp_tensor = torch.randn(len(vocab), args.embedding_dim)

            print("loading " +_type+ " embeddings...")
            with open(vec_path, "r") as fi:
                for n, line in enumerate(fi.readlines()):
                    # 1st line contains stats
                    if n == 0:
                        continue
                    line_list = line.strip().split(" ", 1)
                    word = line_list[0].lower() if args.caseless else line_list[0]
                    if word in vocab:
                        value = line.strip().split(" ")[1::]
                        vec = np.fromstring(value, dtype=float, sep=' ')
                        temp_tensor[vocab[word]] = nn.Parameter(torch.from_numpy(vec))

            if _type == "query":
                self.q_word_embeds = nn.Embedding.from_pretrained(temp_tensor)
            elif _type == "document":
                self.d_word_embeds = nn.Embedding.from_pretrained(temp_tensor)

        print("Successfully loaded embeddings.")


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



    def forward(self, qs, ds, rels):

        qs_input = self.q_word_embeds.forward(qs)
        ds_input = self.d_word_embeds.forward(ds)

        q_rep = torch.mean(qs_input, dim=1)
        d_rep = torch.mean(ds_input, dim=1)

        sims = self.cal_sim(q_rep, d_rep)

        return sims


    def predict(self, sims):
        """
        :param sims: similarity derived from the model
        :return: predicted relevance score
        """

        return torch.gt(sims, self.theta[1]).int() + torch.gt(sims, self.theta[0]).int()


    def threshold(self, rels, index):
        """
        thresholding function for ordinal regression loss
        return 1 if rels >= index else -1
        """

        return(torch.gt(rels, index - 0.0001).float() - 0.5) * 2


    def cal_loss(self, sims, rels):
        """
        :param self.theta: thresholds of ordinal regression loss
        :param sims: similarity derived from the model
        :param rels: relevance score: 0, 1, 2
        :return: average loss
        """

        loss_theta1 = F.relu(self.threshold(rels, 1) * (self.theta[0] - sims))
        loss_theta2 = F.relu(self.threshold(rels, 2) * (self.theta[1] - sims))

        return torch.mean(loss_theta1 + loss_theta2)



class DeepDSSM(SimpleDSSM):

    def __init__(self, args):
        super(DeepDSSM, self).__init__()
