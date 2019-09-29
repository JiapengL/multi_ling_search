"""Two class margin based model
Author: Cosmo Zhang
"""

# -*- coding: utf-8 -*-
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional  as F


def embed_seq_batch(embed, seq_batch, dropout=0.):
    xs_f = []
    for x in seq_batch:
        x = embed(x)
        x = F.dropout(x, ratio=dropout)
        xs_f.append(x)
    return xs_f

class SimpleDSSM(nn.Module):
    def __init__(self, args, n_vocab_q, n_vocab_d):
        super(SimpleDSSM, self).__init__()

        self.n_hdim = args.n_hdim
        self.embed_dim = args.embed_dim
        self.n_vocab_q = n_vocab_q
        self.n_vocab_d = n_vocab_d
        self.encode_type = args.encode_type
        self.weighted_sum = args.weighted_sum
        self.cnn_out_channels = args.cnn_out_channels
        self.cnn_ksize = args.cnn_ksize

        # Internal structures

        # embedding of q
        self.q_word_embeds = nn.Embedding(q_vocab_size, embedding_dim)
        # embedding of d
        self.d_word_embeds = nn.Embedding(d_vocab_size, embedding_dim)

        # q_lstm
        self.q_word_lstm = nn.LSTM(embedding_dim, word_hidden_dim // 2, bidirectional=True)
        # d_lstm
        self.d_word_lstm = nn.LSTM(embedding_dim, word_hidden_dim // 2, bidirectional=True)

        """
        to_do: implement cnn
        # q_cnn
        self.q_convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim,
                                              out_channels = n_filters,
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])

        # d_cnn
        self.d_convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = embedding_dim,
                                              out_channels = n_filters,
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])
        """

    def cal_con_sim(self, x, y):
        """
        Input:
            x, bs * q_rep_dim
            y, bs * d_rep_dim

        """

        norm_x = F.normalize(x, p=2, dim=1)
        norm_y = F.normalize(y, p=2, dim=1)

        ret = torch.einsum('bs,bs->b', norm_x, norm_y)

        return F.batch_matmul(norm_x, norm_y, transa=True)

    def init_parameters(self):
        pass

    def load_embeddings(self, args, vocab, _type):
        # load query language embeding and doc language embedding as model parameters

        if _type == "query":
            vec_path = args.q_embed_file
        elif _type == "document":
            vec_path = args.d_embed_file
        else:
            assert False

        temp_tensor = torch.randn(len(vocab), args.embedding_dim)

        print "loading " +_type + " embeddings..."
        with open(vec_path, "r") as fi:
            for n, line in enumerate(fi):
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
        print "Successfully loaded embeddings".

    def forward(self, xs1, xs2, xs3):
        """
        xs1: q
        xs2: pos_d
        xs3: neg_d
        """



        '''
        convert numpy array to embeddings, bs * len * dim
        '''
        q_input = self.q_word_embeds.forward(xs1)

        d_p_input = self.d_word_embeds.forward(xs2)

        d_n_input = self.d_word_embeds.forward(xs3)

        if self.encode_type == "cnn":
            pass

        elif self.encode_type == "lstm":
            q_lstm_out, _ = self.word_lstm.forward(q_input)
            d_p_lstm_out, _ = self.word_lstm.forward(d_p_input)
            d_n_lstm_out, _ = self.word_lstm.forward(d_n_input)

        elif self.encode_type == "avg":
            q_rep = torch.mean(q_input, dim=1)
            d_p_rep = torch.mean(d_p_input, dim=1)
            d_n_emb = torch.mean(d_n_input, dim=1)

            '''
            to_do: weighted average, use weight as parameters for each token
            '''

        else:
            sys.exit("Error: encode_type is invalid")

        # calculate similarity
        cos_sims_rel = self.cal_con_sim(q_rep, d_p_rep)
        cos_sims_nonrel = self.cal_con_sim(q_rep, d_n_emb)
        gap = cos_sims_rel - cos_sims_nonrel


