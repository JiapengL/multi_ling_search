"""Two class margin based model
Author: Cosmo Zhang
"""

# -*- coding: utf-8 -*-
import sys
import pickle as pkl

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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Internal structures

        # embedding of q, the input of embedding should be index tensor
        self.q_word_embeds = nn.Embedding(q_vocab_size, self.embed_dim).to(self.device)
        # embedding of d
        self.d_word_embeds = nn.Embedding(d_vocab_size, self.embed_dim).to(self.device)

    def init_parameters(self):
        pass


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

        if _type == "query":
            vec_path = args.q_extn_embedding
        elif _type == "document":
            vec_path = args.d_extn_embedding
        else:
            sys.exit("Unknown embedding type")

        temp_tensor = torch.randn(len(vocab), args.embed_dim)

        if vec_path:
            print("loading " +_type+ " embeddings...")
            if vec_path.endswith('.txt'):
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
                            temp_tensor[vocab[word]] = torch.from_numpy(vec)
            elif vec_path.endswith('.pkl'):
                with open(vec_path, 'rb') as f:
                    words, vecs = pkl.load(f)
                    for word, vec in zip(words, vecs):
                        word = word.lower() if args.caseless else word
                        if word in vocab:
                            temp_tensor[vocab[word]] = torch.from_numpy(vec)


        if _type == "query":
            if self.args.fine_tune:
                self.q_word_embeds = nn.Embedding.from_pretrained(temp_tensor, freeze=False).to(self.device)
            else:
                self.q_word_embeds = nn.Embedding.from_pretrained(temp_tensor, freeze=False).to(self.device)
        elif _type == "document":
            if self.args.fine_tune:
                self.d_word_embeds = nn.Embedding.from_pretrained(temp_tensor, freeze=False).to(self.device)
            else:
                self.d_word_embeds = nn.Embedding.from_pretrained(temp_tensor, freeze=False).to(self.device)

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

        q_rep = torch.tanh(torch.mean(qs_input, dim=1))
        d_rep = torch.tanh(torch.mean(ds_input, dim=1))

        sims = self.cal_sim(q_rep, d_rep)

        return q_rep.shape, d_rep.shape, sims


    def predict(self, sims):
        """
        :param sims: similarity derived from the model
        :return: predicted relevance score
        """

        return torch.gt(sims, self.theta[1]).int() + torch.gt(sims, self.theta[0]).int()


    def __threshold(self, rels, index):
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

        loss_theta1 = F.relu(self.__threshold(rels, 1) * (self.theta[0] - sims))
        loss_theta2 = F.relu(self.__threshold(rels, 2) * (self.theta[1] - sims))

        return torch.mean(loss_theta1 + loss_theta2)



class DeepDSSM(SimpleDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(DeepDSSM, self).__init__(args, q_vocab_size, d_vocab_size)

        # layers for query
        self.Embedding_size = 64
        self.WINDOW_SIZE = 3
        self.TOTAL_LETTER_GRAMS = int(1 * 1e5)
        self.WORD_DEPTH = self.WINDOW_SIZE * self.Embedding_size
        self.K = 300
        self.q_conv = nn.Conv1d(self.WORD_DEPTH, self.K, 1)
        self.d_conv = nn.Conv1d(self.WORD_DEPTH, self.K, 1)
        if torch.cuda.device_count() > 1:
            self.q_conv = nn.DataParallel(self.q_conv)
            self.d_conv = nn.DataParallel(self.d_conv)
#        self.sem = nn.Linear(K, L)


    def generate_n_gram(self, word_tensor):
        """
        calculate the input vector for n_gram, with windows_size = self.WINDOW_SIZE
        """

        sent_len = word_tensor.shape[1]
        temp = word_tensor[:, :sent_len - self.WINDOW_SIZE + 1, :]

        for i in range(1, self.WINDOW_SIZE):
            temp = torch.cat((temp, word_tensor[:, i:(sent_len - self.WINDOW_SIZE + i + 1), :]), dim=2)

        return temp.transpose(1, 2)


    def max_pooling(self, x):
        """
        maxpooling over the length of sentence dimension
        """

        return torch.squeeze(x.topk(1, dim=2)[0], dim = 2)



    def forward(self, qs, ds, rels):

        # qs_input shape: (batch, len_query, embedding_dim)
        # ds_input shape: (batch, len_doc, embedding_dim)

        qs_input = self.q_word_embeds.forward(qs)
        ds_input = self.d_word_embeds.forward(ds)

        qs_ngram = self.generate_n_gram(qs_input)
        ds_ngram = self.generate_n_gram(ds_input)

        qs_conv = torch.tanh(self.q_conv(qs_ngram))
        ds_conv = torch.tanh(self.d_conv(ds_ngram))

        qs_maxp = self.max_pooling(qs_conv)
        ds_maxp = self.max_pooling(ds_conv)

        sims = self.cal_sim(qs_maxp, ds_maxp)

        return sims
