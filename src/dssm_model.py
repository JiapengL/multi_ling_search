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
import pdb


class SimpleDSSM(nn.Module):
    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(SimpleDSSM, self).__init__()

        self.n_hdim = args.n_hdim
        self.embed_dim = args.embed_dim
        self.theta = args.theta if not args.use_sps else args.sps_theta
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # embedding of q, the input of embedding should be index tensor
        self.q_word_embeds = nn.Embedding(q_vocab_size, self.embed_dim).to(self.device)
        # embedding of d
        self.d_word_embeds = nn.Embedding(d_vocab_size, self.embed_dim).to(self.device)
        self.out_dim = 32
        self.q_fc = nn.Linear(self.embed_dim, self.out_dim)
        self.d_fc = nn.Linear(self.embed_dim, self.out_dim)

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
                self.q_word_embeds = nn.Embedding.from_pretrained(temp_tensor).to(self.device)
            else:
                self.q_word_embeds = nn.Embedding.from_pretrained(temp_tensor, freeze=False).to(self.device)
        elif _type == "document":
            if self.args.fine_tune:
                self.d_word_embeds = nn.Embedding.from_pretrained(temp_tensor).to(self.device)
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


    def get_normalized_vector(self, d):
        # d needs to be a unit vector at each iteration

        d_norm = F.normalize(d.view(d.size(0), -1), dim=1, p=2).view(d.size())
        return d_norm


    def forward(self, qs, ds, q_perturb=None, d_perturb=None):

        qs_emb = self.q_word_embeds.forward(qs)
        ds_emb = self.d_word_embeds.forward(ds)

        # qs_emb = self.get_normalized_vector(qs_emb)
        # ds_emb = self.get_normalized_vector(ds_emb)

        self.qs_emb = qs_emb
        self.ds_emb = ds_emb

        adv_flag = self.training and self.args.vat

        if adv_flag and q_perturb is not None and d_perturb is not None:
            qs_emb += q_perturb
            ds_emb += d_perturb

        q_rep = torch.tanh(torch.mean(qs_emb, dim=1))
        d_rep = torch.tanh(torch.mean(ds_emb, dim=1))

        # q_rep = torch.mean(qs_emb, dim=1)
        # d_rep = torch.mean(ds_emb, dim=1)
       # q_o = self.q_fc(q_rep)
       # d_o = self.q_fc(d_rep)

        sims = self.cal_sim(q_rep, d_rep)

        return sims


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
        if not self.args.use_sps:
            # if using ordinal regression loss
            loss_theta1 = F.relu(self.__threshold(rels, 1) * (self.theta[0] - sims)).pow(self.args.m)
            loss_theta2 = F.relu(self.__threshold(rels, 2) * (self.theta[1] - sims)).pow(self.args.m)

            return torch.mean(loss_theta1 + loss_theta2)

        else:
            # if using Semantic product search paper loss
            loss_type0 = F.relu((rels == 0).float() * (sims - self.theta[0])).pow(self.args.m)
            loss_type1 = F.relu((rels == 1).float() * (sims - self.theta[1])).pow(self.args.m)
            loss_type2 = F.relu((rels == 2).float() * (self.theta[2] - sims)).pow(self.args.m)

            return torch.mean(loss_type0 + loss_type1 + loss_type2)



class DeepDSSM(SimpleDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(DeepDSSM, self).__init__(args, q_vocab_size, d_vocab_size)

        # layers for query
        self.WINDOW_SIZE = 1
        self.TOTAL_LETTER_GRAMS = int(1 * 1e5)
        self.WORD_DEPTH = self.WINDOW_SIZE * self.embed_dim
        self.K = 16   # size of filter
        self.L = 32    # size of output
        self.drop_out = nn.Dropout(p=0.5)
        self.q_conv = nn.Conv1d(self.WORD_DEPTH, self.K, 1)
        #self.d_conv = nn.Conv1d(self.WORD_DEPTH, self.K, 1)

        if torch.cuda.device_count() > 1:
            self.q_conv = nn.DataParallel(self.q_conv)
         #   self.d_conv = nn.DataParallel(self.d_conv)

        self.q_fc = nn.Linear(self.K, self.L)
        #self.d_fc = nn.Linear(self.K, self.L)

        self.dropout = nn.Dropout(p=0.5)


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

        return torch.squeeze(x.topk(1, dim=2)[0], dim=2)



    def forward(self, qs, ds):

        # qs_input shape: (batch, len_query, embedding_dim)
        # ds_input shape: (batch, len_doc, embedding_dim)

        qs_emb = self.q_word_embeds.forward(qs)
        ds_emb = self.q_word_embeds.forward(ds)

        if self.args.dropout:
            qs_emb = self.dropout(qs_emb)
            ds_emb = self.dropout(ds_emb)

        qs_ngram = self.generate_n_gram(qs_emb)
        ds_ngram = self.generate_n_gram(ds_emb)

        qs_conv = torch.tanh(self.q_conv(qs_ngram))
        ds_conv = torch.tanh(self.q_conv(ds_ngram))

        #qs_maxp = self.max_pooling(qs_conv)
        #ds_maxp = self.max_pooling(ds_conv)
        qs_maxp = torch.mean(qs_conv, dim=2)
        ds_maxp = torch.mean(ds_conv, dim=2)

        qs_drop = self.drop_out(qs_maxp)
        ds_drop = self.drop_out(ds_maxp)

        qs_sem = self.q_fc(qs_drop)
        ds_sem = self.q_fc(ds_drop)
        # pdb.set_trace()
        sims = self.cal_sim(qs_sem, ds_sem)

        return sims









class LSTM(SimpleDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(LSTM, self).__init__(args, q_vocab_size, d_vocab_size)

        # layers for query
        self.hidden_size = 256
        self.bilstm = nn.LSTM(self.embed_dim, self.hidden_size,
                              batch_first=True)


        self.out_size = 64
        self.lin = nn.Linear(self.hidden_size, self.out_size)


    def forward(self, qs, ds):

        qs_input = self.q_word_embeds.forward(qs)
        ds_input = self.q_word_embeds.forward(ds)
        # size = [batch, len, embed_size]
        self.bilstm.flatten_parameters()

        qs_out, _ = self.bilstm(qs_input)
        ds_out, _ = self.bilstm(ds_input)

        qs_lin = qs_out.transpose(0, 1)[-1]
        ds_lin = ds_out.transpose(0, 1)[-1]

        qs_sem = self.lin(qs_lin)
        ds_sem = self.lin(ds_lin)
        # rnn_out:[B, L, hidden_size*2]
        sims = self.cal_sim(qs_sem, ds_sem)

        return sims
