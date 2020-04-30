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
        self.epsilon = args.epsilon
        # embedding of q, the input of embedding should be index tensor
        self.q_word_embeds = nn.Embedding(q_vocab_size, self.embed_dim).to(self.device)
        # embedding of d
        self.d_word_embeds = nn.Embedding(d_vocab_size, self.embed_dim).to(self.device)
        self.out_dim = 3
        self.crosslin = nn.Linear(self.embed_dim*2, self.out_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm = nn.BatchNorm1d(args.embed_dim, momentum=0)



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

        if self.args.dropout:
            qs_emb = self.dropout(qs_emb)
            ds_emb = self.dropout(ds_emb)

        self.qs_emb = qs_emb
        self.ds_emb = ds_emb

        adv_flag = self.training and self.args.vat

        if adv_flag and q_perturb is not None and d_perturb is not None:
            qs_emb += q_perturb
            ds_emb += d_perturb


        if self.args.batchnorm:
            q_avg = self.batchnorm(torch.mean(qs_emb, dim=1))
            d_avg = self.batchnorm(torch.mean(ds_emb, dim=1))

            q_rep = torch.tanh(q_avg)
            d_rep = torch.tanh(d_avg)
        else:
            q_rep = torch.tanh(torch.mean(qs_emb, dim=1))
            d_rep = torch.tanh(torch.mean(ds_emb, dim=1))

        if self.args.use_cross:
            qd_cat = torch.cat([q_rep, d_rep], dim=1)
            qd_cross = self.crosslin(qd_cat)
            return qd_cross

        else:
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

        return(torch.ge(rels, index).float() - 0.5) * 2

    def laplace_loss(self, x):
        #pdb.set_trace()
        type_p = torch.ge(x, 0).float() * (-torch.log(2 - torch.exp(-x)))
        type_n = torch.lt(x, 0).float() * (-x)
        return type_p + type_n

    def laplace_trun(self, x):
        return torch.lt(x, 0).float() * (-x)

    def sigmoid_trun(self, x):
        m = nn.LogSigmoid()
        return torch.lt(x, 0).float() * m(-x)


    def cal_loss(self, sims, rels):
        """
        :param self.theta: thresholds of ordinal regression loss
        :param sims: similarity derived from the model
        :param rels: relevance score: 0, 1, 2
        :return: average loss
        """
        if self.args.use_sps:
            # if using Semantic product search paper loss

            loss_type0 = F.relu((rels == 0).float() * (sims - self.theta[0])).pow(self.args.m)
            loss_type1 = F.relu((rels == 1).float() * (sims - self.theta[1])).pow(self.args.m)
            loss_type2 = F.relu((rels == 2).float() * (self.theta[2] - sims)).pow(self.args.m)
            return torch.mean(loss_type0 + loss_type1 + loss_type2)

        elif self.args.use_cross:
            loss_f = nn.CrossEntropyLoss()
            return loss_f(sims, rels)

        elif self.args.im_loss:
            # if using immediate loss
            """
            loss_type0 = (rels == 0).float() * F.relu(sims - self.theta[0])
            loss_type1 = (rels == 1).float() * (F.relu(sims - self.theta[1]) + F.relu(self.theta[0] - sims))
            loss_type2 = (rels == 2).float() * F.relu(self.theta[1] - sims)
            return torch.mean(loss_type0 + loss_type1 + loss_type2)
            """

            loss_type0 = (rels == 0).float() * (self.laplace_trun(self.theta[0] - sims) + self.laplace_trun(sims + 1)).pow(self.args.m)
            loss_type1 = (rels == 1).float() * (self.laplace_trun(self.theta[1] - sims) + self.laplace_trun(sims - self.theta[0])).pow(self.args.m)
            loss_type2 = (rels == 2).float() * (self.laplace_trun(1 - sims) + self.laplace_trun(sims - self.theta[1])).pow(self.args.m)
            return torch.mean(loss_type0 + loss_type1 + loss_type2)



        elif self.args.pol:
            # if using Proportional Odds model with Laplace

            loss_type0 = (rels == 0).float() * (self.laplace_loss(self.theta[0] - sims) + self.laplace_loss(sims + 1))
            loss_type1 = (rels == 1).float() * (self.laplace_loss(self.theta[1] - sims) + self.laplace_loss(sims - self.theta[0]))
            loss_type2 = (rels == 2).float() * (self.laplace_loss(1 - sims) + self.laplace_loss(sims - self.theta[1]))
            return torch.mean(loss_type0 + loss_type1 + loss_type2)


        elif self.args.pos_trun:
            # if using Proportional Odds model with sigmoid
            loss_type0 = (rels == 0).float() * (self.sigmoid_trun(self.theta[0] - sims) + self.sigmoid_trun(sims + 1))
            loss_type1 = (rels == 1).float() * (self.sigmoid_trun(self.theta[1] - sims) + self.sigmoid_trun(sims - self.theta[0]))
            loss_type2 = (rels == 2).float() * (self.sigmoid_trun(1 - sims) + self.sigmoid_trun(sims - self.theta[1]))
            return torch.mean(loss_type0 + loss_type1 + loss_type2)


        elif self.args.pos:
            # if using Proportional Odds model with sigmoid
            m = nn.Sigmoid()
            loss_type0 = (rels == 0).float() * (torch.log(m(self.theta[0] - sims)) + torch.log(1 - m(-1 - sims)))
            loss_type1 = (rels == 1).float() * (torch.log(m(self.theta[1] - sims)) + torch.log(1 - m(self.theta[0] - sims)))
            loss_type2 = (rels == 2).float() * (torch.log(m(1 - sims)) + torch.log(1 - m(self.theta[1] - sims)))
            return torch.mean(-1*(loss_type0 + loss_type1 + loss_type2))


        elif self.args.leaky_loss:
            # if using leaky_loss

            loss_theta1 = F.relu(self.__threshold(rels, 1) * (self.theta[0] - sims)).pow(self.args.m)
            loss_theta2 = F.relu(self.__threshold(rels, 2) * (self.theta[1] - sims)).pow(self.args.m)
            loss_type0 = (rels == 0).float() * (self.theta[0] + 1 - F.relu(self.theta[0] - sims))/(self.theta[0] + 1)
            loss_type2 = (rels == 2).float() * (1 - self.theta[1] - F.relu(sims - self.theta[1]))/(1 - self.theta[1])
            loss_type11 = (np.mean(self.theta) - self.theta[0] - F.relu(sims - self.theta[0])) * torch.lt(sims, np.mean(self.theta)).float()/(np.mean(self.theta) - self.theta[0])
            loss_type12 = (self.theta[1] - np.mean(self.theta) - F.relu(self.theta[1] - sims)) * torch.gt(sims, np.mean(self.theta)).float()/(self.theta[1] - np.mean(self.theta))
            loss_type1 = (rels == 1).float() * (loss_type11 + loss_type12)
            loss_leaky = self.epsilon * (loss_type0 + loss_type1 + loss_type2)
            return torch.mean(loss_theta1 + loss_theta2 + loss_leaky)

        elif self.args.laplace:
            # if using laplace loss

            loss_type0 = (rels == 0).float() * (sims - (self.theta[0]-1)/2).abs().pow(self.args.m)
            loss_type1 = (rels == 1).float() * (sims - (self.theta[0]+self.theta[1])/2).abs().pow(self.args.m)
            loss_type2 = (rels == 2).float() * (sims - (self.theta[1]+1)/2).abs().pow(self.args.m)
            return torch.mean(loss_type0 + loss_type1 + loss_type2)

        elif self.args.combine:

            loss_type0 = (rels == 0).float() * F.relu(sims - self.theta[0])
            loss_type1 = (rels == 1).float() * (F.relu(sims - self.theta[1]) + F.relu(self.theta[0] - sims))
            loss_type2 = (rels == 2).float() * F.relu(self.theta[1] - sims)
            loss_im = loss_type0 + loss_type1 + loss_type2

            loss_typel0 = (rels == 0).float() * (sims - (self.theta[0]-1)/2).abs().pow(self.args.m)
            loss_typel1 = (rels == 1).float() * (sims - (self.theta[0]+self.theta[1])/2).abs().pow(self.args.m)
            loss_typel2 = (rels == 2).float() * (sims - (self.theta[1]+1)/2).abs().pow(self.args.m)
            loss_laplace = self.epsilon * (loss_typel0 + loss_typel1 + loss_typel2)
            return torch.mean(loss_im + loss_laplace)

        else:
            # if using ordinal regression loss
            loss_theta1 = F.relu(self.__threshold(rels, 1) * (self.theta[0] - sims)).pow(self.args.m)
            loss_theta2 = F.relu(self.__threshold(rels, 2) * (self.theta[1] - sims)).pow(self.args.m)
            return torch.mean(loss_theta1 + loss_theta2)










class CrossEntropy(SimpleDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(CrossEntropy, self).__init__(args, q_vocab_size, d_vocab_size)

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




    def forward(self, qs, ds, q_perturb=None, d_perturb=None):

        qs_emb = self.q_word_embeds.forward(qs)
        ds_emb = self.d_word_embeds.forward(ds)

        self.qs_emb = qs_emb
        self.ds_emb = ds_emb

        q_rep = torch.tanh(torch.mean(qs_emb, dim=1))
        d_rep = torch.tanh(torch.mean(ds_emb, dim=1))

        if self.args.use_cross:
            qd_cat = torch.cat([q_rep, d_rep], dim=1)
            qd_cross = self.crosslin(qd_cat)
            return qd_cross

        else:
            sims = self.cal_sim(q_rep, d_rep)
            return sims







class DeepDSSM(SimpleDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(DeepDSSM, self).__init__(args, q_vocab_size, d_vocab_size)

        # layers for query
        self.WINDOW_SIZE = 3
        self.TOTAL_LETTER_GRAMS = int(1 * 1e5)
        self.WORD_DEPTH = self.WINDOW_SIZE * self.embed_dim
        self.K = 300   # size of filter
        self.L = 128    # size of output
        self.drop_out = nn.Dropout(p=0.5)
        self.q_conv = nn.Conv1d(self.WORD_DEPTH, self.K, 1)
        self.d_conv = nn.Conv1d(self.WORD_DEPTH, self.K, 1)

        if torch.cuda.device_count() > 1:
            self.q_conv = nn.DataParallel(self.q_conv)
         #   self.d_conv = nn.DataParallel(self.d_conv)

        self.q_fc = nn.Linear(self.K, self.L)
        self.d_fc = nn.Linear(self.K, self.L)

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
        ds_conv = torch.tanh(self.d_conv(ds_ngram))

        qs_maxp = self.max_pooling(qs_conv)
        ds_maxp = self.max_pooling(ds_conv)

        #qs_maxp = torch.mean(qs_conv, dim=2)
        #ds_maxp = torch.mean(ds_conv, dim=2)

        #qs_drop = self.drop_out(qs_maxp)
        #ds_drop = self.drop_out(ds_maxp)

        qs_sem = self.q_fc(qs_maxp)
        ds_sem = self.d_fc(ds_maxp)
        # pdb.set_trace()
        sims = self.cal_sim(qs_sem, ds_sem)

        return sims









class LSTM(SimpleDSSM):

    def __init__(self, args, q_vocab_size, d_vocab_size):
        super(LSTM, self).__init__(args, q_vocab_size, d_vocab_size)

        # layers for query
        self.hidden_size = 128
        self.bilstm = nn.LSTM(self.embed_dim, self.hidden_size, batch_first=True, bidirectional=True)
        #self.d_bilstm = nn.LSTM(self.embed_dim, self.hidden_size, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.5)
        self.out_size = 128
        self.lin = nn.Linear(self.hidden_size*2, self.out_size)



    def forward(self, qs, ds, q_perturb=None, d_perturb=None):

        qs_emb = self.q_word_embeds.forward(qs)
        ds_emb = self.d_word_embeds.forward(ds)

        self.qs_emb = qs_emb
        self.ds_emb = ds_emb

        self.bilstm.flatten_parameters()
        """
        #self.d_bilstm.flatten_parameters()

        # qs_emb = self.dropout(qs_emb)
        # ds_emb = self.dropout(ds_emb)

        adv_flag = self.training and self.args.vat

        if adv_flag and q_perturb is not None and d_perturb is not None:
            qs_emb += q_perturb
            ds_emb += d_perturb

        q_rep = torch.tanh(torch.mean(qs_emb, dim=1))
        """

        qs_out = self.bilstm(qs_emb)[0]
        #qs_lin = qs_out.transpose(0, 1)[-1]
        qs_lin = torch.mean(qs_out, dim=1)
        q_rep = torch.tanh(self.lin(qs_lin))


        ds_out = self.bilstm(ds_emb)[0]
        ds_lin = torch.mean(ds_out, dim=1)
        #ds_lin = ds_out.transpose(0, 1)[-1]
        d_rep = torch.tanh(self.lin(ds_lin))

        sims = self.cal_sim(q_rep, d_rep)

        return sims

