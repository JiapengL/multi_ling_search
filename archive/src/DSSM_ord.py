import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



LETTER_GRAM_SIZE = 3 # See section 3.2.
WINDOW_SIZE = 3 # See section 3.2.
TOTAL_LETTER_GRAMS = int(3 * 1e4) # Determined from data. See section 3.2.
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
# Uncomment it, if testing
# WORD_DEPTH = 1000
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
SR = 2
NEG = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 1 # We only consider one time step for convolutions.


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)

def normalize(x):
   x_normed = x/x.norm()

   return x_normed

class CDSSM(nn.Module):


    def __init__(self):
        super(CDSSM, self).__init__()
        # layers for query
        self.query_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.query_sem = nn.Linear(K, L)
        # layers for docs
        self.doc_conv = nn.Conv1d(WORD_DEPTH, K, FILTER_LENGTH)
        self.doc_sem = nn.Linear(K, L)
        # learning gamma
        self.learn_gamma = nn.Conv1d(1, 1, 1)


    def forward(self, q, pos, sre, negs):
        # Query model. The paper uses separate neural nets for queries and documents (see section 5.2).
        # To make it compatible with Conv layer we reshape it to: (batch_size, WORD_DEPTH, query_len)
        q = q.transpose(1, 2)
        # In this step, we transform each word vector with WORD_DEPTH dimensions into its
        # convolved representation with K dimensions. K is the number of kernels/filters
        # being used in the operation. Essentially, the operation is taking the dot product
        # of a single weight matrix (W_c) with each of the word vectors (l_t) from the
        # query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh activation.
        # That is, h_Q = tanh(W_c • l_Q + b_c). Note: the paper does not include bias units.
        q_c = F.tanh(self.query_conv(q))
        # Next, we apply a max-pooling layer to the convolved query matrix.
        q_k = kmax_pooling(q_c, 2, 1)
        q_k = q_k.transpose(1, 2)
        # In this step, we generate the semantic vector represenation of the query. This
        # is a standard neural network dense layer, i.e., y = tanh(W_s • v + b_s). Again,
        # the paper does not include bias units.
        q_s = F.tanh(self.query_sem(q_k))
        q_s = q_s.resize(L)
        # # The document equivalent of the above query model for positive document
        pos = pos.transpose(1, 2)
        pos_c = F.tanh(self.doc_conv(pos))
        pos_k = kmax_pooling(pos_c, 2, 1)
        pos_k = pos_k.transpose(1, 2)
        pos_s = F.tanh(self.doc_sem(pos_k))
        pos_s = normalize(pos_s)
        pos_s = pos_s.resize(L)
        # # The document equivalent of the above query model for slightly-relevant documents
        sre = [sr.transpose(1, 2) for sr in sre]
        sre_cs = [F.tanh(self.doc_conv(sr)) for sr in sre]
        sre_ks = [kmax_pooling(sre_c, 2, 1) for sre_c in sre_cs]
        sre_ks = [sre_k.transpose(1, 2) for sre_k in sre_ks]
        sre_ss = [F.tanh(self.doc_sem(sre_k)) for sre_k in sre_ks]
        sre_ss = [normalize(x) for x in sre_ss]
        sre_ss = [sre_s.resize(L) for sre_s in sre_ss]
        # # The document equivalent of the above query model for negative documents
        negs = [neg.transpose(1, 2) for neg in negs]
        neg_cs = [F.tanh(self.doc_conv(neg)) for neg in negs]
        neg_ks = [kmax_pooling(neg_c, 2, 1) for neg_c in neg_cs]
        neg_ks = [neg_k.transpose(1, 2) for neg_k in neg_ks]
        neg_ss = [F.tanh(self.doc_sem(neg_k)) for neg_k in neg_ks]
        neg_ss = [normalize(x) for x in neg_ss]
        neg_ss = [neg_s.resize(L) for neg_s in neg_ss]
        # Now let us calculates the cosine similarity between the semantic representations of
        # a queries and documents
        # dots[0] is the dot-product for positive document, this is necessary to remember
        # because we set the target label accordingly
        pred_pos = [q_s.dot(pos_s)]
        pred_rs = [q_s.dot(sre_s) for sre_s in sre_ss]
        pred_neg = [q_s.dot(neg_s) for neg_s in neg_ss]
        # dots is a list as of now, lets convert it to torch variable
        #dots2 = torch.stack(dots1)
        # In this step, we multiply each dot product value by gamma. In the paper, gamma is
        # described as a smoothing factor for the softmax function, and it's set empirically
        # on a held-out data set. We're going to learn gamma's value by pretending it's
        # a single 1 x 1 kernel.
        # This Gamma may be negative...
        # with_gamma = self.learn_gamma(dots2.resize(SR + NEG + 1, 1, 1))
        # Finally, we use the softmax function to calculate P(D+|Q).
        # prob = F.softmax(with_gamma)
        return pred_pos, pred_rs, pred_neg


if __name__ == '__main__':
    """
    for debug purpose
    """

    model = CDSSM()

    # Build a random data set.
    sample_size = 10
    l_Qs = []
    pos_l_Ds = []

    (query_len, doc_len) = (5, 100)

    for i in range(sample_size):
        query_len = np.random.randint(1, 10)
        l_Q = np.random.rand(1, query_len, WORD_DEPTH)
        l_Qs.append(l_Q)

        doc_len = np.random.randint(50, 500)
        l_D = np.random.rand(1, doc_len, WORD_DEPTH)
        pos_l_Ds.append(l_D)

    # Generate negative and slightly relevant docs
    neg_l_Ds = [[] for j in range(NEG)]
    sr_l_Ds = [[] for j in range(SR)]
    for i in range(sample_size):
        possibilities = list(range(sample_size))
        possibilities.remove(i)
        index =  np.random.choice(possibilities, NEG + SR, replace=False)
        for j in range(NEG):
            negative = index[j]
            neg_l_Ds[j].append(pos_l_Ds[negative])
        for k in range(SR):
            sre = index[k + NEG]
            sr_l_Ds[k].append(pos_l_Ds[sre])

    # Till now, we have made a complete numpy dataset
    # Now let's convert the numpy variables to torch Variable

    for i in range(len(l_Qs)):
        l_Qs[i] = Variable(torch.from_numpy(l_Qs[i]).float())
        pos_l_Ds[i] = Variable(torch.from_numpy(pos_l_Ds[i]).float())
        for j in range(NEG):
            neg_l_Ds[j][i] = Variable(torch.from_numpy(neg_l_Ds[j][i]).float())
        for k in range(SR):
            sr_l_Ds[k][i] = Variable(torch.from_numpy(sr_l_Ds[k][i]).float())

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    theta = [-1, -0.5, 0.5, 1]

    # output variable, remember the cosine similarity with positive doc was at 0th index
    y = np.ndarray(1) # CrossEntropyLoss expects only the index as a long tensor
    y[0] = 0
    y = Variable(torch.from_numpy(y).long())


    for i in range(sample_size):
        pred_pos, pred_rs, pred_neg = model(l_Qs[i], pos_l_Ds[i], [sr_l_Ds[j][i] for j in range(SR)], [neg_l_Ds[j][i] for j in range(NEG)])
        loss_pos =
        loss_sr =
        loss_neg =
        loss = loss_pos + loss_sr + loss_neg
        print(i, loss.data.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for i in range(sample_size):
        sre_ss, neg_ss, dots, dots1 = model(l_Qs[i], pos_l_Ds[i], [sr_l_Ds[j][i] for j in range(SR)], [neg_l_Ds[j][i] for j in range(NEG)])
        print(dots1)