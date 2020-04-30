"""Todo: write the evaluation functions
"""
import numpy as np
import torch
import pdb


def predict(sims, theta):
    """
    Predict the relevance score based on theta
    """
    return torch.gt(sims, theta[0]).long() + torch.gt(sims, theta[1]).long()



def predict_cross(sims):
    """
    Predict the relevance score based on theta
    """
    value, indices = sims.max(dim=1)
    return indices


def precision_at_k(sims, qd_index, rels_index, k):
    """Score is precision @ k
    We only consider relevant as relevance score = 2
    Args:
        sims: similarity vector from the model
        qd_index: the index of relevant (query, doc) pair. The following pairs are docs with the given query
        rels_index: a list with same length of queries. Each element is a list of relevant score of the given query
    Returns:
        Precision @ k vector, each element is for one query
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    assert len(qd_index) == len(rels_index) + 1
    n_query = len(rels_index)
    precision_nn = []
    precision_2 = []
    precision_21 = []
    for i in range(n_query):
        r = sims[qd_index[i]:qd_index[i+1]]
        rank = torch.sort(r, descending=True).indices
        result_nn = [rels_index[i][n] != 0 for n in rank[:k]]
        result_2 = [rels_index[i][n] == 2 for n in rank[:k]]
        result_21 = [rels_index[i][n] == 2 for n in rank[:1]]
        precision_nn.append(np.mean(result_nn))
        precision_2.append(np.mean(result_2))
        precision_21.append(np.mean(result_21))
        if len(rels_index[i]) < k:
            raise ValueError('Relevance score length < k')
    return np.mean(precision_nn), np.mean(precision_2) * 5, np.mean(precision_21)



def dcg_at_k(sims, qd_index, rels_index, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    Args:
        sims: similarity vector from the model
        qd_index: the index of relevant (query, doc) pair. The following pairs are docs with the given query
        rels_index: a list with same length of queries. Each element is a list of relevant score of the given query
        k: Number of results to consider
    Returns:
        Discounted cumulative gain
    """
    assert k >= 1
    assert len(qd_index) == len(rels_index) + 1
    n_query = len(rels_index)
    dcg = []
    for i in range(n_query):
        r = sims[qd_index[i]:qd_index[i+1]]
        rank = torch.sort(r, descending=True).indices
        result = [rels_index[i][n] for n in rank[:k]]

        dcg_value = 0
        for j in range(k):
            dcg_value += result[j]/np.log2(j + 2)
        dcg.append(dcg_value)
        if len(rels_index[i]) < k:
            raise ValueError('Relevance score length < k')
    return dcg





def ndcg_at_k(sims, qd_index, rels_index, k):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    n_query = len(rels_index)
    rels = []
    for i in range(n_query):
        rels += rels_index[i]
    rels = torch.tensor(rels)
    dcg_max = dcg_at_k(rels, qd_index, rels_index, k)
    dcg = dcg_at_k(sims, qd_index, rels_index, k)
    ndcg = []
    for i in range(n_query):
        ndcg.append(dcg[i]/dcg_max[i])
    return np.mean(ndcg)



def prediction_accuracy(pred, dev_label):
    """
    return: the accuracy of prediction
    """
    assert len(pred == len(dev_label))
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    class_pred = list(0. for i in range(3))
    c = (pred == dev_label).squeeze()
    for i in range(len(dev_label)):
        label = dev_label[i]
        class_correct[label] += c[i].item()
        class_pred[pred[i]] += 1
        class_total[label] += 1
    return class_total, class_correct, class_pred



def prediction_matrix(prediction, rels_index):
    """Score is precision @ k
    We only consider relevant as relevance score = 2
    Args:
        sims: similarity vector from the model
        qd_index: the index of relevant (query, doc) pair. The following pairs are docs with the given query
        rels_index: a list with same length of queries. Each element is a list of relevant score of the given query
    Returns:
        Accuracy for each class
    Raises:
        ValueError: len(r) must be >= k
    """
    n_query = len(rels_index)
    rels = []
    for i in range(n_query):
        rels += rels_index[i]
    assert len(rels) == len(prediction)
    pre_mat = torch.zeros([4, 4])
    for j in range(len(prediction)):
        pre_mat[rels[j], prediction[j]] += 1
    pre_mat[3, :3] = pre_mat[:3, :3].sum(dim=0)
    pre_mat[:, 3] = pre_mat[:, :3].sum(dim=1)
    """
    acc_t = (pre_mat[0, 0] + pre_mat[1, 1] + pre_mat[2, 2])/len(rels)
    acc_2 = correct_2/rels.count(2)
    acc_1 = correct_1/rels.count(1)
    acc_0 = correct_0/rels.count(0)
    """
    return pre_mat




def map_mrr(sims, qd_index, rels_index):
    """Score is mean average precision
    Returns: MAP
    """
    assert len(qd_index) == len(rels_index) + 1
    n_query = len(rels_index)
    ave_p = []
    mrr_2 = []
    mrr_nn = []
    for i in range(n_query):
        # relevance score list for the given query
        sims_q = sims[qd_index[i]:qd_index[i+1]]
        # ranking index of the documents based on relevance score
        rank = torch.sort(sims_q, descending=True).indices
        # label of the documents sorted by relevance score
        result = [rels_index[i][j] for j in rank]
        # the index of the first non-zero and first 2
        index_nn = result.index(next(filter(lambda x: x != 0, result)))
        index_2 = result.index(next(filter(lambda x: x == 2, result)))
        # append the mrr value to the list
        mrr_nn.append(1/(index_nn + 1))
        mrr_2.append(1 / (index_2 + 1))
        # whether a document is relevant
        r = np.asarray(result) != 0
        delta_r = 1. / sum(r)
        out = sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
        ave_p.append(out)
    return np.mean(ave_p), np.mean(mrr_nn), np.mean(mrr_2)







