"""Todo: write the evaluation functions
"""
import numpy as np
import torch

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
    precision = []
    for i in range(n_query):
        r = sims[qd_index[i]:qd_index[i+1]]
        rank = torch.sort(r, descending=True).indices
        result = [rels_index[i][n] != 0 for n in rank[:k]]
        precision.append(np.mean(result))
        if len(rels_index[i]) < k:
            raise ValueError('Relevance score length < k')
    return precision


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])




def dcg_at_k(sims, qd_index, rels_index, k):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    Args:
        sims: similarity vector from the model
        qd_index: the index of relevant (query, doc) pair. The following pairs are docs with the given query
        rels_index: a list with same length of queries. Each element is a list of relevant score of the given query
        n: Number of results to consider
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
    return ndcg