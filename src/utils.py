import random
import math

import numpy as np
import torch

def pad_convert(sent, vocab, max_l, pad_type="back"):

    num_st = [vocab[t] if t in vocab else vocab["<unk>"] for t in sent]

    if pad_type=="front":
        num_st = [vocab["pad"]]*(max_l-len(sent)) + num_st
    elif pad_type=="both":
        if (max_l-len(sent))%2 == 0:
            pad_l = (max_l-len(sent))/2
            num_st = [vocab["pad"]]*pad_l + num_st + [vocab["pad"]]*pad_l
        else:
            pad_l, pad_r = (max_l-len(sent))/2, (max_l-len(sent))/2+1
            num_st = [vocab["pad"]]*pad_l + num_st + [vocab["pad"]]*pad_r
    else:
        num_st = num_st + [vocab["pad"]]*(max_l-len(sent))

    return num_st


def numerize(batch, q_vocab, d_vocab, pad_type="back"):

    labels = [int(pair[0]) for pair in batch]

    batch_max_q_len = max(map(lambda st: len(st), [pair[1] for pair in batch]))
    q_x = [pad_convert(pair[1], q_vocab, batch_max_q_len, pad_type) for pair in batch]


    batch_max_d_len = max(map(lambda st: len(st), [pair[2] for pair in batch]))
    d_x = [pad_convert(pair[2], d_vocab, batch_max_d_len, pad_type) for pair in batch]

    return torch.tensor(labels).long(), torch.tensor(q_x).long(), torch.tensor(d_x).long()
