import random
import numpy as np
from collections import defaultdict
import argparse
import copy
import os

from tqdm import tqdm

import pdb



def create_query_doc_dict(rel_path):
    """Create the dictionary for index of (query, doc) pairs from relevance file.
    Only relevant and slightly relevant pairs are included as the relevance file.
    """

    rel_dict = defaultdict(dict)

    with open(rel_path, 'r') as relf:
        for line in relf.readlines():
            q_id, d_id, rel_type = map(int, line.strip('\n').split('\t'))

            if q_id not in rel_dict:
                rel_dict[q_id] = defaultdict(list)

            if rel_type == 2:
                rel_dict[q_id]['rel'].append(d_id)
            else:
                rel_dict[q_id]['srel'].append(d_id)

            # pdb.set_trace()

    return rel_dict


def create_query_text_dict(query_path):
    """Create the dictionary for query text.
    """

    query_dict = dict()

    with open(query_path, 'r') as qf:
        for line in qf.readlines():
            q_id, word, query = line.strip('\n').split('\t')
            query_dict[int(q_id)] = (word, query)

    return query_dict


def create_doc_text_dict(doc_path):
    """Create the dictionary for doc text.
    """

    doc_dict = dict()

    with open(doc_path, 'r') as df:
        for line in df.readlines():
            d_id, word, doc = line.strip('\n').split('\t')

            doc_dict[int(d_id)] = [word, doc]

    return doc_dict



def divide_queries(rel_dict):
    """Create training queries, dev queries and test queries, from the whole query set.
    """

    #all_queries = set([q_id for q_id, v in rel_dict.items()])
    all_queries = set([q_id for q_id, v in rel_dict.items()][:10])
    test_q_num = int(len(all_queries)*0.2)

    test_queries = set(random.sample(all_queries, k=test_q_num))

    remain_queries = all_queries - test_queries
    dev_queries = set(random.sample(remain_queries, k=test_q_num))

    train_queries = remain_queries - dev_queries

    return train_queries, dev_queries, test_queries


def sample_neg(query_set, rel_dict, d_dict, k):
    """Sample negative documents.
    """

    all_docs = set([d_id for d_id, v in d_dict.items()])

    for q_id in tqdm(query_set):
        # pdb.set_trace()

        non_neg_docs = set(rel_dict[q_id]['rel']+rel_dict[q_id]['srel'])
        to_sample_num = len(non_neg_docs) * k

        while True:
            sampled_d = set(random.sample(all_docs, to_sample_num))

            if len(sampled_d & non_neg_docs) == 0:
                break

        rel_dict[q_id]['nrel'] = list(sampled_d)


def create_dataset(query_set, rel_dict, query_dict, doc_dict, _path):
    """Create the (relevance score, query text, doc text) dataset.
    For the relevance score, use 2 represents relevant, 1 to be slightly relevant and 0 be not relevant.
    """

    with open(_path, 'w') as f:

        for q_id in tqdm(query_set):

            query_text = query_dict[q_id][1]

            for d_id in rel_dict[q_id]['rel']:

                f.write('2\t'+query_text+'\t'+doc_dict[d_id][1]+'\n')

            for d_id in rel_dict[q_id]['srel']:
                f.write('1\t'+query_text+'\t'+doc_dict[d_id][1]+'\n')

            for d_id in rel_dict[q_id]['nrel']:
                f.write('0\t'+query_text+'\t'+doc_dict[d_id][1]+'\n')

def main():

    random.seed(666)

    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', dest='query_path', type=str, default="/Users/jiapengliu/Document/Project/raw_data/wiki-clir/english/wiki_en.queries")
    parser.add_argument('--doc_path', dest='doc_path', type=str, default="/Users/jiapengliu/Document/Project/raw_data/wiki-clir/french/wiki_fr.documents")
    parser.add_argument('--rel_path', dest='rel_path', type=str, default="/Users/jiapengliu/Document/Project/raw_data/wiki-clir/french/en2fr.rel")
    parser.add_argument('--train_negsample', dest='train_negsample', type=int, default=4)
    parser.add_argument('--test_negsample', dest='test_negsample', type=int, default=10)

    args = parser.parse_args()

    lang1 = 'eng'
    lang2 = args.doc_path.split('/')[-2]
    print("Start to process queries in English and docs in {}...".format(lang2))

    path_to_save = 'data_'+lang1+'__'+lang2
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    query_dict = create_query_text_dict(args.query_path)
    doc_dict = create_doc_text_dict(args.doc_path)
    rel_dict = create_query_doc_dict(args.rel_path)

    train_queries_set, dev_queries_set, test_queries_set = divide_queries(rel_dict)

    print("Sampling negative examples for training set...")
    train_rel_dict = sample_neg(train_queries_set, rel_dict, doc_dict, args.train_negsample)
    print("Sampling negative examples for validation set...")
    dev_rel_dict = sample_neg(dev_queries_set, rel_dict, doc_dict, args.test_negsample)
    print("Sampling negative examples for testing set...")
    test_rel_dict = sample_neg(test_queries_set, rel_dict, doc_dict, args.test_negsample)

    print("Creating training set...")
    create_dataset(train_queries_set, rel_dict, query_dict, doc_dict, path_to_save+'/train.csv')
    print("Creating validation set...")
    create_dataset(dev_queries_set, rel_dict, query_dict, doc_dict, path_to_save+'/dev.csv')
    print("Creating testing set...")
    create_dataset(test_queries_set, rel_dict, query_dict, doc_dict, path_to_save+'/test.csv')

    print("Data set successfully created!")

if __name__ == '__main__':
    main()
