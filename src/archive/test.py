import numpy as np
import pickle
from collections import defaultdict


def load_dataset_for_neg_sampled_data(self, _type):
    data_limit = 10
    if _type == "train":
        path = self.train_data_path
        data_limit = 32
        if self.sub_sample_train:
            data_limit = self.sub_sample_data_limit
        # data_limit = 20
    elif _type == "dev":
        path = self.dev_data_path
    elif _type == "test":
        path = self.test_data_path


    if not (self.test):
        with open(path, "r") as input_data:
            total_line = len([0 for _ in input_data])
    else:
        total_line = 0

import pickle


path_en = '/Users/jiapengliu/Document/Project/data/polyglot-en.pkl'
path_fr = '/Users/jiapengliu/Document/Project/data/polyglot-fr.pkl'
vocab_q = {}
vocab_d = {}

with open(path_en, "r") as f_q, \
     open(path_fr , "r") as f_d:
    vocab_qu = pickle.load(f_q)
    vocab_do = pickle.load(f_d)
print('done')


with open(path_en, "r") as f_q:
    vocab_qu = pickle.load(f_q)



for i in range(len(vocab_qu[0])):
    vocab_q[vocab_qu[0][i]] = vocab_qu[1][i]

for i in range(len(vocab_do[0])):
    vocab_d[vocab_do[0][i]] = vocab_do[1][i]




word2vec_vocab_q = self.load_vocab(args.vocab_path+"vocab_enwiki.txt", args.vocab_size)
word2vec_vocab_d = self.load_vocab(args.vocab_path+"vocab_"+args.doc_lang+"wiki.txt", args.vocab_size)

size = 1000
def load_vocab(path, size):
    vocab = {}
    for i, w in enumerate(open(path, "r")):
        if i < size:
            vocab[w.strip()] = 1
    return vocab


if 'aaa' in vocab_q[0]:
    print(1)
else: print(0)


word2vec_vocab_q = load_vocab(path, size)

from collections import defaultdict

vocab_q = defaultdict(lambda: len(vocab_q))
vocab_q["<pad>"]
vocab_q["<unk>"]
vocab_d = defaultdict(lambda: len(vocab_d))
vocab_d["<pad>"]
vocab_d["<unk>"]







encode_type = "cnn"
extract_parameter = False
cnn_ksize = 4
path = '/Users/jiapengliu/Document/Project/data/2class/data_en_fr/train.txt'

n_qd_pairs = []
count = 0
dataset = []

with open(path, "r") as input_data:
    # for i, line in tqdm(enumerate(input_data), total=total_line):
    for i, line in enumerate(input_data):
        rel, query, doc = line.strip().split("\t")
        if i == 10:
            break
        if rel == str(2):
            if i != 0:
                n_qd_pairs.append(count)
                count = 0
            first_line_query = query
            doc_rel = doc

        else:
            count += 1
            assert first_line_query == query
            assert rel == str(0)
            doc_nonrel = doc

            # convert text data to index data as numpy array
            if extract_parameter:
                x1s = np.array([1,0], dtype=np.int32)
                x2s = np.array([1,0], dtype=np.int32)
                x3s = np.array([0,0], dtype=np.int32)
            else:
                arg1 = [vocab_q[token] if token in vocab_q else vocab_q["<UNK>"] for token in query.strip('.').split()]
                arg2 = [vocab_d[token] if token in vocab_d else vocab_d["<UNK>"] for token in doc_rel.split()]
                arg3 = [vocab_d[token] if token in vocab_d else vocab_d["<UNK>"] for token in doc_nonrel.split()]

                if encode_type == "cnn":
                    if len(arg1) < cnn_ksize:
                        arg1 += [0 for _ in range(cnn_ksize-len(arg1))]
                    if len(arg2) < cnn_ksize:
                        arg2 += [0 for _ in range(cnn_ksize-len(arg2))]
                    if len(arg3) < cnn_ksize:
                        arg3 += [0 for _ in range(cnn_ksize-len(arg3))]

                x1s = np.array(arg1, dtype=np.int32)
                x2s = np.array(arg2, dtype=np.int32)
                x3s = np.array(arg3, dtype=np.int32)

            t = np.array([0], dtype=np.int32)

            dataset.append((x1s, x2s, x3s, t))

    else:
        n_qd_pairs.append(count)




return dataset, n_qd_pairs