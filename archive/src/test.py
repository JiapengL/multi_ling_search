import numpy as np
import pickle
from collections import defaultdict
import pickle
import torch



def load_vocab(path, size):
    vocab = {}
    for i, w in enumerate(open(path, "r")):
        if i < size:
            vocab[w.strip()] = 1
    return vocab


#load embeddings
path_en = '/Users/jiapengliu/Document/Project/raw_data/embedding/polyglot-en.pkl'
path_fr = '/Users/jiapengliu/Document/Project/raw_data/embedding/polyglot-fr.pkl'
path_ja = '/Users/jiapengliu/Document/Project/raw_data/embedding/polyglot-ja.pkl'
path_de = '/Users/jiapengliu/Document/Project/raw_data/embedding/polyglot-de.pkl'

vocab_en = {}
vocab_fr = {}

with open(path_en, "rb") as f_q, \
     open(path_fr , "rb") as f_d:
    vocab_qu = pickle.load(f_q,  encoding='latin1')
    vocab_doc = pickle.load(f_d,  encoding='latin1')


with open(path_ja, "rb") as f_j:
    vocab_ja = pickle.load(f_j, encoding='latin1')


with open(path_de, "rb") as f_j:
    vocab_de = pickle.load(f_j, encoding='latin1')


# save embedding dict
path_en_emb = '/Users/jiapengliu/Document/Project/raw_data/polyglot_en_dict.pkl'
path_fr_emb = '/Users/jiapengliu/Document/Project/raw_data/polyglot_fr_dict.pkl'
path_ja_emb = '/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/polyglot_ja_dict.pkl'
path_de_emb = '/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/polyglot_de_dict.pkl'


pickle.dump(vocab_qu, open(path_en_emb, "wb"))
pickle.dump(vocab_doc, open(path_fr_emb, "wb"))
pickle.dump(vocab_ja, open(path_ja_emb, "wb"))
pickle.dump(vocab_de, open(path_de_emb, "wb"))



# write vocabulary of eng
path_en_vocab = '/Users/jiapengliu/Document/Project/raw_data/vocab_en.pkl'
pickle.dump(vocab_qu[0], open(path_en_vocab, "wb"))


# write vocabulary of fr
path_fr_vocab = '/Users/jiapengliu/Document/Project/raw_data/vocab_fr.pkl'
pickle.dump(vocab_doc[0], open(path_fr_vocab, "wb"))


# write vocabulary of ja
path_ja_vocab = '/Users/jiapengliu/Document/Project/raw_data/embedding/vocab_ja.pkl'
pickle.dump(vocab_ja[0], open(path_ja_vocab, "wb"))



# write vocabulary of ja
path_de_vocab = '/Users/jiapengliu/Document/Project/raw_data/embedding/vocab_de.pkl'
pickle.dump(vocab_de[0], open(path_de_vocab, "wb"))








path = '/Users/jiapengliu/Document/Project/multi_ling_search/word_embed/polyglot_fr_dict.pkl'
with open(path, "rb") as fi:
    vocab = pickle.load(fi)




print("loading " + _type + " embeddings...")
with open(path_en_emb, "rb") as fi:
    vocab_qu = pickle.load(fi, encoding='latin1')


    for n, line in enumerate(fi.readlines()):
        # 1st line contains stats
        if n == 0:
            continue
        line_list = line.strip().split(" ", 1)
        word = line_list[0].lower() if args.caseless else line_list[0]
        if word in vocab:
            value = line.strip().split(" ")[1::]
            vec = np.fromstring(value, dtype=float, sep=' ')
            temp_tensor[vocab[word]] = nn.Parameter(torch.from_numpy(vec))

if _type == "query":
    self.q_word_embeds = nn.Embedding.from_pretrained(temp_tensor)
elif _type == "document":
    self.d_word_embeds = nn.Embedding.from_pretrained(temp_tensor)

with open(args.q_extn_embedding, "rb") as f_q, \
        open(args.d_extn_embedding, "rb") as f_d:
    self.q_word_embeds = pickle.load(f_q, encoding='latin1')
    self.q_word_embeds = pickle.load(f_d, encoding='latin1')






with open(path_en_emb, "rb") as f_q:
    vocab = pickle.load(f_q,  encoding='latin1')
print('done')

temp_tensor = torch.randn(len(vocab), args.embedding_dim)

print("loading " + _type + " embeddings...")
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
            temp_tensor[vocab[word]] = nn.Parameter(torch.from_numpy(vec))

for i in range(len(vocab_qu[0])):
    vocab_q[vocab_qu[0][i]] = vocab_qu[1][i]
    vocab_d[vocab_doc[0][i]] = vocab_doc[1][i]




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











# load training data
train_file = '/home/liu1769/scratch/data_eng__french50000/train.pkl'


with open(train_file, "rb") as f:
    data_ls = pickle.load(f)


# load vocab

from collections import defaultdict


q_vocab_path = '/home/liu1769/multi_ling_search/word_embed/vocab_en.pkl'
d_vocab_path = '/home/liu1769/multi_ling_search/word_embed/vocab_fr.pkl'
vocab_size = 100000

q_vocab = defaultdict(int)
d_vocab = defaultdict(int)

with open(q_vocab_path, "rb") as f_q:
    voc_q = pickle.load(f_q)

for i, w in enumerate(voc_q):
    if i < vocab_size:
        if w not in q_vocab:
            q_vocab[w.strip()] = len(q_vocab)

with open(d_vocab_path, "rb") as f_d:
    voc_d = pickle.load(f_d)

for i, w in enumerate(voc_d):
    if i < vocab_size:
        if w not in d_vocab:
            d_vocab[w.strip()] = len(d_vocab)



# build data
from nltk.tokenize import sent_tokenize, word_tokenize

def _build_data(data_path, caseless):
    """Build the token dataset for sentence pairs, each data point is represented by (relevance score, query_text, doc_text)
    """
    data_ls = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            rel, q_text_s, d_text_s = line.strip('\n').split('\t')
            q_text, d_text = word_tokenize(q_text_s), word_tokenize(d_text_s)
            if caseless:
                q_text = [t.lower() for t in q_text]
                d_text = [t.lower() for t in d_text]
            data_ls.append([rel, q_text, d_text])
    return data_ls
train_path = 'train.csv'






# create query dict and doc dict

query_path = "/home/liu1769/scratch/english/wiki_en.queries"
doc_path = "/home/liu1769/scratch/french/wiki_fr.documents"

query_dict = dict()

with open(query_path, 'r', encoding="utf-8") as qf:
    for line in qf.readlines():
        q_id, word, query = line.strip('\n').split('\t')
        query_dict[int(q_id)] = (word, query)


doc_dict = dict()
with open(doc_path, 'r+', encoding="utf-8") as df:
    for line in df.readlines():
        d_id, word, doc = line.strip('\n').split('\t')
        doc_dict[int(d_id)] = [word, doc]


rel_path = "/home/liu1769/scratch/french/en2fr.rel"
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


all_queries = [q_id for q_id, v in rel_dict.items()]
(all_queries.sort()[:200])
all_queries = set([q_id for q_id, v in rel_dict.items()][:10])
test_q_num = int(len(all_queries)*0.2)

test_queries = set(random.sample(all_queries, k=test_q_num))

remain_queries = all_queries - test_queries
dev_queries = set(random.sample(remain_queries, k=test_q_num))

train_queries = remain_queries - dev_queries




train_rel_dict = sample_neg(train_queries, rel_dict, doc_dict, 4)


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






from data_prep import
query_path = "/home/liu1769/scratch/english/wiki_en.queries"
doc_path = "/home/liu1769/scratch/japanese/wiki_ja.documents"
rel_path = "/home/liu1769/scratch/japanese/en2ja.rel"

query_dict = data_prep.create_query_text_dict(query_path)
doc_dict = data_prep.create_doc_text_dict(doc_path)
rel_dict = data_prep.create_query_doc_dict(rel_path)

train_queries_set, dev_queries_set, test_queries_set = data_prep.divide_queries(rel_dict)






### create unlabel data

train_path = "/home/liu1769/scratch/data_eng__french/train.pkl"


def _load_data(data_path):
    """Build the token dataset for sentence pairs, each data point is represented by (relevance score, query_text, doc_text)
    """
    with open(data_path, "rb") as f:
        data_ls = pickle.load(f)
    return data_ls

train = _load_data(train_path)

# sample n_query from the train dataset as labeled data

n_query = 1000

def generate_eval_index(data_ls):
    """
    input: data
    return: n_qd_pairs: the index of rel == 2, the last item of n_qd_pairs is the length of the dataset
            rels: a list with length of queries, each item is the list of relevance score for the query
    """
    n_qd_pairs = [0]
    rels = []
    count = -1
    for i, line in enumerate(data_ls):
        count += 1
        rel = int(line[0])
        if rel == 2:
            if i != 0:
                n_qd_pairs.append(count)
                rels.append(rels_vector)
                rels_vector = [rel]
            else:
                rels_vector = [rel]
        else:
            rels_vector.append(rel)
    rels.append(rels_vector)
    n_qd_pairs.append(len(data_ls))
    return n_qd_pairs, rels

n_qd_pairs, _ = generate_eval_index(train)


# generate random sample labeled query index
label_index = np.random.choice(len(n_qd_pairs) - 1, n_query, replace=False)
# other queries should be unlabeled
len_index = list(range(len(n_qd_pairs) - 1))
unlabel_index = [item for item in len_index if item not in label_index]

def generate_data(train, n_qd_pairs, n_query, label_index):
    data_list = []
    for i in range(n_query):
        start_index = n_qd_pairs[label_index[i]]
        end_index = n_qd_pairs[label_index[i] + 1]
        data_list += train[start_index:end_index]
    return data_list

data_ll = generate_data(train, n_qd_pairs, n_query, label_index)
data_ul = generate_data(train, n_qd_pairs, len(n_qd_pairs)-1-n_query, unlabel_index)



train_l_path = "/home/liu1769/scratch/data_eng__french/train_l.pkl"
train_ul_path = "/home/liu1769/scratch/data_eng__french/train_ul.pkl"


pickle.dump(data_ll, open(train_l_path, "wb"))
pickle.dump(data_ul, open(train_ul_path, "wb"))



### Sample a smaller dev data
dev_2000_path = "/home/liu1769/scratch/data_eng__french/dev_2000.pkl"
pickle.dump(test_2000, open(dev_2000_path, "wb"))



for x, y in zip(a, b):
    print(x, y)

d = torch.tensor([[6.3277e+04, 1.6722e+04, 1.0000e+00],
        [5.7060e+03, 9.1780e+03, 4.0000e+00],
        [9.2300e+02, 1.0770e+03, 0.0000e+00]])
