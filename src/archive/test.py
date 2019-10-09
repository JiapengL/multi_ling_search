import numpy as np
import pickle
from collections import defaultdict
import pickle




def load_vocab(path, size):
    vocab = {}
    for i, w in enumerate(open(path, "r")):
        if i < size:
            vocab[w.strip()] = 1
    return vocab


#load embeddings
path_en = '/Users/jiapengliu/Document/Project/raw_data/polyglot-en.pkl'
path_fr = '/Users/jiapengliu/Document/Project/raw_data/polyglot-fr.pkl'
vocab_q = {}
vocab_d = {}

with open(path_en, "rb") as f_q, \
     open(path_fr , "rb") as f_d:
    vocab_qu = pickle.load(f_q,  encoding='latin1')
    vocab_doc = pickle.load(f_d,  encoding='latin1')




#save embedding dict
path_en_emb = '/Users/jiapengliu/Document/Project/raw_data/polyglot_en_dict.pkl'
path_fr_emb = '/Users/jiapengliu/Document/Project/raw_data/polyglot_fr_dict.pkl'

pickle.dump(vocab_q, open(path_en_emb, "wb"))
pickle.dump(vocab_d, open(path_fr_emb, "wb"))



# write vocabulary of eng
path_en_vocab = '/Users/jiapengliu/Document/Project/raw_data/vocab_en.txt'
with open(path_en_vocab, 'w') as f:
    for word in vocab_qu[0]:
        f.write(word+'\n')


# write vocabulary of fr
path_fr_vocab = '/Users/jiapengliu/Document/Project/raw_data/vocab_fr.txt'
with open(path_fr_vocab, 'w') as f:
    for word in vocab_doc[0]:
        f.write(word+'\n')

np.piecewise(sims, [x < 0, x >= 0], [lambda x: -x, lambda x: x])


def threshold(rels, index):
    return(torch.gt(rels, index - 0.0001).float() - 0.5)*2



loss = F.relu(threshold(rels, 2)*(theta[1]-sims)) + F.relu(threshold(rels, 1)*(theta[0]-sims))

word_embed/polyglot_en_dict.pkl
word_embed/polyglot_fr_dict.pkl


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