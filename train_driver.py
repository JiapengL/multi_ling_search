# coding: utf-8
import os
import random
import numpy as np
import argparse
from datetime import datetime
import cPickle, json

HOME = os.getenv("HOME")

import torch
import torch.nn as nn
import torch.nn.functional  as F

from src.margin_model import SimpleDSSM, DeepDSSM, DataProcessor, RankingEvaluator



def main(args):
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    if args.gpu >= 0 and torch.cuda.is_available():
        t = np.array([0], dtype=np.int32)
        tmp = torch.from_numpy(t).cuda()

    # setup result directory
    start_time = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    if args.test:
        start_time = "test_" + start_time
    result_dest = HOME + "/clir/sep_encode_model/result/"+start_time
    result_abs_dest = os.path.abspath(result_dest)
    if not args.extract_parameter:
        os.makedirs(result_dest)
        with open(os.path.join(result_abs_dest, "settings.json"), "w") as fo:
            fo.write(json.dumps(vars(args), sort_keys=True, indent=4))

    # data setup
    data_processor = DataProcessor(args)
    data_processor.prepare_dataset()
    vocab_q = data_processor.vocab_q
    vocab_d = data_processor.vocab_d

    if args.create_vocabulary:
        print 'dump'
        with open(args.vocab_path+"en_{}_vocab_for_index.txt".format(args.doc_lang),"wb") as f_q,\
             open(args.vocab_path+"{}_vocab_for_index.txt".format(args.doc_lang),'wb') as f_d:
            cPickle.dump(dict(vocab_q), f_q)
            cPickle.dump(dict(vocab_d), f_d)
        print 'done'

    # model setup
    if args.deep:
        model = DeepDSSM(args, len(vocab_q), len(vocab_d))
    else:
        model = SimpleDSSM(args, len(vocab_q), len(vocab_d))

    # load embedding
    if args.load_embedding:
        model.load_embeddings(args, vocab_q, "query")
        model.load_embeddings(args, vocab_d, "document")

    # optimizer setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2), eps=config.epsilon)


    for epoch in range(1, 1+epoch_list):

        model.train()
        for batch in batchs:

            model.zero_grad()

            loss = model.forward(batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gpu  ', dest='gpu', type=int,default=-1, help='negative value indicates CPU')

    # training parameter
    parser.add_argument('--epoch', dest='epoch', type=int,default=5, help='number of epochs to learn')
    parser.add_argument('--batchsize', dest='batchsize', type=int,default=32, help='learning minibatch size')
    parser.add_argument('--doc_lang', dest='doc_lang', type=str, default="ja")
    parser.add_argument('--encode_type', dest='encode_type', type=str, default="cnn")
    parser.add_argument('--op', dest='optimizer', type=str, default="adam")
    parser.add_argument('--sub_sample_data_limit', type=int, default=31716, help='')

    # training flag
    parser.add_argument('--deep', action='store_true', help='')
    parser.add_argument('--load_embedding', action='store_true', help='')
    parser.add_argument('--load_parameter', action='store_true', help='')
    parser.add_argument('--load_snapshot', action='store_true', help='')
    parser.add_argument('--weighted_sum', action='store_true', help='')
    parser.add_argument('--sub_sample_train', action='store_true', help='')
    parser.add_argument('--test', action='store_true', help='use tiny dataset')

    # other flag
    parser.add_argument('--create_vocabulary', action='store_true', help='')
    parser.add_argument('--extract_parameter', action='store_true', help='')

    # model parameter
    parser.add_argument('--n_layer', dest='n_layer', type=int, default=2, help='# of layer')
    parser.add_argument('--n_hdim', dest='n_hdim', type=int, default=200, help='dimension of hidden layer')
    parser.add_argument('--vocab_size', dest='vocab_size', type=int, default=100000, help='')
    parser.add_argument('--cnn_out_channels', dest='cnn_out_channels', type=int, default=100, help='')
    parser.add_argument('--cnn_ksize', dest='cnn_ksize', type=int, default=4, help='')
    parser.add_argument('--embed_dim', dest='embed_dim', type=int, default=100, help='# of layer')

    # data path
    parser.add_argument('--vocab_path', dest='vocab_path', type=str, default=HOME+"/clir/vocab/")
    parser.add_argument('--data_path', dest='data_path', type=str,default="/export/a13/shota/clir/")
    parser.add_argument('--vec_path', dest='vec_path', type=str, default=HOME+"/word2vec/trunk/")
    parser.add_argument('--model_path', dest='model_path', type=str, default='')
    args = parser.parse_args()
    main(args)


