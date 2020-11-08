import torch
import torch.nn as nn

from src.dt_processor import DataProcessor
from src.utils import numerize



class generate_index(DataProcessor):


        def generate_train_index(self):

            """generate index for training data"""

        i = 0
        for batch in data_processor.generate_train_batch(args.train_batchsize, args.is_shuffle):
            rels, qs, ds = numerize(batch, vocab_q, vocab_d)

            if args.use_gpu:
                rels, qs, ds = rels.to(device), qs.to(device), ds.to(device)

            sims = model(qs, ds, rels)
            model.zero_grad()

            loss = model.cal_loss(sims, rels)

            if args.use_adv:
                loss.backward(retain_graph=True)
                q_grads = torch.autograd.grad(loss, model.qs_input, retain_graph=True)[0]
                d_grads = torch.autograd.grad(loss, model.ds_input, retain_graph=True)[0]

                sims = model(qs, ds, rels, q_grads, d_grads)
                loss_adv = model.cal_loss(sims, rels)
                loss += 0.5 * loss_adv

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            loss_total.append(loss.item())

            if i % 10000 == 0:
                print(i)

            i += 1

        print('The training loss at Epoch ', epoch, 'is ', np.mean(loss_total))
