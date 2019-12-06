import torch
import torch.nn as nn
import torch.nn.functional as F
from src.dssm_model import SimpleDSSM, DeepDSSM

# NOTE hyper-parameters we use in VAT
# n_power: a number of power iteration for approximation of r_vadv
# XI: a small float for the approx. of the finite difference method
# epsilon: the value for how much deviate from original data point X





class VAT_Simple(SimpleDSSM):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, model, args, q_vocab_size, d_vocab_size):
        super(VAT_Simple, self).__init__(args, q_vocab_size, d_vocab_size)
        self.model = model
        self.n_power = args.n_power

        self.XI = 1
        self.epsilon = 2
        self.regression_loss = nn.L1Loss()


    def get_normalized_vector(self, d):
        # d needs to be a unit vector at each iteration

        d_norm = F.normalize(d.view(d.size(0), -1), dim=1, p=2).view(d.size())
        return d_norm


    def generate_virtual_adversarial_perturbation(self, qs_ul, ds_ul, sims_ul):

        # initialize random vector

        qs_ul_emb = self.model.q_word_embeds.forward(qs_ul)
        ds_ul_emb = self.model.d_word_embeds.forward(ds_ul)

        dq = torch.randn_like(qs_ul_emb)
        dd = torch.randn_like(ds_ul_emb)

        for _ in range(self.n_power):
            dq = self.XI * self.get_normalized_vector(dq).requires_grad_()
            dd = self.XI * self.get_normalized_vector(dd).requires_grad_()
            # run the model with perturbation
            sims_ul_v = self.model(qs_ul, ds_ul, dq, dd)
            dist = self.regression_loss(sims_ul, sims_ul_v)

            q_grads = torch.autograd.grad(dist, self.model.qs_emb, retain_graph=True)[0]
            d_grads = torch.autograd.grad(dist, self.model.ds_emb, retain_graph=True)[0]
            dq = q_grads.detach()
            dd = d_grads.detach()

        return self.epsilon * self.get_normalized_vector(dq), self.epsilon * self.get_normalized_vector(dd)



    def virtual_adversarial_loss(self, qs_ul, ds_ul, sims_ul):
        # generate the virtual adversarial loss for unlabeled data
        # generate perturbation
        rq_vadv, rd_vadv = self.generate_virtual_adversarial_perturbation(qs_ul, ds_ul, sims_ul)

        sims_ul_data = sims_ul.detach()
        sims_ul_vat = self.model(qs_ul, ds_ul, rq_vadv, rd_vadv)
        vat_loss = self.regression_loss(sims_ul_data, sims_ul_vat)
        return vat_loss


    def forward(self, qs_ul, ds_ul, sims_ul):

        vat_loss = self.virtual_adversarial_loss(qs_ul, ds_ul, sims_ul)

        return vat_loss  # already averaged
