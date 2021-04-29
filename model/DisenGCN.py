import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
from scipy.sparse import csr_matrix
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss

from model.BaseModel import BaseModel


class NeibRoutLayer(nn.Module):
    def __init__(self, num_caps, niter, tau=1.0):
        super(NeibRoutLayer, self).__init__()
        self.k = num_caps
        self.niter = niter
        self.tau = tau

    def forward(self, x, src_trg):
        m, src, trg = src_trg.shape[1], src_trg[0], src_trg[1]
        n, d = x.shape
        k, delta_d = self.k, d // self.k
        x = fn.normalize(x.view(n, k, delta_d), dim=2).view(n, d)
        z = x[src].view(m, k, delta_d)
        u = x
        scatter_idx = trg.view(m, 1).expand(m, d)
        for clus_iter in range(self.niter):
            p = (z * u[trg].view(m, k, delta_d)).sum(dim=2)
            p = fn.softmax(p / self.tau, dim=1)
            scatter_src = (z * p.view(m, k, 1)).view(m, d)
            u = torch.zeros(n, d, device=x.device)
            u.scatter_add_(0, scatter_idx, scatter_src)
            u += x
            u = fn.normalize(u.view(n, k, delta_d), dim=2).view(n, d)
        return u


class DisenGCN(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, ncaps: int, num_layers: int,
                 interaction_matrix: csr_matrix):
        super(DisenGCN, self).__init__()
        self.linear = SparseInputLinear(nfeat, nfeat)
        self.conv = NeibRoutLayer(ncaps, num_layers)
        interaction_matrix = interaction_matrix.tocoo().astype(np.float32)
        row = interaction_matrix.row.tolist()
        col = interaction_matrix.col.tolist()
        col = [iid + num_users for iid in col]

        self.edge_index = torch.LongTensor([row + col, col + row]).to('cuda')
        self.user_embedding = nn.Embedding(num_users, nfeat)
        self.item_embedding = nn.Embedding(num_items, nfeat)
        self.num_users = num_users
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.apply(xavier_normal_initialization)

    def forward(self):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        x = fn.leaky_relu(self.linear(x))
        x = self.conv(x, self.edge_index)
        u_g_embeddings, i_g_embeddings = x[:self.num_users, :], x[self.num_users:, :]
        return u_g_embeddings, i_g_embeddings

    def calculate_loss(self, user, pos_item, neg_item):
        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        return mf_loss + 1e-3 * reg_loss

    def get_user_ratings(self, user):
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        rating = torch.matmul(u_embeddings, item_embeddings.t())
        return rating


class SparseInputLinear(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(SparseInputLinear, self).__init__()
        weight = np.zeros((inp_dim, out_dim), dtype=np.float32)
        weight = nn.Parameter(torch.from_numpy(weight))
        bias = np.zeros(out_dim, dtype=np.float32)
        bias = nn.Parameter(torch.from_numpy(bias))
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.weight, self.bias = weight, bias
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        return torch.mm(x, self.weight) + self.bias
