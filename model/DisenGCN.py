import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
from scipy.sparse import csr_matrix
from recbole.model.loss import BPRLoss, EmbLoss

from model.BaseModel import BaseModel


class NeibRoutLayer(nn.Module):
    def __init__(self, num_caps, niter, tau=1.0):
        super(NeibRoutLayer, self).__init__()
        self.num_cap = num_caps
        self.n_iter = niter
        self.tau = tau

    def forward(self, x, edge_index):
        # x: d-dimensional node representations.
        # edge_index: a list that contains m edges.
        # src: the source nodes of the edges.
        # trg: the target nodes of the edges.
        num_edge, src, trg = edge_index.shape[1], edge_index[0], edge_index[1]
        num_node, num_dim = x.shape
        num_dim_per_node = num_dim // self.num_cap

        # Normalize different user\item interest
        x = fn.normalize(x.view(num_node, self.num_cap, num_dim_per_node), dim=2).view(num_node, num_dim)
        # embedding of the source node
        z = x[src].view(num_edge, self.num_cap, num_dim_per_node)
        scatter_idx = trg.view(num_edge, 1).expand(num_edge, num_dim)
        u = x
        for clus_iter in range(self.n_iter):
            # source embedding * target embedding
            p = (z * u[trg].view(num_edge, self.num_cap, num_dim_per_node)).sum(dim=2)
            p = fn.softmax(p / self.tau, dim=1)

            scatter_src = (z * p.view(num_edge, self.num_cap, 1)).view(num_edge, num_dim)
            u = torch.zeros(num_node, num_dim, device=x.device)
            u.scatter_add_(0, scatter_idx, scatter_src)
            u += x

            u = fn.normalize(u.view(num_node, self.num_cap, num_dim_per_node), dim=2).view(num_node, num_dim)
        return u


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


class DisenGCN(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, ncaps: int, n_iter: int, num_layers: int,
                 dropout: float, interaction_matrix: csr_matrix, tag_table: dict):
        """
        :params num_users: the num of users
        :params num_items: the num if items
        :params nfeat: dimension of a node's input feature
        :params ncaps: number of capsules/channels/factors per layer
        :params n_iter: routing iterations
        :params num_layers: num layers of routing layers
        :params dropout: dropout ratio
        :params tag_table: tag_id and its corresponding item_ids
        :params interaction_matrix: the input sparse interaction_matrix
        """
        super(DisenGCN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, nfeat)
        self.item_embedding = nn.Embedding(num_items, nfeat)
        self.pca = SparseInputLinear(nfeat, nfeat)
        self.num_users = num_users
        self.num_items = num_items
        self.tag_table = tag_table

        coo = interaction_matrix.tocoo().astype(np.float32)
        src, trg = coo.row.tolist(), coo.col.tolist()
        self.edge_index = torch.LongTensor([src, trg]).to('cuda')

        conv_ls = []
        for i in range(num_layers):
            conv = NeibRoutLayer(ncaps, n_iter)
            self.add_module('conv_%d' % i, conv)
            conv_ls.append(conv)
        self.conv_ls = conv_ls

        self.dropout = dropout
        self.f = nn.Sigmoid()
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        self.reg_weight = 0.1

    def _dropout(self, x):
        return fn.dropout(x, self.dropout, training=self.training)

    def calculate_loss(self, user, pos_item, neg_item):
        user_emb, item_emb = self.forward()
        user_emb = user_emb[user]
        pos_item_emb = item_emb[pos_item]
        neg_item_emb = item_emb[neg_item]

        pos_scores = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_scores = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        return mf_loss, reg_loss

    def forward(self):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        x = fn.relu(self.pca(x))
        for conv in self.conv_ls:
            x = self._dropout(fn.relu(conv(x, self.edge_index)))
        users_emb, items_emb = x[:self.num_users, :], x[self.num_users:, :]
        return users_emb, items_emb

    def get_user_ratings(self, user):
        users_emb, items_emb = self.forward()
        users_emb = users_emb[user]
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
