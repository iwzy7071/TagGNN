import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.sparse import csr_matrix
from recbole.model.loss import BPRLoss, EmbLoss

from model.BaseModel import BaseModel


class DGCF(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, ncaps: int, n_iter: int, num_layers: int,
                 dropout: float, interaction_matrix: csr_matrix, tag_table: dict):
        super(DGCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.nfeat = nfeat
        self.num_layers = num_layers
        self.ncaps = ncaps
        self.n_iter = n_iter
        self.dropout = dropout
        self.tag_table = tag_table

        interaction_matrix = interaction_matrix
        coo = interaction_matrix.tocoo().astype(np.float32)
        row = coo.row.tolist()
        col = coo.col.tolist()
        col = [item_index + self.num_users for item_index in col]
        all_h_list = row + col
        all_t_list = col + row
        num_edge = len(all_h_list)
        edge_ids = range(num_edge)
        self.all_h_list = torch.LongTensor(all_h_list).to('cuda')
        self.all_t_list = torch.LongTensor(all_t_list).to('cuda')
        self.edge2head = torch.LongTensor([all_h_list, edge_ids]).to('cuda')
        self.head2edge = torch.LongTensor([edge_ids, all_h_list]).to('cuda')
        self.tail2edge = torch.LongTensor([edge_ids, all_t_list]).to('cuda')

        val_one = torch.ones_like(self.all_h_list).float().to('cuda')
        num_node = self.num_users + self.num_items
        self.edge2head_mat = torch.sparse.FloatTensor(self.edge2head, val_one, (num_node, num_edge)).to('cuda')
        self.head2edge_mat = torch.sparse.FloatTensor(self.head2edge, val_one, (num_edge, num_node)).to('cuda')
        self.tail2edge_mat = torch.sparse.FloatTensor(self.tail2edge, val_one, (num_edge, num_node)).to('cuda')
        self.num_edge = num_edge
        self.num_node = num_node

        self.user_embedding = nn.Embedding(self.num_users, self.nfeat)
        self.item_embedding = nn.Embedding(self.num_items, self.nfeat)
        self.f = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

    def _build_norm_matrix_(self, values):
        norm_values = self.softmax(values)
        factor_edge_weight = []
        for i in range(self.ncaps):
            tp_values = norm_values[:, i].unsqueeze(1)
            d_values = torch.sparse.mm(self.edge2head_mat, tp_values)
            d_values = torch.clamp(d_values, min=1e-8)
            try:
                assert not torch.isnan(d_values).any()
            except AssertionError:
                self.logger.info("d_values", torch.min(d_values), torch.max(d_values))

            d_values = 1.0 / torch.sqrt(d_values)
            head_term = torch.sparse.mm(self.head2edge_mat, d_values)
            tail_term = torch.sparse.mm(self.tail2edge_mat, d_values)
            edge_weight = tp_values * head_term * tail_term
            factor_edge_weight.append(edge_weight)
        return factor_edge_weight

    def get_user_ratings(self, user):
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        rating = self.f(torch.matmul(u_embeddings, item_embeddings.t()))
        return rating

    def calculate_loss(self, user, pos_item, neg_item):
        user_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)
        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        return mf_loss, reg_loss

    def forward(self):
        user_item_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [user_item_embeddings.unsqueeze(1)]
        edge_att_values = torch.ones((self.num_edge, self.ncaps)).to('cuda')
        edge_att_values = Variable(edge_att_values, requires_grad=True)
        for k in range(self.num_layers):
            layer_embeddings = []
            ego_layer_embeddings = torch.chunk(user_item_embeddings, self.ncaps, 1)
            for t in range(0, self.n_iter):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self._build_norm_matrix_(edge_att_values)
                for i in range(0, self.ncaps):
                    edge_weight = factor_edge_weight[i]
                    edge_val = torch.sparse.mm(self.tail2edge_mat, ego_layer_embeddings[i])
                    edge_val = edge_val * edge_weight
                    factor_embeddings = torch.sparse.mm(self.edge2head_mat, edge_val)
                    iter_embeddings.append(factor_embeddings)
                    if t == self.n_iter - 1:
                        layer_embeddings = iter_embeddings
                    head_factor_embeddings = torch.index_select(factor_embeddings, dim=0, index=self.all_h_list)
                    tail_factor_embeddings = torch.index_select(ego_layer_embeddings[i], dim=0, index=self.all_t_list)
                    head_factor_embeddings = F.normalize(head_factor_embeddings, p=2, dim=1)
                    tail_factor_embeddings = F.normalize(tail_factor_embeddings, p=2, dim=1)
                    A_factor_values = torch.sum(
                        head_factor_embeddings * torch.tanh(tail_factor_embeddings), dim=1, keepdim=True
                    )
                    A_iter_values.append(A_factor_values)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                edge_att_values = edge_att_values + A_iter_values
            side_embeddings = torch.cat(layer_embeddings, dim=1)
            user_item_embeddings = side_embeddings
            all_embeddings += [user_item_embeddings.unsqueeze(1)]
        all_embeddings = torch.cat(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        u_g_embeddings = all_embeddings[:self.num_users, :]
        i_g_embeddings = all_embeddings[self.num_users:, :]
        return u_g_embeddings, i_g_embeddings
