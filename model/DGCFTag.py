import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.sparse import csr_matrix
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss

from model.BaseModel import BaseModel


class DGCFTAG(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, ncaps: int, n_iter: int, num_layers: int,
                 dropout: float, interaction_matrix: csr_matrix, tag_table: dict, drop_tag_ratio: float, reg_w: float):
        super(DGCFTAG, self).__init__()
        interaction_matrix = interaction_matrix.tocoo().astype(np.float32)
        self.num_items = num_items
        self.num_users = num_users
        self.ncaps = ncaps
        self.n_iterations = n_iter
        self.num_layers = num_layers
        self.dropout = dropout
        iid_tedge, tid_tedge = self._prepare_node_information_(tag_table, drop_tag_ratio)
        iid_tedge = [iid + self.num_users for iid in iid_tedge]
        tid_tedge = [tid + self.num_users + self.num_items for tid in tid_tedge]
        row = interaction_matrix.row.tolist()
        col = interaction_matrix.col.tolist()
        col = [iid + self.num_users for iid in col]
        all_h_list = row + col + iid_tedge + tid_tedge
        all_t_list = col + row + tid_tedge + iid_tedge
        num_edge = len(all_h_list)
        edge_ids = range(num_edge)
        self.all_h_list = torch.LongTensor(all_h_list).to('cuda')
        self.all_t_list = torch.LongTensor(all_t_list).to('cuda')
        edge2head = torch.LongTensor([all_h_list, edge_ids]).to('cuda')
        head2edge = torch.LongTensor([edge_ids, all_h_list]).to('cuda')
        tail2edge = torch.LongTensor([edge_ids, all_t_list]).to('cuda')
        val_one = torch.ones_like(self.all_h_list).float().to('cuda')
        num_node = self.num_users + self.num_items + self.num_tags
        self.edge2head_mat = self._build_sparse_tensor(edge2head, val_one, (num_node, num_edge))
        self.head2edge_mat = self._build_sparse_tensor(head2edge, val_one, (num_edge, num_node))
        self.tail2edge_mat = self._build_sparse_tensor(tail2edge, val_one, (num_edge, num_node))
        self.num_edge = num_edge
        self.num_node = num_node

        self.user_embedding = nn.Embedding(self.num_users, nfeat)
        self.item_embedding = nn.Embedding(self.num_items, nfeat)
        self.tag_embedding = nn.Embedding(self.num_tags, nfeat)
        self.softmax = torch.nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.reg_w = reg_w
        self.apply(xavier_normal_initialization)

    def _build_sparse_tensor(self, indices, values, size):
        return torch.sparse.FloatTensor(indices, values, size).to('cuda')

    def build_matrix(self, A_values):
        norm_A_values = self.softmax(A_values)
        factor_edge_weight = []
        for i in range(self.ncaps):
            tp_values = norm_A_values[:, i].unsqueeze(1)
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

    def forward(self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight, self.tag_embedding.weight],
                                   dim=0)
        all_embeddings = [ego_embeddings.unsqueeze(1)]
        A_values = torch.ones((self.num_edge, self.ncaps)).to('cuda')
        A_values = Variable(A_values, requires_grad=True)
        for k in range(self.num_layers):
            layer_embeddings = []
            ego_layer_embeddings = torch.chunk(ego_embeddings, self.ncaps, 1)
            for t in range(0, self.n_iterations):
                iter_embeddings = []
                A_iter_values = []
                factor_edge_weight = self.build_matrix(A_values=A_values)
                for i in range(0, self.ncaps):
                    edge_weight = factor_edge_weight[i]
                    # [num_edge,num_node] x [num_node,cap_dim] = [num_edge, cap_dim]
                    edge_val = torch.sparse.mm(self.tail2edge_mat, ego_layer_embeddings[i])
                    distanged_edge_val = edge_val * edge_weight
                    # [num_node, num_edge] x [num_edge, cap_dim] = [num_edge, cap_dim]
                    factor_embeddings = torch.sparse.mm(self.edge2head_mat, distanged_edge_val)
                    # [num_node, cap_dim]
                    iter_embeddings.append(factor_embeddings)
                    if t == self.n_iterations - 1:
                        layer_embeddings = iter_embeddings
                    head_factor_embeddings = torch.index_select(factor_embeddings, dim=0, index=self.all_h_list)
                    tail_factor_embeddings = torch.index_select(ego_layer_embeddings[i], dim=0, index=self.all_t_list)
                    head_factor_embeddings = F.normalize(head_factor_embeddings, p=2, dim=1)
                    tail_factor_embeddings = F.normalize(tail_factor_embeddings, p=2, dim=1)
                    A_factor_values = torch.sum(head_factor_embeddings * torch.tanh(tail_factor_embeddings), dim=1,
                                                keepdim=True)
                    A_iter_values.append(A_factor_values)
                A_iter_values = torch.cat(A_iter_values, dim=1)
                A_values = A_values + A_iter_values

            side_embeddings = torch.cat(layer_embeddings, dim=1)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings.unsqueeze(1)]

        all_embeddings = torch.cat(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        u_g_embeddings = all_embeddings[:self.num_users, :]
        i_g_embeddings = all_embeddings[self.num_users:self.num_users + self.num_items, :]

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
        return mf_loss + self.reg_w * reg_loss

    def get_user_ratings(self, user):
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        rating = torch.matmul(u_embeddings, item_embeddings.t())
        return rating

    def _prepare_node_information_(self, tag_table: dict, drop_tag_ratio=0.1):
        self.num_tags = max([int(key) for key in tag_table.keys()]) + 1
        tag_table = sorted(tag_table.items(), key=lambda x: len(x[1]), reverse=True)
        drop_tag_index = int(len(tag_table) * drop_tag_ratio)
        tag_table = tag_table[drop_tag_index:]
        src, trg = [], []

        for tid, iids in tag_table:
            for iid in iids:
                src.append(int(iid))
            trg.extend([int(tid)] * len(iids))

        return src, trg
