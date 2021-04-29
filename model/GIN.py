import numpy as np
import torch
import torch.autograd
import torch.nn as nn
from scipy.sparse import csr_matrix
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from torch_geometric.nn.conv.gin_conv import GINConv

from model.BaseModel import BaseModel


class GIN(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, num_layers: int, interaction_matrix: csr_matrix):
        super(GIN, self).__init__()
        interaction_matrix = interaction_matrix.tocoo().astype(np.float32)
        row = interaction_matrix.row.tolist()
        col = interaction_matrix.col.tolist()
        col = [iid + num_users for iid in col]

        self.edge_index = torch.LongTensor([row + col, col + row]).to('cuda')
        self.user_embedding = nn.Embedding(num_users, nfeat)
        self.item_embedding = nn.Embedding(num_items, nfeat)
        self.num_users = num_users

        self.convs = []
        for _ in range(num_layers):
            self.convs.append(GINConv(nn.Linear(nfeat, nfeat).cuda()).cuda())
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.apply(xavier_normal_initialization)

    def forward(self):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for conv in self.convs:
            x = conv(x, self.edge_index)
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
