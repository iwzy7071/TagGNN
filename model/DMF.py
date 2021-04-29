import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_
from recbole.model.layers import MLPLayers
from recbole.model.init import xavier_normal_initialization

from model.BaseModel import BaseModel


class DMF(BaseModel):
    def __init__(self, num_items, num_users, nfeat, interaction_matrix:):
        self.user_linear = nn.Linear(in_features=num_items, out_features=nfeat, bias=False)
        self.item_linear = nn.Linear(in_features=num_users, out_features=nfeat, bias=False)
        self.user_fc_layers = MLPLayers([nfeat] + [nfeat, nfeat])
        self.item_fc_layers = MLPLayers([nfeat] + [nfeat, nfeat])
        self.sigmoid = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.i_embedding = None
        self.interaction_matrix = interaction_matrix
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user = self.get_user_embedding(user)
        col_indices = self.history_user_id[item].flatten()
        row_indices = torch.arange(item.shape[0]).to(self.device). \
            repeat_interleave(self.history_user_id.shape[1], dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(item.shape[0], self.n_users)
        matrix_01.index_put_((row_indices, col_indices), self.history_user_value[item].flatten())
        item = self.item_linear(matrix_01)

        user = self.user_fc_layers(user)
        item = self.item_fc_layers(item)

        # cosine distance is replaced by dot product according the result of our experiments.
        vector = torch.mul(user, item).sum(dim=1)
        vector = self.sigmoid(vector)

        return vector

    def calculate_loss(self, interaction):
        # when starting a new epoch, the item embedding we saved must be cleared.
        if self.training:
            self.i_embedding = None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        if self.inter_matrix_type == '01':
            label = interaction[self.LABEL]
        elif self.inter_matrix_type == 'rating':
            label = interaction[self.RATING] * interaction[self.LABEL]
        output = self.forward(user, item)

        label = label / self.max_rating
        loss = self.bce_loss(output, label)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        return self.forward(user, item)

    def get_user_embedding(self, user):
        col_indices = self.history_item_id[user].flatten()
        row_indices = torch.arange(user.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(self.history_item_id.shape[1], dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(user.shape[0], self.n_items)
        matrix_01.index_put_((row_indices, col_indices), self.history_item_value[user].flatten())
        user = self.user_linear(matrix_01)

        return user

    def get_item_embedding(self):
        interaction_matrix = self.interaction_matrix.tocoo()
        row = interaction_matrix.row
        col = interaction_matrix.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(interaction_matrix.data)
        item_matrix = torch.sparse.FloatTensor(i, data, torch.Size(interaction_matrix.shape)).to(self.device). \
            transpose(0, 1)
        item = torch.sparse.mm(item_matrix, self.item_linear.weight.t())

        item = self.item_fc_layers(item)
        return item

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        u_embedding = self.get_user_embedding(user)
        u_embedding = self.user_fc_layers(u_embedding)

        if self.i_embedding is None:
            self.i_embedding = self.get_item_embedding()

        similarity = torch.mm(u_embedding, self.i_embedding.t())
        similarity = self.sigmoid(similarity)
        return similarity.view(-1)
