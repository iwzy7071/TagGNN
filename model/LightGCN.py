import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from scipy.sparse import csr_matrix
from model.BaseModel import BaseModel


class LightGCN(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, num_layers: int, dropout: float,
                 interaction_matrix: csr_matrix):
        super(LightGCN, self).__init__()
        self.interaction_matrix = interaction_matrix.tocoo()
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=nfeat)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=nfeat)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.norm_adj_matrix = self.get_norm_adj_mat().to('cuda')
        self.apply(xavier_uniform_initialization)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.num_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.num_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def forward(self):
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings,
                                                               [self.num_users, self.num_items])
        return user_all_embeddings, item_all_embeddings

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

    def get_user_ratings(self, user):
        user_embeddings, item_embeddings = self.forward()
        u_embeddings = user_embeddings[user]
        rating = torch.matmul(u_embeddings, item_embeddings.t())
        return rating
