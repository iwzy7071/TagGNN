import numpy as np
import scipy.sparse as sp
import torch
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from scipy.sparse import csr_matrix
from model.BaseModel import BaseModel


class LightGCNTag(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, num_layers: int, dropout: float,
                 interaction_matrix: csr_matrix, reg_w: float, tag_table: dict, drop_tag_ratio: float):
        super(LightGCNTag, self).__init__()
        self.num_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items
        iid_tedge, tid_tedge = self._prepare_node_information_(tag_table, drop_tag_ratio)

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=nfeat)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=nfeat)
        self.tag_embedding = torch.nn.Embedding(num_embeddings=self.num_tags, embedding_dim=nfeat)

        iid_tedge = [iid + self.num_users for iid in iid_tedge]
        tid_tedge = [tid + self.num_users + self.num_items for tid in tid_tedge]

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.norm_adj_matrix = self.get_norm_adj_mat(interaction_matrix, iid_tedge, tid_tedge).to('cuda')
        self.reg_w = reg_w
        self.apply(xavier_uniform_initialization)

    def get_norm_adj_mat(self, interaction_matrix, iid_tedge, tid_tedge):
        dok_matrix = sp.dok_matrix((self.num_users + self.num_items + self.num_tags,
                                    self.num_users + self.num_items + self.num_tags), dtype=np.float32)
        interaction_matrix_t = interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(interaction_matrix.row, interaction_matrix.col + self.num_users), [1] * interaction_matrix_t.nnz))
        data_dict.update(dict(zip(zip(interaction_matrix_t.row + self.num_users, interaction_matrix_t.col),
                                  [1] * interaction_matrix_t.nnz)))
        dok_matrix._update(data_dict)

        data_dict = dict(zip(zip(iid_tedge, tid_tedge), [1] * len(iid_tedge)))
        data_dict.update(dict(zip(zip(tid_tedge, iid_tedge), [1] * len(iid_tedge))))
        dok_matrix._update(data_dict)

        sumArr = (dok_matrix > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * dok_matrix * D
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
