import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.model.init import xavier_normal_initialization
import torch

from model.BaseModel import BaseModel


# torch random sample
# np.random.choice(strs)

class RoutingLayers(nn.Module):
    def __init__(self, num_caps: int, random_ratio: float, num_edges: int, num_users: int, tag_topk: int, num_tags: int,
                 num_items: int):
        super(RoutingLayers, self).__init__()
        self.num_caps = num_caps
        self.softmax = nn.Softmax(dim=1)
        self.random_ratio = random_ratio
        self.num_edges = num_edges
        self.num_users = num_users
        self.num_tags = num_tags
        self.tag_topk = tag_topk
        self.num_nodes = num_users + num_items

    def normalize_edge_index(self, edge_ncaps: torch.Tensor, src_trg2eid: torch.Tensor, eid2src_trg: torch.Tensor,
                             eid2trg_src: torch.Tensor) -> list:
        edge_weight_ncaps = []
        # 归一化所有兴趣下所有边的权重 [num_edge, cap_index]
        norm_edge_ncaps = self.softmax(edge_ncaps)
        for cap_index in range(self.num_caps):
            # 当前兴趣下边的权重 [num_edge, 1]
            edge_cap_vals = norm_edge_ncaps[:, cap_index].unsqueeze(1)

            # src_trg所对应的兴趣权重 [num_node,num_edge],[num_edge,1]
            src_trg_cap_values = torch.sparse.mm(src_trg2eid, edge_cap_vals)
            src_trg_cap_values = torch.clamp(src_trg_cap_values, min=1e-8)
            src_trg_cap_values = 1.0 / torch.sqrt(src_trg_cap_values)

            # [num_edge,num_node] x [num_node, 1] = [num_edge,1]
            eid_cap_value_src = torch.sparse.mm(eid2src_trg, src_trg_cap_values)
            # [num_edge,src+trg] x [trg+src, 1] = [num_edge,1]
            eid_cap_value_trg = torch.sparse.mm(eid2trg_src, src_trg_cap_values)
            # edge_weight: 在当前兴趣下 每条边的权重 [num_edge,1]
            edge_weight = edge_cap_vals * eid_cap_value_src * eid_cap_value_trg
            edge_weight_ncaps.append(edge_weight)

        return edge_weight_ncaps

    def forward(self, x: torch.Tensor, edge_ncaps: torch.Tensor, src_trg2eid: torch.Tensor, eid2src_trg: torch.Tensor,
                eid2trg_src: torch.Tensor, iid2tid: torch.Tensor, tid2iid: torch.Tensor, src_trg: torch.Tensor,
                trg_src: torch.Tensor):
        x = torch.chunk(x, self.num_caps, 1)
        edge_weight_ncaps = self.normalize_edge_index(edge_ncaps, src_trg2eid, eid2src_trg, eid2trg_src)
        new_edge_ncaps = []
        new_x = []
        for index in range(self.num_caps):
            # [num_edge, 1]
            edge_weight_cap = edge_weight_ncaps[index]
            edge_sample_index = torch.multinomial(edge_weight_cap[self.num_edges // 2:].squeeze(dim=1),
                                                  int(self.random_ratio * len(edge_weight_cap)))
            edge_sample_index = self.num_edges // 2 + edge_sample_index
            # [num_edge, 1]
            edge_sample_cap = torch.zeros_like(edge_weight_cap)
            edge_sample_cap[edge_sample_index] = 1
            # [src + trg, num_edge] x [num_edge, 1] = [src + trg, 1]
            src_trg_select = torch.sparse.mm(src_trg2eid, edge_sample_cap)
            iid_select = src_trg_select[self.num_users:]
            # [num_tags,num_items] x [num_items,1] = [num_tags,1]
            tag_count = torch.sparse.mm(tid2iid, iid_select)
            tag_count = tag_count.squeeze(dim=-1)
            _, top_count_tag_index = torch.topk(tag_count, k=self.tag_topk, dim=0, largest=True)
            selected_tags = torch.zeros(self.num_tags).unsqueeze(dim=-1).to('cuda')
            # [num_tags,1]
            selected_tags[top_count_tag_index] = 1
            # [num_items,num_tags] x [num_tags,1] = [num_items,1]
            new_add_iids = torch.sparse.mm(iid2tid, selected_tags).squeeze(dim=1)
            new_add_iids = torch.nonzero(new_add_iids)
            new_add_iids = self.num_users + new_add_iids
            new_iid_edges = torch.combinations(new_add_iids.squeeze(dim=1))
            iid_src, iid_tid = new_iid_edges[:, 0], new_iid_edges[:, 1]
            new_iid_length = iid_src.size()[0]
            iid_src, iid_tid = iid_src.unsqueeze(dim=0), iid_tid.unsqueeze(dim=0)

            new_iid_edges = torch.cat([iid_src, iid_tid], dim=0).long()
            val_one = torch.ones(new_iid_length).to('cuda')
            # [num_node,num_node]
            new_iid_edges = torch.sparse.FloatTensor(new_iid_edges, val_one,
                                                     (self.num_nodes, self.num_nodes)).to('cuda')

            # [num_edge,num_node] x [num_node,cap_dim] = [num_edge,cap_dim]
            edge_val = torch.sparse.mm(eid2trg_src, x[index])
            edge_val = edge_val * edge_weight_cap
            # [num_node,num_edge] x [num_edge,cap_dim] = [num_node,cap_dim]
            ncap_embed = torch.sparse.mm(src_trg2eid, edge_val)
            # [num_node,num_node] x [num_node,cap_dim] = [num_node,cap_dim]
            new_item_ncap_embed = torch.sparse.mm(new_iid_edges, ncap_embed)
            # FIXME: ncap_embed = ncap_embed + new_item_ncap_embed
            new_x.append(ncap_embed)
            # [num_edge,cap_dim]
            src_trg_ncap_embed = torch.index_select(ncap_embed, dim=0, index=src_trg)
            # [num_edge,cap_dim]
            trg_src_ncap_embed = torch.index_select(x[index], dim=0, index=trg_src)
            head_factor_embeddings = F.normalize(src_trg_ncap_embed, p=2, dim=1)
            tail_factor_embeddings = F.normalize(trg_src_ncap_embed, p=2, dim=1)
            # [num_edge, 1]
            edge_ncap = torch.sum(head_factor_embeddings * torch.tanh(tail_factor_embeddings), dim=1,
                                  keepdim=True)
            new_edge_ncaps.append(edge_ncap)
        # [num_edge, ncap]
        new_edge_ncaps = torch.cat(new_edge_ncaps, dim=1)
        new_x = torch.cat(new_x, dim=1)
        return new_x, new_edge_ncaps


class TagGCN(BaseModel):
    def __init__(self, num_users: int, num_items: int, nfeat: int, num_caps: int, num_layers: int,
                 interaction_matrix: csr_matrix, tag_table: dict, tag_drop_ratio=0.3, random_ratio=0.1, tag_topk=2):
        super(TagGCN, self).__init__()

        # 处理用户、物品和标签的基本信息
        self.num_users = num_users
        self.num_items = num_items
        self.num_caps = num_caps
        self.num_tags = 0

        self.tid2iid, self.iid2tid = self._prepare_node_information_(tag_table, tag_drop_ratio)
        self.user_embedding = nn.Embedding(num_users, nfeat)
        self.item_embedding = nn.Embedding(num_items, nfeat)

        # 创建图解耦的边
        coo = interaction_matrix.tocoo().astype(np.float32)
        src, trg = coo.row.tolist(), coo.col.tolist()
        trg = [iid + self.num_users for iid in trg]
        self.num_edges = len(src) + len(trg)
        eids = range(self.num_edges)
        self.num_nodes = self.num_users + self.num_items
        src_trg = src + trg
        trg_src = trg + src
        self.src_trg = torch.LongTensor(src_trg).to('cuda')
        self.trg_src = torch.LongTensor(trg_src).to('cuda')

        src_trg2eid = torch.LongTensor([src_trg, eids]).to('cuda')
        eid2src_trg = torch.LongTensor([eids, src_trg]).to('cuda')
        eid2trg_src = torch.LongTensor([eids, trg_src]).to('cuda')
        val_one = torch.ones(self.num_edges).float().to('cuda')
        self.src_trg2eid = torch.sparse.FloatTensor(src_trg2eid, val_one, (self.num_nodes, self.num_edges)).to('cuda')
        self.eid2src_trg = torch.sparse.FloatTensor(eid2src_trg, val_one, (self.num_edges, self.num_nodes)).to('cuda')
        self.eid2trg_src = torch.sparse.FloatTensor(eid2trg_src, val_one, (self.num_edges, self.num_nodes)).to('cuda')
        print(self.eid2trg_src)
        exit()
        self.edge_ncaps = Variable(torch.ones((self.num_edges, self.num_caps)).to('cuda'), requires_grad=True)
        self.convs = []

        # 创建RoutingLayers
        for i in range(num_layers):
            conv = RoutingLayers(num_caps, random_ratio, self.num_edges, num_users, tag_topk, self.num_tags,
                                 num_items)
            self.add_module('routing_conv%d' % i, conv)
            self.convs.append(conv)

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.apply(xavier_normal_initialization)

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
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        edge_ncaps = self.edge_ncaps
        for conv in self.convs:
            x, edge_ncaps = conv(x, edge_ncaps, self.src_trg2eid, self.eid2src_trg, self.eid2trg_src,
                                 self.iid2tid, self.tid2iid, self.src_trg, self.trg_src)
        user_embeddings, item_embeddings = torch.split(x, [self.num_users, self.num_items])
        return user_embeddings, item_embeddings

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

        val_one = torch.ones(len(src))
        iid2tid = torch.LongTensor([src, trg])
        iid2tid = torch.sparse.FloatTensor(iid2tid, val_one, (self.num_items, self.num_tags)).to('cuda')

        tid2iid = torch.LongTensor([trg, src])
        tid2iid = torch.sparse.FloatTensor(tid2iid, val_one, (self.num_tags, self.num_items)).to('cuda')
        return tid2iid, iid2tid
