from model.DisenGCN import DisenGCN
from model.DGCF import DGCF
from model.BaseModel import BaseModel
from model.LightGCN import LightGCN
from data.Data import Data
from model.DGCFTag import DGCFTAG
from model.GIN import GIN
from model.LightGCNTag import LightGCNTag


class ModelFactory:
    @staticmethod
    def get_model(model_name, dataloader: Data) -> BaseModel:
        if model_name == "DGCF":
            return DGCF(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=64,
                        ncaps=4, n_iter=3, num_layers=1, dropout=0, interaction_matrix=dataloader.train_Gmatrix,
                        tag_table=dataloader.tag_tables, reg_w=1e-3)
        if model_name == "DisenGCN":
            return DisenGCN(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=64, ncaps=4, num_layers=3,
                            interaction_matrix=dataloader.train_Gmatrix)
        if model_name == 'LightGCN':
            return LightGCN(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=64,
                            num_layers=3, dropout=0, interaction_matrix=dataloader.train_Gmatrix, reg_w=1e-3)
        if model_name == 'DGCFTag':
            return DGCFTAG(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=64,
                           ncaps=2, n_iter=3, num_layers=1, dropout=0, interaction_matrix=dataloader.train_Gmatrix,
                           tag_table=dataloader.tag_tables, drop_tag_ratio=0.3, reg_w=1e-3)
        if model_name == 'GIN':
            return GIN(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=64, num_layers=3,
                       interaction_matrix=dataloader.train_Gmatrix)
        if model_name == "LightGCNTag":
            return LightGCNTag(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=32, num_layers=1,
                               dropout=0, interaction_matrix=dataloader.train_Gmatrix,
                               tag_table=dataloader.tag_tables, drop_tag_ratio=0.3, reg_w=1e-3)
        raise NameError("Model Name not Found")
