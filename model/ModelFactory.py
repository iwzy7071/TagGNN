from model.DisenGCN import DisenGCN
from model.DGCF import DGCF
from model.BaseModel import BaseModel
from dataloader import DataLoader


class ModelFactory:
    @staticmethod
    def get_model(model_name, config: dict, dataloader: DataLoader) -> BaseModel:
        if model_name == 'DisenGCN':
            return DisenGCN(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=config["nfeat"],
                            ncaps=config["ncaps"], n_iter=config["n_iter"], num_layers=config["num_layers"],
                            dropout=config["dropout"], interaction_matrix=dataloader.train_Gmaxtrix,
                            tag_table=dataloader.tag_table)
        if model_name == "DGCF":
            return DGCF(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=config["nfeat"],
                            ncaps=config["ncaps"], n_iter=config["n_iter"], num_layers=config["num_layers"],
                            dropout=config["dropout"], interaction_matrix=dataloader.train_Gmaxtrix,
                            tag_table=dataloader.tag_table)
