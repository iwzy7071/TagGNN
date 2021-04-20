from model.DisenGCN import DisenGCN
from model.DGCF import DGCF
from model.TagGCN import TagGCN
from model.BaseModel import BaseModel
from model.LightGCN import LightGCN
from data.Data import Data
from model.DGCFTag import DGCFTAG


class ModelFactory:
    @staticmethod
    def get_model(model_name, config: dict, dataloader: Data) -> BaseModel:
        # if model_name == 'DisenGCN':
        #     return DisenGCN(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=config["nfeat"],
        #                     ncaps=config["ncaps"], n_iter=config["n_iter"], num_layers=config["num_layers"],
        #                     dropout=config["dropout"], interaction_matrix=dataloader.train_Gmatrix,
        #                     tag_table=dataloader.tag_tables)
        if model_name == "DGCF":
            return DGCF(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=config["nfeat"],
                        ncaps=config["ncaps"], n_iter=config["n_iter"], num_layers=config["num_layers"],
                        dropout=config["dropout"], interaction_matrix=dataloader.train_Gmatrix,
                        tag_table=dataloader.tag_tables)
        if model_name == "TagGCN":
            return TagGCN(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=config["nfeat"],
                          num_caps=config["ncaps"], num_layers=config["num_layers"],
                          interaction_matrix=dataloader.train_Gmatrix, tag_table=dataloader.tag_tables)
        if model_name == 'LightGCN':
            return LightGCN(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=config["nfeat"],
                            num_layers=config["num_layers"], dropout=config["dropout"],
                            interaction_matrix=dataloader.train_Gmatrix)
        if model_name == 'DGCFTag':
            return DGCFTAG(num_users=dataloader.n_user, num_items=dataloader.n_item, nfeat=config["nfeat"],
                        ncaps=config["ncaps"], n_iter=config["n_iter"], num_layers=config["num_layers"],
                        dropout=config["dropout"], interaction_matrix=dataloader.train_Gmatrix,
                        tag_table=dataloader.tag_tables)
        raise NameError("Model Name not Found")
