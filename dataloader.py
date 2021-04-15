import os
import pandas as pd
import numpy as np
import itertools
import json
from os.path import join
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz, load_npz
import pickle
from tqdm import tqdm
from collections import Counter
import logging

class DataLoader(object):
    def __init__(self, dir_path: str, split_ratio: float = 0.8):
        dir_path = dir_path
        self.split_ratio = split_ratio
        self.n_item, self.n_user, self.n_tag, = 0, 0, 0
        self.train_data_size, self.test_data_size = 0, 0

        self.train_Gmaxtrix, test_Gmaxtrix = self._generate_graph_(dir_path)
        self.tag_table = self._generate_tag_table_(dir_path)
        self.train_pos_items = self.get_user_pos_items(list(range(self.n_user)))
        self.test_dict = self._generate_test_data_(test_Gmaxtrix)

    def _generate_cleaned_csv_(self, dir_path: str) -> pd.DataFrame:
        save_path = join(dir_path, "graph.csv")
        if os.path.exists(save_path):
            logging.info("直接加载已处理好的图数据表")
            return pd.read_csv(save_path)

        logging.info("开始生成清洗后的数据表")
        inter_csv = pd.read_csv(join(dir_path, "inter"), sep="\t", usecols=["user_id:token", "item_id:token"], )

        user_inter_num = Counter(inter_csv["user_id:token"].values)
        item_inter_num = Counter(inter_csv["item_id:token"].values)
        ban_users = {id_ for id_ in user_inter_num if user_inter_num[id_] < 15}
        ban_items = {id_ for id_ in item_inter_num if item_inter_num[id_] < 15}

        dropped_inter = pd.Series(False, index=inter_csv.index)
        dropped_inter |= inter_csv["user_id:token"].isin(ban_users)
        dropped_inter |= inter_csv["item_id:token"].isin(ban_items)
        dropped_index = inter_csv.index[dropped_inter]
        inter_csv.drop(dropped_index, inplace=True)
        logging.info("完成用户和物品交互关系的过滤")
        inter_csv.to_csv(join(dir_path, "clean_inter"), index=False)

        user_encoder = LabelEncoder()
        inter_csv["user_id:token"] = user_encoder.fit_transform(inter_csv["user_id:token"].values.tolist())
        logging.info("完成用户id的转换")

        item_encoder = LabelEncoder()
        inter_csv["item_id:token"] = item_encoder.fit_transform(inter_csv["item_id:token"].values.tolist())
        logging.info("完成物品id的转换")

        inter_csv.to_csv(save_path, index=False)
        pickle.dump(user_encoder, open(join(dir_path, 'user_encoder.pkl'), 'wb'))
        pickle.dump(item_encoder, open(join(dir_path, 'item_encoder.pkl'), 'wb'))
        logging.info("完成图数据表的生成，保存user、item的映射关系")
        return inter_csv

    def _generate_graph_(self, dir_path: str):
        train_graph_path = join(dir_path, "train_Gmaxtrix.npz")
        test_graph_path = join(dir_path, "test_Gmaxtrix.npz")
        if os.path.exists(train_graph_path) and os.path.exists(test_graph_path):
            logging.info("直接加载已有的训练图和测试图")
            train_Gmaxtrix, test_Gmaxtrix = load_npz(train_graph_path), load_npz(test_graph_path)
            self.n_user, self.n_item = train_Gmaxtrix.shape
            self.train_data_size = int(train_Gmaxtrix.sum())
            self.test_data_size = test_Gmaxtrix
            return train_Gmaxtrix, test_Gmaxtrix

        graph_csv = self._generate_cleaned_csv_(dir_path)
        train_user, train_item = [], []
        test_user, test_item = [], []

        for uid, iids in tqdm(graph_csv.groupby("user_id:token")):
            iids = list(iids["item_id:token"].values)
            iids_split_idx = int(len(iids) * self.split_ratio)

            train_iids = iids[:iids_split_idx]
            train_user.extend([uid] * len(train_iids))
            train_item.extend(train_iids)
            self.train_data_size += len(train_iids)

            test_iids = iids[iids_split_idx:]
            test_user.extend([uid] * len(test_iids))
            test_item.extend(test_iids)
            self.test_data_size += len(test_iids)
            self.n_item = max(self.n_item, max(iids))
            self.n_user = max(self.n_user, uid)

        self.n_item += 1
        self.n_user += 1
        test_user, test_item = np.array(test_user), np.array(test_item)
        train_user, train_item = np.array(train_user), np.array(train_item)
        train_Gmaxtrix = csr_matrix((np.ones(len(train_user)), (train_user, train_item)),
                                    shape=(self.n_user, self.n_item))
        test_Gmaxtrix = csr_matrix((np.ones(len(test_user)), (test_user, test_item)),
                                   shape=(self.n_user, self.n_item))
        save_npz(train_graph_path, train_Gmaxtrix)
        save_npz(test_graph_path, test_Gmaxtrix)
        logging.info("完成训练图和测试图的生成")
        return train_Gmaxtrix, test_Gmaxtrix

    def _generate_test_data_(self, test_Gmatrix: csr_matrix):
        test_dict = {}
        for uid in range(test_Gmatrix.shape[0]):
            iids = test_Gmatrix.getrow(5684).indices.tolist()
            if len(iids) > 0:
                test_dict[uid] = iids
        return test_dict

    def _generate_tag_table_(self, dir_path: str) -> dict:
        tag_tables_path = join(dir_path, "tag_tables.json")
        if os.path.exists(tag_tables_path):
            logging.info("直接加载已处理好的Tag字典")
            tag_table = json.load(open(tag_tables_path))
            self.n_tag = max([int(i) for i in list(tag_table.keys())])
            return tag_table

        graph_csv = pd.read_csv(join(dir_path, "clean_inter"), dtype=str)
        item_df = pd.read_csv(join(dir_path, "item"), sep="\t", usecols=["item_id:token", "categories:token_seq"],
                              dtype=str)
        graph_csv = pd.merge(graph_csv, item_df, how='inner')
        del graph_csv["user_id:token"]
        graph_csv.drop_duplicates(inplace=True)

        tag_encoder = LabelEncoder()
        graph_csv["categories:token_seq"] = graph_csv["categories:token_seq"].apply(lambda x: x.split(", "))
        tag_list = list(itertools.chain(*(list(graph_csv["categories:token_seq"].values))))
        tag_encoder.fit(tag_list)
        graph_csv["categories:token_seq"] = graph_csv["categories:token_seq"].apply(
            lambda x: list(tag_encoder.transform(x)))
        logging.info("完成标签id的转换")
        pickle.dump(tag_encoder, open(join(dir_path, 'tag_encoder.pkl'), 'wb'))

        tag_tables = {}
        for iid, tids in tqdm(graph_csv.groupby("item_id:token")):
            tids = tids["categories:token_seq"].values[0]
            for tid in tids:
                tag_tables.setdefault(tid, []).append(iid)
        tag_tables = sorted(tag_tables.items(), key=lambda x: len(x[1]), reverse=True)
        tag_tables_split = int(0.1 * len(tag_tables))
        tag_tables = tag_tables[tag_tables_split:]
        tag_tables = [(str(table[0]), table[1]) for table in tag_tables]
        tag_tables = dict(tag_tables)
        json.dump(tag_tables, open(tag_tables_path, 'w'))
        logging.info("完成Tag字典的生成")
        return tag_tables

    def get_user_pos_items(self, users):
        posItems = []
        for user in users:
            pos_item = self.train_Gmaxtrix[user].nonzero()[1].tolist()
            posItems.append(pos_item)
        return posItems


if __name__ == '__main__':
    dataloader = DataLoader(dir_path="/home/wzy/LightGCN/data/amazon_toy")
