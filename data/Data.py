import pandas as pd
import numpy as np
import itertools
import json
import pickle
import logging
from os.path import join, exists as path_exist
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, save_npz, load_npz
from collections import Counter


class Data(object):
    def __init__(self, dir_path: str, split_ratios: list, min_inter: int):
        self.n_item = 0
        self.n_user = 0
        self.train_dsize = 0
        self.test_dsize = 0
        self.train_Gmatrix = None
        self.test_Gmatrix = None
        self.tag_tables = None
        self._prepare_matrix_(dir_path, min_inter, split_ratios)
        print("用户数量:", self.n_user, "物品数量:", self.n_item, "训练交互数量:", self.train_dsize,
              "稀疏程度:", self.train_dsize / self.n_item / self.n_user, "测试交互数量:", self.test_dsize)

    def _prepare_matrix_(self, dir_path, min_inter, split_ratios):
        train_Gmatrix_path = join(dir_path, "train_Gmatrix.npz")
        test_Gmatrix_path = join(dir_path, "test_Gmatrix.npz")
        tag_tables_path = join(dir_path, "tag_tables.json")

        if path_exist(train_Gmatrix_path) and path_exist(test_Gmatrix_path) and path_exist(tag_tables_path):
            logging.info("直接加载已有的训练图和测试图")
            train_Gmaxtrix = load_npz(train_Gmatrix_path)
            test_Gmaxtrix = load_npz(test_Gmatrix_path)

            self.tag_tables = json.load(open(tag_tables_path))
            self.n_user, self.n_item = train_Gmaxtrix.shape
            self.train_dsize = int(train_Gmaxtrix.sum())
            self.test_dsize = int(test_Gmaxtrix.sum())
            self.train_Gmatrix = train_Gmaxtrix
            self.test_Gmatrix = test_Gmaxtrix
            return

        inter_feat = self._generate_cleaned_inter_feat_(dir_path, min_inter)
        inter_feat, tag_tables = self._encode_feat_label_(dir_path, tag_tables_path, inter_feat)
        train_feat, test_feat = self._split_by_ratio_(inter_feat, split_ratios)
        self.train_Gmatrix, self.test_Gmatrix = self._build_matrix_(train_feat, test_feat)
        self.tag_tables = tag_tables
        save_npz(train_Gmatrix_path, self.train_Gmatrix)
        save_npz(test_Gmatrix_path, self.test_Gmatrix)

    def _generate_cleaned_inter_feat_(self, dir_path: str, min_inter: int) -> pd.DataFrame:
        logging.info("开始生成清洗后的数据表")
        inter_feat = pd.read_csv(join(dir_path, "inter"), sep="\t", usecols=["user_id:token", "item_id:token"], )
        user_feat = pd.DataFrame(inter_feat["user_id:token"]).drop_duplicates()
        item_feat = pd.DataFrame(inter_feat["item_id:token"]).drop_duplicates()
        user_inter_num = Counter(inter_feat["user_id:token"].values)
        item_inter_num = Counter(inter_feat["item_id:token"].values)

        while True:
            ban_users = self._get_illegal_ids_by_inter_num_("user_id:token", user_feat, user_inter_num,
                                                            min_num=min_inter)
            ban_items = self._get_illegal_ids_by_inter_num_("item_id:token", item_feat, item_inter_num,
                                                            min_num=min_inter)
            if len(ban_users) == 0 and len(ban_items) == 0:
                break

            if user_feat is not None:
                dropped_user = user_feat["user_id:token"].isin(ban_users)
                user_feat.drop(user_feat.index[dropped_user], inplace=True)

            if item_feat is not None:
                dropped_item = item_feat["item_id:token"].isin(ban_items)
                item_feat.drop(item_feat.index[dropped_item], inplace=True)

            dropped_inter = pd.Series(False, index=inter_feat.index)
            user_inter = inter_feat["user_id:token"]
            item_inter = inter_feat["item_id:token"]
            dropped_inter |= user_inter.isin(ban_users)
            dropped_inter |= item_inter.isin(ban_items)
            user_inter_num -= Counter(user_inter[dropped_inter].values)
            item_inter_num -= Counter(item_inter[dropped_inter].values)
            dropped_index = inter_feat.index[dropped_inter]
            logging.info(f'剔除了[{len(dropped_index)}]个交互关系')
            inter_feat.drop(dropped_index, inplace=True)
        print(inter_feat)
        logging.info("完成用户和物品交互关系的过滤")
        inter_feat.reset_index(drop=True, inplace=True)
        return inter_feat

    def _encode_feat_label_(self, dir_path: str, tag_table_path: str, inter_feat: pd.DataFrame):
        item_df = pd.read_csv(join(dir_path, "item"), sep="\t", usecols=["item_id:token", "categories:token_seq"],
                              dtype=str)
        inter_feat = pd.merge(inter_feat, item_df, how='inner')

        user_encoder = LabelEncoder()
        inter_feat["user_id:token"] = user_encoder.fit_transform(inter_feat["user_id:token"].values.tolist())
        logging.info("完成用户id的转换")

        item_encoder = LabelEncoder()
        inter_feat["item_id:token"] = item_encoder.fit_transform(inter_feat["item_id:token"].values.tolist())
        logging.info("完成物品id的转换")

        tag_encoder = LabelEncoder()
        inter_feat["categories:token_seq"] = inter_feat["categories:token_seq"].apply(lambda x: x.split(", "))
        tag_list = list(itertools.chain(*(list(inter_feat["categories:token_seq"].values))))
        tag_encoder.fit(tag_list)
        inter_feat["categories:token_seq"] = inter_feat["categories:token_seq"].apply(
            lambda x: list(tag_encoder.transform(x)))
        logging.info("完成标签id的转换")

        tag_tables = {}
        for iid, tids in inter_feat.groupby("item_id:token"):
            tids = tids["categories:token_seq"].values[0]
            for tid in tids:
                tag_tables.setdefault(int(tid), set()).add(int(iid))
        logging.info("完成Tag字典的生成")

        pickle.dump(user_encoder, open(join(dir_path, 'user_encoder.pkl'), 'wb'))
        pickle.dump(item_encoder, open(join(dir_path, 'item_encoder.pkl'), 'wb'))
        pickle.dump(tag_encoder, open(join(dir_path, 'tag_encoder.pkl'), 'wb'))

        for key in tag_tables.keys():
            tag_tables[key] = list(tag_tables[key])
        json.dump(tag_tables, open(tag_table_path, 'w'))

        logging.info("完成图数据表的生成，保存user、item、tag的映射关系")
        del inter_feat["categories:token_seq"]
        return inter_feat, tag_tables

    def _build_matrix_(self, train_feat: pd.DataFrame, test_feat: pd.DataFrame):
        train_user, train_item = [], []
        test_user, test_item = [], []

        for uid, iids in train_feat.groupby("user_id:token"):
            iids = list(iids["item_id:token"].values)
            train_user.extend([uid] * len(iids))
            train_item.extend(iids)
            self.train_dsize += len(iids)
            self.n_user = max(self.n_user, uid)
            self.n_item = max(self.n_item, max(iids))

        for uid, iids in test_feat.groupby("user_id:token"):
            iids = list(iids["item_id:token"].values)
            test_user.extend([uid] * len(iids))
            test_item.extend(iids)
            self.test_dsize += len(iids)
            self.n_user = max(self.n_user, uid)
            self.n_item = max(self.n_item, max(iids))

        self.n_item += 1
        self.n_user += 1
        test_user, test_item = np.array(test_user), np.array(test_item)
        train_user, train_item = np.array(train_user), np.array(train_item)
        train_Gmaxtrix = csr_matrix((np.ones(len(train_user)), (train_user, train_item)),
                                    shape=(self.n_user, self.n_item))
        test_Gmaxtrix = csr_matrix((np.ones(len(test_user)), (test_user, test_item)),
                                   shape=(self.n_user, self.n_item))
        logging.info("完成训练图和测试图的生成")
        return train_Gmaxtrix, test_Gmaxtrix

    def _split_by_ratio_(self, inter_feat: pd.DataFrame, ratios: list):
        grouped_inter_feat_index = self._grouped_index_(inter_feat["user_id:token"].to_numpy())
        next_index = [[] for _ in range(len(ratios))]
        for grouped_index in grouped_inter_feat_index:
            tot_cnt = len(grouped_index)
            split_ids = self._calculate_split_ids_(tot=tot_cnt, ratios=ratios)
            for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                index.extend(grouped_index[start:end])

        train_feat, test_feat = [inter_feat.loc[index] for index in next_index]
        # 保留一些冷启动的物品 从而提高TagGCN的模型性能
        # ban_items = list(set(test_feat["item_id:token"]) - set(train_feat["item_id:token"]))
        # dropped_index = test_feat["item_id:token"].isin(ban_items)
        # dropped_index = test_feat.index[dropped_index]
        # test_feat.drop(dropped_index, inplace=True)
        return train_feat, test_feat

    def _calculate_split_ids_(self, tot, ratios):
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def _grouped_index_(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index.values()

    def _get_illegal_ids_by_inter_num_(self, field, feat, inter_num, max_num=None, min_num=None):
        max_num = max_num or np.inf
        min_num = min_num or -1
        ids = {id_ for id_ in inter_num if inter_num[id_] < min_num or inter_num[id_] > max_num}
        if feat is not None:
            for id_ in feat[field].values:
                if inter_num[id_] < min_num:
                    ids.add(id_)
        return ids