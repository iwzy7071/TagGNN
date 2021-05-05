import json
import random
from os.path import join, exists as path_exist
from scipy.sparse import csr_matrix, load_npz, save_npz


class Data(object):
    def __init__(self, dir_path: str, **kwargs):
        self.n_item = 0
        self.n_user = 0
        self.train_dsize = 0
        self.test_dsize = 0
        self.train_Gmatrix = None
        self.test_Gmatrix = None
        self.tag_tables = None
        cold_train_path = join(dir_path, "cold_train_Gmatrix.npz")
        code_test_path = join(dir_path, "cold_test_Gmatrix.npz")
        tag_tables_path = join(dir_path, "tag_tables.json")

        if path_exist(cold_train_path) and path_exist(code_test_path):
            print("直接加载预处理的冷启动训练集和测试集")
            self.train_Gmatrix = load_npz(cold_train_path)
            self.test_Gmatrix = load_npz(code_test_path)
        else:
            print("开始生成冷启动的训练集和测试集")
            ori_train_Gmatrix = load_npz(join(dir_path, "train_Gmatrix.npz"))
            ori_test_Gmatrix = load_npz(join(dir_path, "test_Gmatrix.npz"))
            self.train_Gmatrix, self.test_Gmatrix = self._generate_cold_start_matrix_(cold_train_path, code_test_path,
                                                                                      ori_train_Gmatrix,
                                                                                      ori_test_Gmatrix)

        self.tag_tables = json.load(open(tag_tables_path))
        self.n_user, self.n_item = self.train_Gmatrix.shape
        self.train_dsize = int(self.train_Gmatrix.sum())
        self.test_dsize = int(self.test_Gmatrix.sum())
        print("用户数量:", self.n_user, "物品数量:", self.n_item, "训练交互数量:", self.train_dsize,
              "稀疏程度:", self.train_dsize / self.n_item / self.n_user, "测试交互数量:", self.test_dsize)

    def _generate_cold_start_matrix_(self, train_path: str, test_path: str, train_Gmatrix: csr_matrix,
                                     test_Gmatrix: csr_matrix):
        # 合并训练图和测试图
        train_Gmatrix = train_Gmatrix.tolil()
        test_Gmatrix = test_Gmatrix.tocoo()
        row = test_Gmatrix.row
        col = test_Gmatrix.col
        train_Gmatrix[row, col] = 1
        test_Gmatrix = train_Gmatrix.copy()
        test_Gmatrix[:, :] = 0
        n_user, n_item = train_Gmatrix.shape

        for iid in random.sample(range(n_item), 1000):
            uids = train_Gmatrix[:, iid].nonzero()[0]
            test_Gmatrix[uids, [iid] * len(uids)] = 1
            train_Gmatrix[:, iid] = 0
        train_Gmatrix = train_Gmatrix.tocsr()
        test_Gmatrix = test_Gmatrix.tocsr()
        save_npz(train_path, train_Gmatrix)
        save_npz(test_path, test_Gmatrix)
        return train_Gmatrix, test_Gmatrix


if __name__ == '__main__':
    data = Data(dir_path="/home/wzy/LightGCN/data/diginetica")
