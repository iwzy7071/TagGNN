import numpy as np
from scipy.sparse import csr_matrix
from torch.utils.data.dataset import Dataset
import torch


class RSTrainDataset(Dataset):
    def __init__(self, Gmatrix: csr_matrix):
        self.num_user, self.num_item = Gmatrix.shape
        pos_items = []
        for uid in range(self.num_user):
            pos_item = Gmatrix.getrow(uid).indices.tolist()
            pos_items.append(pos_item)
        self.pos_items = pos_items

    def __len__(self):
        return self.num_user

    def __getitem__(self, uid):
        user_pos_items = self.pos_items[uid]
        pos_item_index = np.random.randint(0, len(user_pos_items))
        pos_item = user_pos_items[pos_item_index]
        while True:
            neg_item = np.random.randint(0, self.num_item)
            if neg_item not in user_pos_items:
                break
        return [uid, pos_item, neg_item]


class RSTestDataset(Dataset):
    def __init__(self, train_Gmatrix: csr_matrix, test_Gmatrix: csr_matrix):
        self.num_user, self.num_item = test_Gmatrix.shape
        true_items, test_items = [], []
        for uid in range(self.num_user):
            true_item = train_Gmatrix.getrow(uid).indices.tolist()
            true_items.append(true_item)
            test_item = test_Gmatrix.getrow(uid).indices.tolist()
            test_items.append(test_item)

        self.true_items = true_items
        self.test_items = test_items

    def __len__(self):
        return self.num_user

    def __getitem__(self, uid):
        true_items = self.true_items[uid]
        test_items = self.test_items[uid]
        return [uid, true_items, test_items]


def collate_fn(batch_item):
    uids, train_items, test_items = [], [], []
    for item in batch_item:
        uid, train_item, test_item = item[0], item[1], item[2]
        if len(test_item) == 0: continue
        uids.append(uid)
        train_items.append(train_item)
        test_items.append(test_item)
    return uids, train_items, test_items
