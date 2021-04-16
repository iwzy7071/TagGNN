from os.path import join
import json
from scipy.sparse import load_npz
import pickle
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import numpy as np
from collections import Counter

# 处理数据
dir_path = "/home/wzy/LightGCN/data/amazon_book"
train_Gmatrix = join(dir_path, "train_Gmatrix.npz")
test_Gmatrix = join(dir_path, "test_Gmatrix.npz")
user_encoder = join(dir_path, "user_encoder.pkl")
item_encoder = join(dir_path, "item_encoder.pkl")
tag_encoder = join(dir_path, "tag_encoder.pkl")
tag_table = join(dir_path, "tag_tables.json")

train_Gmatrix = load_npz(train_Gmatrix)
test_Gmatrix = load_npz(test_Gmatrix)
tag_table = json.load(open(tag_table))
user_encoder = pickle.load(open(user_encoder, 'rb'))
item_encoder = pickle.load(open(item_encoder, 'rb'))
tag_encoder = pickle.load(open(tag_encoder, 'rb'))

train_Gmatrix: csr_matrix
test_Gmatrix: csr_matrix
tag_table: dict
user_encoder: LabelEncoder
item_encoder: LabelEncoder
tag_encoder: LabelEncoder

# 统计用户、物品、标签的数量
num_users = len(user_encoder.classes_.tolist())
num_items = len(item_encoder.classes_.tolist())
num_tags = len(tag_encoder.classes_.tolist())

counter = Counter()
tag_table_length = [len(value) for value in tag_table.values()]
tag_table_length = sorted(tag_table_length)
tag_table_length = tag_table_length[int(len(tag_table_length) * 0.1):]
cold_start_items = set(test_Gmatrix.indices.tolist()) - set(train_Gmatrix.indices.tolist())
useless_items = set(train_Gmatrix.indices.tolist()) - set(test_Gmatrix.indices.tolist())

print("用户数量", num_users)
print("物品数量", num_items)
print("标签数量", num_tags)
print("训练集的交互数量", train_Gmatrix.sum())
print("测试集的交互数量", test_Gmatrix.sum())
print("数据集稀疏程度", train_Gmatrix.sum() / num_users / num_items)
print("标签平均对应的item数量", np.array(tag_table_length).mean())
