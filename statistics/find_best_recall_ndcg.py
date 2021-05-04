from os.path import join

name = "DGCFTag_diginetica_256.log"
dir_path = "/home/wzy/LightGCN/log"
file_path = join(dir_path, name)

max_recall = 0
max_ndcg = 0
for line in open(file_path).readlines():
    if "pred" not in line:
        continue
    line = line.split(" ")
    _, recall, ndcg = line[1:4]
    recall = eval(recall.split(":")[-1])[0]
    ndcg = eval(ndcg.split(":")[-1])[0]
    max_ndcg, max_recall = max(max_ndcg, ndcg), max(max_recall, recall)

print(max_recall, max_ndcg)
