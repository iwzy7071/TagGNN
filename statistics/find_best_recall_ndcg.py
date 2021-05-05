from os.path import join

name = "DGCFTag_diginetica_cold.log"
dir_path = "/home/wzy/LightGCN/log"
file_path = join(dir_path, name)

max_recall = 0
max_ndcg = 0
for line in open(file_path).readlines():
    if "pred" not in line:
        continue
    line = line.split(" ")
    try:
        recall, ndcg = ','.join(line[4:7]), ','.join(line[7:-1])
        recall, ndcg = eval(recall.split(":")[-1]), eval(ndcg.split(":")[-1])
        recall, ndcg = recall[-1], ndcg[-1]
        max_ndcg, max_recall = max(max_ndcg, ndcg), max(max_recall, recall)
    except:
        continue
print(max_recall, max_ndcg)
