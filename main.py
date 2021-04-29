import torch
import numpy as np
from utils import evalulate_one_batch
from torch import optim
import logging
from torch.utils.data.dataloader import DataLoader
from os.path import join

from model.BaseModel import BaseModel
from model.ModelFactory import ModelFactory
from data.Data import Data
from data.Dataset import RSTrainDataset, RSTestDataset, collate_fn


def Train(model: BaseModel, dataloader: DataLoader, optimizer: optim.Optimizer):
    model = model.train()
    average_loss = 0.0

    for batch_users, batch_pos, batch_neg in dataloader:
        batch_users = batch_users.cuda()
        batch_pos, batch_neg = batch_pos.cuda(), batch_neg.cuda()
        loss = model.calculate_loss(batch_users, batch_pos, batch_neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_loss += loss.cpu().item()
    average_loss = average_loss / len(dataloader)
    return average_loss


@torch.no_grad()
def Test(config: dict, model: BaseModel, dataloader: DataLoader):
    model = model.eval()
    topk = config["topks"]
    results = {'precision': np.zeros(len(topk)), 'recall': np.zeros(len(topk)), 'ndcg': np.zeros(len(topk))}
    rating_list, batch_true_list, pre_results = [], [], []
    test_dsize = 0
    for batch_users, batch_train_items, batch_test_items in dataloader:
        batch_users = torch.Tensor(batch_users).long()
        batch_users = batch_users.cuda()
        test_dsize += batch_users.numel()
        rating = model.get_user_ratings(batch_users)
        exclude_index = []
        exclude_items = []
        for train_uid, train_items in enumerate(batch_train_items):
            exclude_index.extend([train_uid] * len(train_items))
            exclude_items.extend(train_items)
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = torch.topk(rating, k=max(topk))
        rating.cpu().numpy()
        del rating
        pre_result = evalulate_one_batch([rating_K.cpu(), batch_test_items], topk)
        pre_results.append(pre_result)

    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']

    results['recall'] = results['recall'] / test_dsize
    results['precision'] = results['precision'] / test_dsize
    results['ndcg'] = results['ndcg'] / test_dsize
    return results


if __name__ == '__main__':
    # 配置基本的环境
    config = dict(epoch=500, topks=[20], name="DGCFTag", dataset="amazon_toy", lr=5e-3,
                  train_batch_size=1024, test_batch_size=1024)
    dataset_path = join("/home/wzy/LightGCN/data", config["dataset"])
    log_path = join("/home/wzy/LightGCN/log", f"{config['name']}_{config['dataset']}_128.log")
    save_model_path = join("/home/wzy/LightGCN/save_pt", f"{config['name']}_{config['dataset']}_32.pt")

    # 生成日志信息
    logging.basicConfig(level=logging.INFO, filename=log_path, filemode='w')
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logging.info(config)

    # 处理数据集
    data = Data(user_column_name="user_id:token", item_column_name="item_id:token",
                tag_column_name="genre:token_seq", dir_path=dataset_path,
                split_ratios=[0.8, 0.2], item_min_inter=5, user_min_inter=5)

    train_dataset = RSTrainDataset(data.train_Gmatrix)
    test_dataset = RSTestDataset(data.train_Gmatrix, data.test_Gmatrix)
    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["test_batch_size"], num_workers=8,
                                 collate_fn=collate_fn)

    # 创建模型
    model = ModelFactory.get_model(config["name"], data).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    best_recall = 0
    for epoch in range(config["epoch"]):
        loss = Train(model, train_dataloader, optimizer)
        if epoch % 5 == 0:
            results = Test(config, model, test_dataloader)
            pred, recall, ndcg = results["precision"], results["recall"], results["ndcg"]
            if recall > best_recall:
                best_recall = recall
                torch.save(model.state_dict(), save_model_path)
            logging.info(f"epoch:{epoch} pred:{pred} recall:{recall} ndcg:{ndcg} loss:{loss}")
        else:
            logging.info(f"epoch:{epoch} loss:{loss}")
