import torch
import numpy as np
from utils import evalulate_one_batch
from torch import optim
import logging
from torch.utils.data.dataloader import DataLoader

from model.BaseModel import BaseModel
from model.ModelFactory import ModelFactory
from data.Data import Data
from data.Dataset import RSTrainDataset, RSTestDataset, collate_fn


def Train(config: dict, model: BaseModel, dataloader: DataLoader, optimizer: optim.Optimizer):
    model = model.train()
    weight_decay = config["weight_decay"]
    average_loss = 0.0

    for batch_users, batch_pos, batch_neg in dataloader:
        batch_users = batch_users.cuda()
        batch_pos, batch_neg = batch_pos.cuda(), batch_neg.cuda()
        mf_loss, reg_loss = model.calculate_loss(batch_users, batch_pos, batch_neg)
        reg_loss = reg_loss * weight_decay
        loss = mf_loss + reg_loss
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
    config = dict(epoch=500, topks=[20], name="TagGCN", dataset="/home/wzy/LightGCN/data/amazon_toy", lr=5e-3,
                  train_batch_size=4096, n_iter=2, num_layers=3, dropout=0, ncaps=4, min_inter=10,
                  test_batch_size=1024, weight_decay=0.1, nfeat=64, split_ratios=[0.8, 0.2])
    logging.basicConfig(level=logging.INFO, filename=f'{config["name"]}.log', filemode='w')
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    logging.info(config)

    data = Data(dir_path=config["dataset"], split_ratios=config['split_ratios'], min_inter=config['min_inter'])
    train_dataset = RSTrainDataset(data.train_Gmatrix)
    test_dataset = RSTestDataset(data.train_Gmatrix, data.test_Gmatrix)
    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"], num_workers=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["test_batch_size"], num_workers=8,
                                 collate_fn=collate_fn)

    model = ModelFactory.get_model(config["name"], config, data).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config["epoch"]):
        loss = Train(config, model, train_dataloader, optimizer)
        if epoch % 5 == 0:
            results = Test(config, model, test_dataloader)
            pred, recall, ndcg = results["precision"], results["recall"], results["ndcg"]
            logging.info(f"epoch:{epoch} pred:{pred} recall:{recall} ndcg:{ndcg} loss:{loss}")
        else:
            logging.info(f"epoch:{epoch} loss:{loss}")
