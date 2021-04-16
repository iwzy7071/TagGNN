import torch
import numpy as np
from Data import Data
from utils import evalulate_one_batch, uniform_sample, train_shuffle, minibatch
from torch import optim
import logging
from tqdm import tqdm

from model.BaseModel import BaseModel
from model.ModelFactory import ModelFactory



def Train(config: dict, model: BaseModel, dataset: Data, optimizer: optim.Optimizer):
    model = model.train()
    batch_size, weight_decay = config["train_batch_size"], config["weight_decay"]
    train_pairs = uniform_sample(dataloader)

    users = torch.Tensor(train_pairs[:, 0]).long()
    pos_items = torch.Tensor(train_pairs[:, 1]).long()
    neg_items = torch.Tensor(train_pairs[:, 2]).long()

    users = users.to("cuda")
    pos_items = pos_items.to("cuda")
    neg_items = neg_items.to("cuda")

    users, pos_items, neg_items = train_shuffle(users, pos_items, neg_items)
    total_batch = len(users) // batch_size + 1
    average_loss = 0.0

    for batch_users, batch_pos, batch_neg in tqdm(minibatch(users, pos_items, neg_items, batch_size=batch_size)):
        mf_loss, reg_loss = model.calculate_loss(batch_users, batch_pos, batch_neg)
        reg_loss = reg_loss * weight_decay
        loss = mf_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        average_loss += loss.cpu().item()
    average_loss = average_loss / total_batch
    return average_loss


@torch.no_grad()
def Test(config: dict, model: BaseModel, dataloader: Data):
    model = model.eval()
    topk, batch_size = config["topks"], config["test_batch_size"]
    results = {'precision': np.zeros(len(topk)), 'recall': np.zeros(len(topk)), 'ndcg': np.zeros(len(topk))}
    test_users = dataloader.test_dict
    users_list = []
    rating_list = []
    batch_true_list = []
    pre_results = []
    for batch_users in minibatch(list(test_users.keys()), batch_size=batch_size):
        batch_train_items = dataloader.get_user_pos_items(batch_users)
        batch_true_items = [test_users[uid] for uid in batch_users]
        batch_users_gpu = torch.Tensor(batch_users).long()
        batch_users_gpu = batch_users_gpu.to("cuda")
        rating = model.get_user_ratings(batch_users_gpu)
        exclude_index = []
        exclude_items = []
        for train_item_idx, train_items in enumerate(batch_train_items):
            exclude_index.extend([train_item_idx] * len(train_items))
            exclude_items.extend(train_items)
        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = torch.topk(rating, k=max(topk))
        rating = rating.cpu().numpy()
        del rating
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        batch_true_list.append(batch_true_items)
    for x in zip(rating_list, batch_true_list):
        pre_results.append(evalulate_one_batch(x, topk))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(test_users))
    results['precision'] /= float(len(test_users))
    results['ndcg'] /= float(len(test_users))
    return results


if __name__ == '__main__':
    config = dict(epoch=500, topks=[20], name="DGCF", dataset="/home/wzy/LightGCN/data/amazon_book", lr=5e-3,
                  train_batch_size=4096, n_iter=2, num_layers=3, dropout=0, ncaps=4, min_inter=15,
                  test_batch_size=2048, weight_decay=0.1, nfeat=64, split_ratios=[0.8, 0.2])
    logging.basicConfig(level=logging.INFO, filename=f'{config["name"]}.log', filemode='w')
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    dataloader = Data(dir_path=config["dataset"], split_ratios=config['split_ratios'],
                      min_inter=config['min_inter'])
    model = ModelFactory.get_model(config["name"], config, dataloader).cuda()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    for epoch in range(config["epoch"]):
        loss = Train(config, model, dataloader, optimizer)
        if epoch % 5 == 0:
            results = Test(config, model, dataloader)
            pred, recall, ndcg = results["precision"], results["recall"], results["ndcg"]
            logging.info(f"epoch:{epoch} pred:{pred} recall:{recall} ndcg:{ndcg} loss:{loss}")
        else:
            logging.info(f"epoch:{epoch} loss:{loss}")
