import numpy as np
from dataloader import DataLoader


def _get_label_(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def _recall_predict_(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def _ndcg_predict_(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def evalulate_one_batch(x, top_k):
    predict_items = x[0].numpy()
    true_items = x[1]
    r = _get_label_(true_items, predict_items)
    pre, recall, ndcg = [], [], []
    for k in top_k:
        ret = _recall_predict_(true_items, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(_ndcg_predict_(true_items, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def uniform_sample(dataloader: DataLoader):
    user_num = dataloader.train_data_size
    users = np.random.randint(0, dataloader.n_user, user_num)
    train_pos_items = dataloader.train_pos_items
    ans = []
    for index, uid in enumerate(users):
        user_pos_items = train_pos_items[uid]
        if len(user_pos_items) == 0: continue
        pos_item_index = np.random.randint(0, len(user_pos_items))
        pos_item = user_pos_items[pos_item_index]
        while True:
            neg_item = np.random.randint(0, dataloader.n_item)
            if neg_item not in train_pos_items:
                break
        ans.append([uid, pos_item, neg_item])
    return np.array(ans)


def train_shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    if require_indices:
        return result, shuffle_indices
    else:
        return result


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)
