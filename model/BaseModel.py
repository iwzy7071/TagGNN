import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BaseModel, self).__init__()

    def get_user_ratings(self, user):
        raise NotImplementedError

    def calculate_loss(self, user, pos_item, neg_item):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
