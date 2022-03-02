import torch
from engine import Engine
from utils import use_cuda

class EE(torch.nn.Module):
    def __init__(self, config):
        super(EE, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.bias_user = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.bias_item = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices, global_mean=0):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        user_bias = self.bias_user[user_indices]
        item_bias = self.bias_item[item_indices]
        # difference = user_embedding - item_embedding
        distance = (user_embedding - item_embedding).pow(2).sum(1).sqrt()
        rating = global_mean + user_bias + item_bias - distance  # torch.linalg.norm(torch.mul(difference, difference))
        rating = self.logistic(rating)
        return rating

    def init_weight(self):
        pass


class EEEngine(Engine):
    """Engine for training & evaluating Euclidean Embedding (EE) model"""
    def __init__(self, config):
        self.model = EE(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(EEEngine, self).__init__(config)