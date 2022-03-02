import torch
from engine import Engine
from utils import use_cuda

class MF(torch.nn.Module):
    def __init__(self, config):
        super(MF, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        rating = self.logistic(element_product.sum(1))
        # rating = self.logistic(torch.inner(user_embedding, item_embedding))
        return rating

    def init_weight(self):
        pass


class MFEngine(Engine):
    """Engine for training & evaluating MF model"""
    def __init__(self, config):
        self.model = MF(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(MFEngine, self).__init__(config)