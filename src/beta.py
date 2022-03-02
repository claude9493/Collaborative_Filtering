import torch
import numpy as np
from engine import Engine
from utils import use_cuda
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class Regularizer:
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class BR(torch.nn.Module):
    def __init__(self, config):
        super(BR, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.latent_dim = config['latent_dim']
        self.distance = config['distance']  # KL or JS

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim*2)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim*2)

        self.regularizer = Regularizer(1, 0.05, 1e9)
        self.embedding_range = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        torch.nn.init.uniform_(tensor=self.embedding_user.weight,
                         a=-self.embedding_range.item(),
                         b=self.embedding_range.item())
        torch.nn.init.uniform_(tensor=self.embedding_item.weight,
                         a=-self.embedding_range.item(),
                         b=self.embedding_range.item())

        self.bias_user = torch.nn.Parameter(torch.randn(self.num_users), requires_grad=True)
        self.bias_item = torch.nn.Parameter(torch.randn(self.num_items), requires_grad=True)
        self.affine_output = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()
    
    def forward(self, user_indices, item_indices, global_mean=0):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        user_bias = self.bias_user[user_indices]
        item_bias = self.bias_item[item_indices]

        alpha_user, beta_user = torch.chunk(self.regularizer(user_embedding).unsqueeze(-1), 2, dim=1)
        alpha_item, beta_item = torch.chunk(self.regularizer(item_embedding).unsqueeze(-1), 2, dim=1)

        dist_user = torch.distributions.beta.Beta(alpha_user, beta_user)
        dist_item = torch.distributions.beta.Beta(alpha_item, beta_item)

        # difference = user_embedding - item_embedding
        distance = self.affine_output(self.JS_divergence(dist_user, dist_item)).squeeze()
        rating = global_mean + user_bias + item_bias - distance  # torch.linalg.norm(torch.mul(difference, difference))
        rating = self.logistic(rating)
        return rating

    def init_weight(self):
        pass

    def JS_divergence(self, u_dist, m_dist):
      mean_dist = torch.distributions.beta.Beta(0.5 * (u_dist.concentration0 + m_dist.concentration0), 0.5 * (u_dist.concentration1 + m_dist.concentration1))
      # 0.5 * (u_dist + m_dist)
      kl_1 = torch.distributions.kl.kl_divergence(u_dist, mean_dist).squeeze()
      kl_2 = torch.distributions.kl.kl_divergence(m_dist, mean_dist).squeeze()
    #   return torch.norm(2.0/torch.pi * torch.atan((kl_1 + kl_2) * 0.5), p=1, dim=-1)
      return (kl_1 + kl_2) / 2


class BREngine(Engine):
    """Engine for training & evaluating Beta Recommendation (BR) model"""
    def __init__(self, config):
        self.model = BR(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.model.cuda()
        super(BREngine, self).__init__(config)
    
    def train_single_batch(self, users, items, ratings):
        loss = super().train_single_batch(users, items, ratings)
        # self.model.embedding_user = self.model.regularizer(self.model.embedding_user)
        # self.model.embedding_item = self.model.regularizer(self.model.embedding_item)
        return loss

    def train_an_epoch(self, train_loader, epoch_id):
        super().train_an_epoch(train_loader, epoch_id)
        # self.plot_kde(user=True, epoch_id=epoch_id)
        # self.plot_kde(user=False, epoch_id=epoch_id)
    
    def plot_kde(self, user=True, epoch_id=0):
        if user == True:
            alpha, beta = torch.chunk(self.model.regularizer(self.model.user_embedding).unsqueeze(-1), 2, dim=1)
            xx, yy, zz = self.kde2D(alpha, beta, 1.0)
            plt.pcolormesh(xx, yy, zz)
            plt.scatter(alpha, beta, s=2, facecolor='white')
            self._writer.add_figure('fig/user', plt.gcf(), epoch_id)
        else:
            alpha, beta = torch.chunk(self.model.regularizer(self.model.item_embedding).unsqueeze(-1), 2, dim=1)
            xx, yy, zz = self.kde2D(alpha, beta, 1.0)
            plt.pcolormesh(xx, yy, zz)
            plt.scatter(alpha, beta, s=2, facecolor='white')
            self._writer.add_figure('fig/item', plt.gcf(), epoch_id)

    def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
        """Build 2D kernel density estimate (KDE)."""

        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[x.min():x.max():xbins, 
                          y.min():y.max():ybins]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train  = np.vstack([y, x]).T

        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        z = np.exp(kde_skl.score_samples(xy_sample))
        return xx, yy, np.reshape(z, xx.shape)