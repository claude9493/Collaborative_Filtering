import pandas as pd
import numpy as np
from mf import MFEngine
from ee import EEEngine
from gmf import GMFEngine
from mlp import MLPEngine
from neumf import NeuMFEngine
from beta import BREngine
from data import SampleGenerator
from tqdm import tqdm
import torch
import argparse
from config import *

def load_data():
    # Load Data
    ml1m_dir = 'data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    # ml100k_dir = 'data/ml-100k/ratings.csv'
    # ml100k_rating = pd.read_csv(ml00k_dir, sep=',', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    rating_data = ml1m_rating  # ml100k_rating
    # Reindex
    user_id = rating_data[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    rating_data = pd.merge(rating_data, user_id, on=['uid'], how='left')
    item_id = rating_data[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    rating_data = pd.merge(rating_data, item_id, on=['mid'], how='left')
    rating_data = rating_data[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(rating_data.userId.min(), rating_data.userId.max()))
    print('Range of itemId is [{}, {}]'.format(rating_data.itemId.min(), rating_data.itemId.max()))
    # DataLoader for training
    sample_generator = SampleGenerator(ratings=rating_data)
    evaluate_data = sample_generator.evaluate_data
    return sample_generator, evaluate_data

def main(args):
    sample_generator, evaluate_data = load_data()
    # Specify the exact model
    model = args.model
    print("The {} model is to be trained.".format(model))
    config = configs[model]
    if torch.cuda.is_available():
        config['use_cuda'] = True
        print("use_cuda is set to be True.")
    config['num_epoch'] = args.epoch
    engine = {
        'mf': MFEngine,
        'ee': EEEngine,
        'gmf': GMFEngine,
        'mlp': MLPEngine,
        'neumf': NeuMFEngine,
        'br': BREngine
    }[model](config)
    # pbar = tqdm(range(config['num_epoch']))
    # for epoch in pbar:
    for epoch in range(config['num_epoch']):
        # pbar.set_description('Epoch {}'.format(epoch))
        # print('Epoch {} starts !'.format(epoch))
        # print('-' * 80)
        train_loader = sample_generator.instance_a_train_loader(config['num_negative'], config['batch_size'])
        engine.train_an_epoch(train_loader, epoch_id=epoch)
        hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
        engine.save(config['alias'], epoch, hit_ratio, ndcg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    choices = ['mf', 'gmf', 'ee', 'mlp', 'neumf', 'br']
    parser.add_argument('--model',
                         type=str.lower,
                         choices=choices,
                         default='mf')
    parser.add_argument('--epoch',
                         type=int,
                         default=50)
    args = parser.parse_args()
    main(args)
