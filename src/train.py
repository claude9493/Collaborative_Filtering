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
import argparse
from config import *

def load_data():
    # Load Data
    ml1m_dir = 'data/ml-1m/ratings.dat'
    ml1m_rating = pd.read_csv(ml1m_dir, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')
    # Reindex
    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    print('Range of userId is [{}, {}]'.format(ml1m_rating.userId.min(), ml1m_rating.userId.max()))
    print('Range of itemId is [{}, {}]'.format(ml1m_rating.itemId.min(), ml1m_rating.itemId.max()))
    # DataLoader for training
    sample_generator = SampleGenerator(ratings=ml1m_rating)
    evaluate_data = sample_generator.evaluate_data
    return sample_generator, evaluate_data

def main(args):
    sample_generator, evaluate_data = load_data()
    # Specify the exact model
    model = args.model
    print("The {} model is to be trained.".format(model))
    config = configs[model]
    engine = {
        'mf': MFEngine,
        'ee': EEEngine,
        'gmf': GMFEngine,
        'mlp': MLPEngine,
        'neumf': NeuMFEngine,
        'br': BREngine
    }[model](config)
    # config = gmf_config
    # engine = GMFEngine(config)
    # config = mlp_config
    # engine = MLPEngine(config)
    # config = neumf_config
    # engine = NeuMFEngine(config)
    pbar = tqdm(range(config['num_epoch']))
    for epoch in pbar:
        pbar.set_description('Epoch {}'.format(epoch))
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
    args = parser.parse_args()
    main(args)
