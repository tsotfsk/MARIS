import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="Beauty")

args = parser.parse_args()


dataset = args.dataset
loadpath = dataset
num_samples = 1000
with open(os.path.join(loadpath, f'{dataset}_info.pkl'), 'rb') as f:
    info = pickle.load(f)
n_items = info['item_num']
set_items = set(range(n_items))


def candidate_sample(seq, target_id):
    sample = []
    cand_set = list(set_items - set([his - 1 for his in seq]) - {target_id - 1})
    items = np.random.choice(cand_set, num_samples, replace=False)
    sample.extend((items + 1).tolist())
    return sample


def load_data(phase):
    user_ids = []
    user_seqs = []
    target_ids = []
    with open(os.path.join(loadpath, 'seqdata', f'{dataset}_{phase}.csv'), 'r') as f:
        for line in f:
            user_id, seq, target_id = line.split(',')
            seq = list(map(int, seq.split('|')))
            user_ids.append(int(user_id))
            user_seqs.append(seq)
            target_ids.append(int(target_id))
    return user_seqs, target_ids


for phase in ['valid', 'test']:
    samples = []
    user_seqs, target_ids = load_data(phase)
    for seq, target_id in tqdm(zip(user_seqs, target_ids)):
        samples.append(candidate_sample(seq, target_id))
    with open(os.path.join(loadpath, 'seqdata', f'{dataset}_{phase}_samples.pkl'), 'wb') as f:
        pickle.dump(samples, f)
