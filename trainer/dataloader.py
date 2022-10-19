import os
import pickle
from functools import partial
from turtle import left

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SequentialInput(object):

    def __init__(self, **kwgs):
        for key, value in kwgs.items():
            value = torch.LongTensor(value)
            setattr(self, key, value)

    def cuda(self):
        for key in self.__dict__:
            setattr(self, key, getattr(self, key).cuda())
        return self

    def sparse_interaction(self):
        user_ids = self.user_seqs.flatten()
        item_ids = self.user_ids.repeat_interleave(self.user_seqs.size(1)).flatten()
        indices = torch.cat((user_ids, item_ids), dim=1)
        return torch.sparse_coo_tensor(indices=indices, values=torch.ones_like(user_ids),
                                       device=user_ids.device)

    @property
    def seq_lens(self):
        return (self.user_seqs > 0).sum(dim=1)


class SequentialDataLoader(object):

    def __init__(self, dataset='Beauty', root='.', train_batch_size=128, eval_batch_size=1024,
                 num_workers=8, negative_sample=-1, candidate_sample=False):
        self.dataset = dataset
        self.root = root
        self.loadpath = os.path.join(self.root, self.dataset)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.negative_sample = negative_sample
        self.num_workers = num_workers
        self.candidate_sample = candidate_sample

        self._build_dataset()

    def _build_dataset(self):
        with open(os.path.join(self.loadpath, f'{self.dataset}_info.pkl'), 'rb') as f:
            self.info = pickle.load(f)
        self.n_items = self.info['item_num']

        self.train_data = SequentialDataset(dataset=self.dataset, root=self.root,
                                            phase='train', candidate_sample=False)
        self.valid_data = SequentialDataset(dataset=self.dataset, root=self.root,
                                            phase='valid', candidate_sample=self.candidate_sample)
        self.test_data = SequentialDataset(dataset=self.dataset, root=self.root,
                                           phase='test', candidate_sample=self.candidate_sample)

    def _collate_fn(self, data, negative_sample=-1):
        feed_dict = {
            'user_ids': [tup[0] for tup in data],
            'user_seqs': [tup[1] for tup in data],
            'target_ids': [tup[2] for tup in data],
        }

        def _negative_sample(target_id, k=1):
            neg_items = []
            for _ in range(k):
                neg_item = np.random.choice(self.n_items, 1)[0] + 1  # XXX padding at 0
                while neg_item == target_id:
                    neg_item = np.random.choice(self.n_items, 1)[0] + 1
                neg_items.append(neg_item)

            return neg_items

        if negative_sample > 0:
            neg_ids = []
            for target_id in feed_dict['target_ids']:
                neg_ids.append(_negative_sample(target_id, k=negative_sample))
            feed_dict['neg_ids'] = neg_ids
        elif self.candidate_sample > 0 and len(data[1]) > 3:
            feed_dict['neg_ids'] = [tup[3] for tup in data]

        return SequentialInput(**feed_dict)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=partial(self._collate_fn, negative_sample=self.negative_sample))

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=partial(self._collate_fn, negative_sample=-1))

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=partial(self._collate_fn, negative_sample=-1))


class SequentialDataset(Dataset):

    def __init__(self, dataset='Beauty', root='.', phase='train', candidate_sample=False) -> None:
        self.loadpath = os.path.join(root, dataset)
        self.dataset = dataset
        self.phase = phase
        self.candidate_sample = candidate_sample

        self._load_data()
        if self.phase in ['valid', 'test'] and self.candidate_sample > 0:
            self._load_candidate_samples()

    def _load_candidate_samples(self):
        savepath = os.path.join(self.loadpath, 'seqdata', f'{self.dataset}_{self.phase}_samples.pkl')
        if os.path.exists(savepath):
            with open(savepath, 'rb') as f:
                self.neg_ids = pickle.load(f)

    def _candidate_sample(self, seq, target_id):
        sample = []
        cand_set = list(self.set_items - set([his - 1 for his in seq]) - {target_id - 1})
        items = np.random.choice(cand_set, self.candidate_sample, replace=False)
        sample.extend((items + 1).tolist())
        return sample

    def _load_data(self):
        self.user_ids = []
        self.user_seqs = []
        self.target_ids = []
        with open(os.path.join(self.loadpath, 'seqdata', f'{self.dataset}_{self.phase}.csv'), 'r') as f:
            # for line in tqdm(f):
            for line in f:
                user_id, seq, target_id = line.split(',')
                seq = list(map(int, seq.split('|')))
                self.user_ids.append(int(user_id))
                self.user_seqs.append(seq)
                self.target_ids.append(int(target_id))

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        if self.phase in ['valid', 'test'] and self.candidate_sample > 0:
            return self.user_ids[idx], self.user_seqs[idx], self.target_ids[idx], self.neg_ids[idx]
        else:
            return self.user_ids[idx], self.user_seqs[idx], self.target_ids[idx]


if __name__ == "__main__":
    loader = SequentialDataLoader('Beauty', '../dataset', negative_sample=10)

    for seq_input in tqdm(loader.train_dataloader()):
        seq_input = seq_input.cuda()
        print(seq_input.__dict__)
        break

    for seq_input in tqdm(loader.valid_dataloader()):
        seq_input = seq_input.cuda()
        print(seq_input.__dict__)
        break

    for seq_input in tqdm(loader.test_dataloader()):
        seq_input = seq_input.cuda()
        print(seq_input.__dict__)
        break
