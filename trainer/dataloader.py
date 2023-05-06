import os
import sys
import pickle
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from utils.constants import *


class SequentialInput(object):

    def __init__(self, **kwgs):
        for key, value in kwgs.items():
            setattr(self, key, value)

    def cuda(self):
        for key, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, key, getattr(self, key).cuda())
        return self

    @property
    def item_seq_len(self):
        return (getattr(self, ITEM_SEQ_ID) > 0).sum(dim=1)

    @property
    def batch_size(self):
        return getattr(self, ITEM_SEQ_ID).size(0)

    @property
    def item_seq_mask(self):
        return (getattr(self, ITEM_SEQ_ID) > 0).long()

    def expand_as_action(self, flatten=True):
        assert hasattr(self, ACTION_SEQ)
        repeat_num = getattr(self, ACTION_SEQ).size(1)
        kwgs = {}
        for key, val in self.__dict__.items():
            if isinstance(getattr(self, key), torch.Tensor):
                if key != ACTION_SEQ:
                    val = val.unsqueeze(1).repeat_interleave(repeat_num, dim=1)
                if flatten:
                    val = val.reshape(-1, *val.shape[2:])
            kwgs[key] = val
        return SequentialInput(**kwgs)    


class SequentialDataLoader(object):

    def __init__(self, config, logger):

        self.config = config
        self.logger = logger

        self.dataset = config.dataset
        self.train_batch_size = config.train_batch_size
        self.eval_batch_size = config.eval_batch_size
        self.sample_strategy = config.sample_strategy
        self.num_workers = config.num_workers

        # build_sample_strategy
        self.dynamic_sample_num = sample_strategy_dict[self.sample_strategy]["dynamic_sample_num"]
        self.static_sample_num = sample_strategy_dict[self.sample_strategy]["static_sample_num"]

        self._build_dataset()

    def _build_dataset(self):

        # Dynamic sampling strategy used during training
        self.train_data = SequentialDataset(
            self.config, self.logger, phase='train', static_sample_num=0)

        # Static sampling strategy used during evaluation
        self.valid_data = SequentialDataset(
            self.config, self.logger, phase='valid', static_sample_num=self.static_sample_num)
        self.test_data = SequentialDataset(
            self.config, self.logger, phase='test', static_sample_num=self.static_sample_num)

    def _collate_fn(self, batch, dynamic_sample_num=0):
        feed_dict = {}
        for sample in batch:
            for key, val in sample.items():
                if key not in feed_dict:
                    feed_dict[key] = []
                feed_dict[key].append(val)
        for key, val in feed_dict.items():
            feed_dict[key] = torch.LongTensor(val)

        def _negative_sample(pos_item, k=1):
            neg_items = []
            for _ in range(k):
                neg_item = np.random.choice(self.config.item_num, 1)[
                    0] + 1  # Add 1 to the count because 0 is the padding token.
                while neg_item == pos_item:
                    neg_item = np.random.choice(self.config.item_num, 1)[0] + 1
                neg_items.append(neg_item)

            return neg_items

        if dynamic_sample_num > 0:
            neg_item_id = []
            for pos_item_id in feed_dict[POS_ITEM_ID]:
                neg_item_id.append(_negative_sample(
                    pos_item_id, k=dynamic_sample_num))
            feed_dict[NEG_ITEM_ID] = torch.LongTensor(neg_item_id)

        return SequentialInput(**feed_dict)

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          collate_fn=partial(self._collate_fn, dynamic_sample_num=self.dynamic_sample_num))

    def valid_dataloader(self):
        return DataLoader(dataset=self.valid_data,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=partial(self._collate_fn, dynamic_sample_num=0))

    def test_dataloader(self):
        return DataLoader(dataset=self.test_data,
                          batch_size=self.eval_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          collate_fn=partial(self._collate_fn, dynamic_sample_num=0))


class SequentialDataset(Dataset):

    def __init__(self, config, logger, phase='train', static_sample_num=1000) -> None:
        self.config = config
        self.logger = logger

        self.dataset = config.dataset
        self.phase = phase
        self.static_sample_num = static_sample_num

        self.data_dict = self._load_data()
        if self.static_sample_num > 0:
            self._load_static_samples()
            self.logger.info(
                "load static samples: phase={phase}, static_sample_num={num}"
                .format(phase=self.phase, num=self.static_sample_num)
            )

    @property
    def feat_cols(self):
        return list(self.data_dict.keys())

    def _load_static_samples(self):
        sample_path = STATIC_SAMPLE_PATH.format(
            dataset=self.dataset, phase=self.phase)
        if os.path.exists(sample_path):
            with open(sample_path, 'rb') as f:
                self.data_dict[NEG_ITEM_ID] = pickle.load(f)

    def _load_data(self):
        data_dict = {
            SAMPLE_ID: [],
            USER_ID: [],
            ITEM_SEQ_ID: [],
            POS_ITEM_ID: []
        }
        with open(SEQ_DATA_PATH.format(dataset=self.dataset, phase=self.phase), 'r') as f:
            # for line in tqdm(f):
            for idx, line in enumerate(f):
                user_id, item_id_seq, pos_item_id = line.split(',')
                item_id_seq = list(map(int, item_id_seq.split('|')))
                data_dict[SAMPLE_ID].append(idx)
                data_dict[USER_ID].append(int(user_id))
                data_dict[ITEM_SEQ_ID].append(item_id_seq)
                data_dict[POS_ITEM_ID].append(int(pos_item_id))
        return data_dict

    def __len__(self):
        return len(self.data_dict[self.feat_cols[0]])

    def __getitem__(self, idx):
        sample = {}
        for feat in self.feat_cols:
            sample[feat] = self.data_dict[feat][idx]
        return sample


class ReplayBuffer(object):

    def __init__(self, capaticy, seq_max_len=5, keep_num=5, agent_num=3, init_policy="01") -> None:
        self.capaticy = capaticy
        self.keep_num = 5

        if init_policy == "01":
            zero_buffer = torch.zeros(
                self.capaticy, keep_num // 2, agent_num, seq_max_len, dtype=torch.long)
            one_buffer = torch.ones(
                self.capaticy, keep_num - keep_num // 2, agent_num, seq_max_len, dtype=torch.long)
            self.buffer = torch.cat([zero_buffer, one_buffer], dim=1)
        elif init_policy == "0":
            self.buffer = torch.zeros(
                self.capaticy, keep_num, agent_num, seq_max_len, dtype=torch.long)
        elif init_policy == "1":
            self.buffer = torch.ones(
                self.capaticy, keep_num, agent_num, seq_max_len, dtype=torch.long)
        self.buffer = self.buffer.cuda()

    def select(self, sample_id):
        return self.buffer[sample_id]

    def replace(self, sample_id, action_seq):
        self.buffer[sample_id] = action_seq


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
