import argparse
import os
import random

import numpy as np
import torch

import models
from config import Config
from trainer import BaseLearner, SequentialDataLoader, learner_dict

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="Beauty")
parser.add_argument('--model', type=str, default="Attention")
parser.add_argument('--candidate_sample', action="store_true")
parser.add_argument('--case_study', action="store_true")
parser.add_argument('--method', type=str, default="")

parser.add_argument('--commit', type=str, default='')
parser.add_argument('--datapath', type=str, default='./dataset')
parser.add_argument('--confpath', type=str, default="./config")
parser.add_argument('--savepath', type=str, default="./saved")

args = parser.parse_args()

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    args = Config(args)
    config = args.easy_use()
    seed_everything(2048)

    loader = SequentialDataLoader(dataset=config.dataset, root=config.datapath,
                                  train_batch_size=config.train_batch_size,
                                  eval_batch_size=config.eval_batch_size,
                                  num_workers=0,
                                  negative_sample=config.negative_sample,
                                  candidate_sample=config.candidate_sample)

    model_cls = getattr(models, config.model)
    if config.model in learner_dict:
        learner = learner_dict[config.model](config, model_cls, loader)
    else:
        learner = BaseLearner(config, model_cls, loader)
    args.tab_printer(learner.logger)
    learner.train()
    learner.test()
