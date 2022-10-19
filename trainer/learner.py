import os
from random import choice

import numpy as np
import torch
import glob
import torch.nn as nn
import tqdm
from sklearn.model_selection import ParameterGrid
from torch.utils.tensorboard import SummaryWriter
from utils import Logger

from .evaluator import Evaluator


def dict2str(result_dict):
    r""" convert result dict to str

    Args:
        result_dict (dict): result dict

    Returns:
        str: result str
    """

    result_str = ''
    for metric, value in result_dict.items():
        result_str += str(metric + ': ' + str(value)) + ' '
    return result_str


class BPRLoss(nn.Module):
    """ BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class BaseLearner(object):

    def __init__(self, config, model_cls, loader):
        self.loader = loader
        self.config = config

        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)
        self.model = model_cls(**config.model_param)
        # for name, parameter in self.model.named_parameters():
        #     print(name, parameter.requires_grad)

        self.model.cuda()

        self.epochs = config.epochs
        self.early_stop_rounds = config.early_stop
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

        self.cur_epoch = 0
        self.best_epoch = 0
        self.best_result = -np.inf

        # if hasattr(nn, config.loss):
        #     self.loss_func = getattr(nn, config.loss)(reduction=config.reduction)
        # elif config.loss == 'BPRLoss':
        #     self.loss_func = BPRLoss()

        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(),
                                                                lr=self.learning_rate,
                                                                weight_decay=self.weight_decay)

        self.save_floder = os.path.join(config.savepath, config.dataset, config.model)
        self.version = self.gen_version()
        self.logpath = os.path.join(self.save_floder, f"version_{self.version}")
        self.writer = SummaryWriter(self.logpath)
        self.logger = Logger(os.path.join(self.logpath, 'run.log'))

        self.evaluator = Evaluator(metrics=config.metrics, topk=config.topk,
                                   case_study=config.case_study,
                                   savepath=self.logpath)
        self.main_metric = config.main_metric
        self.param_space = config.grid_search

    def train_one_epoch(self):
        self.model.train()
        total_train_loss = 0.0
        with tqdm.tqdm(self.loader.train_dataloader(), ncols=100) as pbar:
            for step, input in enumerate(pbar):
                input = input.cuda()
                loss = self.model.calculate_loss(input)
                total_train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(
                    f'train loss: {total_train_loss / (step + 1):.8f}')
        train_loss = total_train_loss / len(pbar)
        self.writer.add_scalar(
            'train/train_loss', train_loss, global_step=self.cur_epoch)
        return train_loss

    def train(self):
        while self.cur_epoch < self.epochs:
            self.train_one_epoch()
            metrics = self.valid()

            self.logger.info(f'epoch: {self.cur_epoch} valid metrics: {dict2str(metrics)}')

            if metrics[self.main_metric] > self.best_result:
                self.best_result = metrics[self.main_metric]
                self.best_epoch = self.cur_epoch
                self.save()

            if self.cur_epoch - self.best_epoch > self.early_stop_rounds:
                self.logger.info("early stop...")
                break
            self.cur_epoch += 1

        self.writer.add_hparams(self.config.to_parm_dict(), {'test/best_result': self.best_result})

    def valid(self):
        metrics = self.eval_one_epoch(loader=self.loader.valid_dataloader())

        for key, value in metrics.items():
            self.writer.add_scalar(
                f'valid/{key}', value, global_step=self.cur_epoch)
        return metrics

    @torch.no_grad()
    def case_study(self):
        setattr(self.model, 'result', [])
        self.model.eval()
        choices = []
        pos_idxs, pos_lens = [], []
        for input in self.loader.test_dataloader():
            input = input.cuda()
            scores = self.model.predict(input)
            c = self.model.idx.gather(dim=-1, index=input.target_ids.view(-1,1)).cpu().numpy()
            choices.append(c)
            pos_idx, pos_len = self.evaluator.collect(scores, input)
            pos_idxs.append(pos_idx)
            pos_lens.append(pos_len)
        pos_idx = np.concatenate(pos_idxs, axis=0).astype(bool)
        pos_len = np.concatenate(pos_lens, axis=0)
        # get metrics
        result_matrix = self.evaluator._calculate_metrics(pos_idx, pos_len)
        result = np.stack(result_matrix, axis=0)[0, :, 0]
        result = result.nonzero()[0]
        choices = np.concatenate(choices).flatten()
        # choices = choices[result]
        np.save(f'{self.config.dataset}.npy', choices)

    def test(self):
        self.load()
        metrics = self.eval_one_epoch(loader=self.loader.test_dataloader())
        for key, value in metrics.items():
            self.writer.add_scalar(
                f'test/{key}', value, global_step=self.cur_epoch)

        self.logger.info(f'best epoch : {self.best_epoch}, test metrics: {dict2str(metrics)}')
        return metrics

    @torch.no_grad()
    def eval_one_epoch(self, loader):
        self.model.eval()
        pos_idxs, pos_lens = [], []
        for input in loader:
            input = input.cuda()
            scores = self.model.predict(input)
            pos_idx, pos_len = self.evaluator.collect(scores, input)
            pos_idxs.append(pos_idx)
            pos_lens.append(pos_len)
        result = self.evaluator.evaluate(pos_idxs, pos_lens)
        return result

    def save(self):
        filename = glob.glob(os.path.join(self.save_floder, f"version_{self.version}", '*.pth'))
        if filename:
            os.remove(filename[0])
        filename = os.path.join(self.save_floder, f"version_{self.version}", f"epoch={self.cur_epoch}.pth")
        state = {
            'best_epoch': self.best_epoch,
            'cur_epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self):
        filename = glob.glob(os.path.join(self.save_floder, f"version_{self.version}", '*.pth'))[0]
        state = torch.load(filename)
        self.cur_epoch = state['cur_epoch']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['state_dict'])

    def gen_version(self):
        dirs = glob.glob(os.path.join(self.save_floder, '*'))
        if not dirs:
            return 0
        if self.config.version < 0:
            version = max([int(x.split(os.sep)[-1].split('_')[-1])
                           for x in dirs]) + 1
        else:
            version = self.config.version
        return version


class RACELearner(object):
    def __init__(self, config, model_cls, loader):
        self.loader = loader
        self.config = config

        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.device)
        self.model = model_cls(**config.model_param)
        for name, parameter in self.model.named_parameters():
            print(name, parameter.requires_grad)

        self.model.cuda()

        self.epochs = config.epochs
        self.early_stop_rounds = config.early_stop
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay

        self.cur_epoch = 0
        self.best_epoch = 0
        self.best_result = -np.inf

        # if hasattr(nn, config.loss):
        #     self.loss_func = getattr(nn, config.loss)(reduction=config.reduction)
        # elif config.loss == 'BPRLoss':
        #     self.loss_func = BPRLoss()

        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(),
                                                                lr=self.learning_rate,
                                                                weight_decay=self.weight_decay)

        self.save_floder = os.path.join(config.savepath, config.dataset, config.model)
        self.version = self.gen_version()
        self.writer = SummaryWriter(os.path.join(self.save_floder, f"version_{self.version}"))

        self.evaluator = Evaluator(metrics=config.metrics, topk=config.topk,
                                   case_study=config.case_study,
                                   savepath=os.path.join(self.save_floder, f"version_{self.version}"))
        self.main_metric = config.main_metric
        self.param_space = config.grid_search

    def train_one_epoch(self):
        self.model.train()
        total_train_loss = 0.0
        with tqdm.tqdm(self.loader.train_dataloader(), ncols=100) as pbar:
            for step, input in enumerate(pbar):
                input = input.cuda()
                loss = self.model.calculate_loss(input)
                total_train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.set_description(
                    f'train loss: {total_train_loss / (step + 1):.8f}')
        train_loss = total_train_loss / len(pbar)
        self.writer.add_scalar(
            'train/train_loss', train_loss, global_step=self.cur_epoch)
        return train_loss

    def train(self):
        while self.cur_epoch < self.epochs:
            self.train_one_epoch()
            metrics = self.valid()

            print(f'epoch: {self.cur_epoch} valid metrics: {dict2str(metrics)}')

            if metrics[self.main_metric] > self.best_result:
                self.best_result = metrics[self.main_metric]
                self.best_epoch = self.cur_epoch
                self.save()

            if self.cur_epoch - self.best_epoch > self.early_stop_rounds:
                print("early stop...")
                break
            self.cur_epoch += 1

        self.writer.add_hparams(self.config.to_parm_dict(), {'test/best_result': self.best_result})

    def valid(self):
        metrics = self.eval_one_epoch(loader=self.loader.valid_dataloader())

        for key, value in metrics.items():
            self.writer.add_scalar(
                f'valid/{key}', value, global_step=self.cur_epoch)
        return metrics

    def test(self):
        self.load()
        metrics = self.eval_one_epoch(loader=self.loader.test_dataloader())
        for key, value in metrics.items():
            self.writer.add_scalar(
                f'test/{key}', value, global_step=self.cur_epoch)

        print(f'best epoch : {self.best_epoch}, test metrics: {dict2str(metrics)}')
        return metrics

    @torch.no_grad()
    def eval_one_epoch(self, loader):
        self.model.eval()
        pos_idxs, pos_lens = [], []
        for input in loader:
            input = input.cuda()
            scores = self.model.predict(input)
            pos_idx, pos_len = self.evaluator.collect(scores, input)
            pos_idxs.append(pos_idx)
            pos_lens.append(pos_len)
        result = self.evaluator.evaluate(pos_idxs, pos_lens)
        return result

    def save(self):
        filename = glob.glob(os.path.join(self.save_floder, f"version_{self.version}", '*.pth'))
        if filename:
            os.remove(filename[0])
        filename = os.path.join(self.save_floder, f"version_{self.version}", f"epoch={self.cur_epoch}.pth")
        state = {
            'best_epoch': self.best_epoch,
            'cur_epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def load(self):
        filename = glob.glob(os.path.join(self.save_floder, f"version_{self.version}", '*.pth'))[0]
        state = torch.load(filename)
        self.cur_epoch = state['cur_epoch']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['state_dict'])

    def gen_version(self):
        dirs = glob.glob(os.path.join(self.save_floder, '*'))
        if not dirs:
            return 0
        if self.config.version < 0:
            version = max([int(x.split(os.sep)[-1].split('_')[-1])
                           for x in dirs]) + 1
        else:
            version = self.config.version
        return version
