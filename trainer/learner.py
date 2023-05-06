import os

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import SAVE_ROOT_PATH, dict2str, flatten_dict, filter_tracked_hparams

from .dataloader import ReplayBuffer
from .evaluator import Evaluator
from box import Box


class BaseLearner(object):

    def __init__(self, config, model, loader, logger):
        self.config = config
        self.model = model
        self.loader = loader
        self.logger = logger

        self.epochs = config.epochs
        self.early_stop_rounds = config.early_stop
        self.learning_rate = config.learning_rate
        self.weight_decay = config.weight_decay
        self.commit = config.commit

        self.cur_epoch = 0
        self.best_epoch = 0
        self.best_result = -np.inf

        self.optimizer = getattr(torch.optim, config.optimizer)(self.model.parameters(),
                                                                lr=self.learning_rate,
                                                                weight_decay=self.weight_decay)

        self.save_floder = SAVE_ROOT_PATH.format(
            dataset=config.dataset, model=config.model, commit=config.commit)
        self.writer = SummaryWriter(self.save_floder)
        self.evaluator = Evaluator(metrics=config.metrics, topk=config.topk)
        self.main_metric = config.main_metric

    def _train_one_epoch(self):
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
            self._train_one_epoch()
            metrics = self.valid()

            self.logger.info(
                f'epoch: {self.cur_epoch} valid metrics: {dict2str(metrics)}')

            if metrics[self.main_metric] > self.best_result:
                self.best_result = metrics[self.main_metric]
                self.best_epoch = self.cur_epoch
                self.save_cur_model()

            if self.cur_epoch - self.best_epoch > self.early_stop_rounds:
                self.logger.info("early stop...")
                break
            self.cur_epoch += 1
        tracked_hparams = filter_tracked_hparams(flatten_dict(self.config.to_dict()))
        self.writer.add_hparams(tracked_hparams, {
                                'test/best_result': self.best_result})

    def valid(self):
        metrics = self._eval_one_epoch(loader=self.loader.valid_dataloader())

        for key, value in metrics.items():
            self.writer.add_scalar(
                f'valid/{key}', value, global_step=self.cur_epoch)
        return metrics

    def test(self):
        self._load_best_model()
        metrics = self._eval_one_epoch(loader=self.loader.test_dataloader())
        for key, value in metrics.items():
            self.writer.add_scalar(
                f'test/{key}', value, global_step=self.cur_epoch)

        self.logger.info(
            f'best epoch : {self.best_epoch}, test metrics: {dict2str(metrics)}')
        return metrics

    @torch.no_grad()
    def _eval_one_epoch(self, loader):
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

    def save_cur_model(self):
        filename = os.path.join(self.save_floder, f"model.pth")
        state = {
            'best_epoch': self.best_epoch,
            'cur_epoch': self.cur_epoch,
            'state_dict': self.model.state_dict(),
        }
        torch.save(state, filename)

    def _load_best_model(self):
        filename = os.path.join(self.save_floder, f"model.pth")
        state = torch.load(filename)
        self.cur_epoch = state['cur_epoch']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['state_dict'])


class RLLearner(BaseLearner):

    def __init__(self, config, model, loader, logger):
        super().__init__(config, model, loader, logger)

        # self.target_model = copy.deepcopy(self.model)
        self.buffer = ReplayBuffer(len(loader.train_data))
        self.train_steps = 0
        self.sample_num = config.model_param.sample_num
        self.alpha = config.model_param.alpha

    def _train_one_epoch(self):
        # train steps
        total_train_loss = 0.0
        with tqdm.tqdm(self.loader.train_dataloader(), ncols=100) as pbar:
            for step, input in enumerate(pbar):
                self.model.train()
                input.action_seq = self.buffer.select(input.sample_id)
                input.test_mode = False
                input.t_env = self.train_steps

                input = input.cuda()

                exp_input = input.expand_as_action()

                rec_loss, td_loss = self.model.calculate_loss(exp_input)
                loss = rec_loss + 0.01 * td_loss

                total_train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.train_steps += 1

                pbar.set_description(
                    f'train loss: {total_train_loss / (step + 1):.8f}')
        train_loss = total_train_loss / len(pbar)
        self.writer.add_scalar(
            'train/train_loss', train_loss, global_step=self.cur_epoch)

        # sample steps
        with tqdm.tqdm(self.loader.train_dataloader(), ncols=100) as pbar:
            for step, input in enumerate(pbar):
                self.model.eval()

                input.test_mode = False
                input.cuda()
                input.t_env = self.train_steps

                sample_output = self.model.sample_episodes(input, self.sample_num)

                input.action_seq = self.buffer.select(input.sample_id)
                exp_input = input.expand_as_action()
                exp_input.cuda()

                buffer_output = self.model.imitate_episodes(exp_input)
                
                for key, val in sample_output.items():
                    sample_output[key] = val.reshape(input.batch_size, self.sample_num, *val.shape[1:])
                
                for key, val in buffer_output.items():
                    buffer_output[key] = val.reshape(input.batch_size, self.buffer.keep_num, *val.shape[1:])

                better_action = self.rank_episodes(sample_output, buffer_output)
                self.buffer.replace(input.sample_id, better_action)

        # update target model
        self.model.update_targets()
        return train_loss

    @torch.no_grad()
    def _eval_one_epoch(self, loader):
        self.model.eval()
        pos_idxs, pos_lens = [], []
        for input in loader:

            input.test_mode = True
            # input.action_seq = None
            input.t_env = self.train_steps  # XXX not used

            input = input.cuda()

            scores = self.model.predict(input)

            pos_idx, pos_len = self.evaluator.collect(scores, input)
            pos_idxs.append(pos_idx)
            pos_lens.append(pos_len)
        result = self.evaluator.evaluate(pos_idxs, pos_lens)
        return result

    def rank_episodes(self, sample_output, buffer_output):        
        total_action = torch.cat([sample_output.actions, buffer_output.actions], dim=1)
        
        max_reward_index = buffer_output.rewards.max(dim=1, keepdim=True)[1]
        gather_index = max_reward_index.unsqueeze(2).expand(-1, -1, buffer_output.enhts.size(2))
        max_enht = buffer_output.enhts.gather(1, gather_index).squeeze(1)  # [B, D]
        
        cos_sim = F.cosine_similarity(sample_output.enhts, max_enht[:, None, :], dim=2)  # [B, sample_num]
        
        rank_exploit = sample_output.rewards * cos_sim

        rank_explore = 1 - cos_sim

        rank_score = self.alpha * rank_explore + (1 - self.alpha) * rank_exploit

        rank_score = torch.cat([rank_score, buffer_output.rewards], dim=1)  # [B, sample_num + keep_num]
        
        top_idx = torch.topk(rank_score, self.buffer.keep_num, dim=1)[1]
        
        gather_index = top_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, self.buffer.keep_num, *total_action.size()[2:])
        
        better_action = total_action.gather(1, gather_index)
        return better_action

    # def save_q_net(self):
    #     torch.save(self.model.state_dict(), "{}/maris.th".format(self.save_floder))
    #     torch.save(self.optimizer.state_dict(), "{}/opt.th".format(self.save_floder))

    # def load_q_net(self):
    #     self.model.load_state_dict(torch.load("{}/maris.th".format(self.save_floder),
    #                                map_location=lambda storage, loc: storage))
    #     self.target_model.load_state_dict(torch.load("{}/maris.th".format(self.save_floder),
    #                                       map_location=lambda storage, loc: storage))
    #     self.optimizer.load_state_dict(torch.load("{}/opt.th".format(self.save_floder),
    #                                    map_location=lambda storage, loc: storage))
