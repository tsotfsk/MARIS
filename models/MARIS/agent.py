import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .selector import EpsilonGreedyActionSelector

from models.abstract_model import SequentialModel
from utils import MODAL_PATH_DICT


class ModalAgent(SequentialModel):

    def __init__(self, config, logger, modal='ent'):
        super(ModalAgent, self).__init__(config, logger)

        self.embedding_size = self.model_param.embedding_size
        self.hidden_size = self.model_param.hidden_size
        self.num_layers = self.model_param.num_layers
        self.freeze_embedding = self.model_param.freeze_embedding
        self.dropout = self.model_param.dropout
        self.modal = modal

        # load embedding
        # self.shift_embedding = nn.Embedding(
        #     self.item_num + 1, 128, padding_idx=0)
        self.modal_embedding = nn.Embedding(
            self.item_num + 1, 128, padding_idx=0)

        self.gru_layers = nn.GRU(
            input_size=self.embedding_size * 2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense = nn.Linear(self.hidden_size, self.embedding_size * 2)
        self.q_net = nn.Linear(2 * self.hidden_size, 2)
        self.dropout = nn.Dropout(self.dropout)

        self.action_selector = EpsilonGreedyActionSelector(self.model_param)

        self.apply(self._init_weights)

    def move_one_step(self, input, hidden, step, step_action):
        item_id = input.item_seq_id[:, step, None]  # [B, 1]
        shift_embed = self.shift_embedding(item_id)  # [B, 1, D]
        modal_embed = self.modal_embedding(item_id)  # [B, 1, D]
        item_embed = torch.cat([shift_embed, modal_embed * step_action.unsqueeze(-1).float()], dim=-1)  # [B, 1, 2D]
        item_embed = self.dropout(item_embed)
        _, hidden = self.gru_layers(item_embed, hidden.contiguous())  # [B, 1, D]
        return hidden

    def get_observation(self, hidden, item_id):
        shift_embed = self.shift_embedding(item_id)  # [B, 1, D]
        modal_embed = self.modal_embedding(item_id)  # [B, 1, D]

        item_embed = shift_embed + modal_embed  # [B, 1, D]
        observation = torch.cat([item_embed, hidden.transpose(0, 1)], dim=-1)  # [B, 1, 2D]
        return observation

    def get_item_embedding(self):
        modal_emebd = self.modal_embedding.weight
        modal_emebd = self.dropout(modal_emebd)
        return modal_emebd

    def select_action(self, input, hidden, step):
        item_id = input.item_seq_id[:, step, None]  # [B, 1]
        observation = self.get_observation(hidden, item_id)
        q_value = self.q_net(observation)  # [B, 1, 2]
        action = self.action_selector.select_action(q_value, input.t_env, test_mode=input.test_mode)
        return q_value, action

    def imitate_action(self, input, hidden, step):
        item_id = input.item_seq_id[:, step, None]  # [B, 1]
        observation = self.get_observation(hidden, item_id)
        q_value = self.q_net(observation)  # [B, 1, 2]
        return q_value

    @torch.no_grad()
    def load_embedding(self, normalize=True):
        for modal in [self.modal]:
            embed = np.load(
                MODAL_PATH_DICT[modal].format(dataset=self.dataset))
            embed = torch.from_numpy(embed)
            if normalize:
                embed = F.normalize(embed, p=2, dim=1)
            weight = self.modal_embedding.weight
            weight[1:].copy_(embed[:self.item_num])
            if self.freeze_embedding:
                weight.requires_grad = False
