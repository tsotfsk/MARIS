import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from models.abstract_model import SequentialModel
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_


class GRU4Rec(SequentialModel):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        
        # load parameters info
        self.embedding_size = self.model_param.embedding_size
        self.hidden_size = self.model_param.hidden_size
        self.num_layers = self.model_param.num_layers

        self.bsc_embedding = nn.Embedding(
            self.item_num + 1, self.embedding_size, padding_idx=0)

        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        if isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
        if isinstance(module, nn.Linear):
            normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def get_user_embedding(self, item_seq_id, item_seq_len):
        seqs_embed = self.bsc_embedding(item_seq_id)
        ht, _ = self.gru_layers(seqs_embed)

        out = self.gather_indexes(ht, item_seq_len - 1)
        result = self.dense(out)
        return result

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def get_item_embedding(self):
        result = self.bsc_embedding.weight
        return result

    def forward(self, item_seq_id, item_seq_len):
        seqs_embed = self.bsc_embedding(item_seq_id)
        return self.get_user_embedding(seqs_embed, item_seq_len)

    def calculate_loss(self, input):
        item_seq_id = input.item_seq_id
        item_seq_len = input.item_seq_len
        pos_item_id = input.pos_item_id
        neg_item_id = input.neg_item_id

        user_embed = self.get_user_embedding(item_seq_id, item_seq_len)  # [B, E]
        neg_embed = self.bsc_embedding(neg_item_id)  # [B, E]
        pos_embed = self.bsc_embedding(pos_item_id)  # [B, E]
        pos_score = torch.sum(user_embed * pos_embed, dim=-1)  # [B]
        neg_score = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=-1)  # [B]
        loss = self.loss_fct(pos_score, neg_score)
        return loss

    def predict(self, input):
        item_seq_id = input.item_seq_id
        item_seq_len = input.item_seq_len

        user_embed = self.get_user_embedding(item_seq_id, item_seq_len)
        item_embed = self.get_item_embedding()
        scores = user_embed @ item_embed.t()
        return scores
