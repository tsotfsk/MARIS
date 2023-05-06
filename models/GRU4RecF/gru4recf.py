import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.abstract_model import SequentialModel
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_
from utils import MODAL_PATH_DICT



class GRU4RecF(SequentialModel):
    def __init__(self, config, logger):
        super(GRU4RecF, self).__init__(config, logger)

        self.embedding_size = self.model_param.embedding_size
        self.hidden_size = self.model_param.hidden_size
        self.num_layers = self.model_param.num_layers
        self.freeze_embedding = self.model_param.freeze_embedding
        self.dropout = self.model_param.dropout


        # load embedding
        self.bsc_embedding = nn.Embedding(
            self.item_num + 1, self.embedding_size, padding_idx=0)
        
        self.img_embedding = nn.Embedding(
                self.item_num + 1, 128, padding_idx=0)
        self.txt_embedding = nn.Embedding(
                self.item_num + 1, 128, padding_idx=0)
        self.ent_embedding = nn.Embedding(
                self.item_num + 1, 128, padding_idx=0)
        self.dropout = nn.Dropout(self.dropout)

        self.bsc_gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.fea_gru = nn.GRU(
            input_size=128 * 3,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense = nn.Linear(self.hidden_size * 2, self.embedding_size)

        self.apply(self._init_weights)
        self.load_embedding()


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

    @torch.no_grad()
    def load_embedding(self, normalize=True):
        for modal in ['ent', 'img', 'txt']:
            embed = np.load(MODAL_PATH_DICT[modal].format(dataset=self.dataset))
            embed = torch.from_numpy(embed)
            if normalize:
                embed = F.normalize(embed, p=2, dim=1)
            weight = getattr(self, f"{modal}_embedding").weight
            weight[1:].copy_(embed[:self.item_num])
            if self.freeze_embedding:
                weight.requires_grad = False

    def get_user_embedding(self, item_seq_id, item_seq_len):
        bsc_embed = self.dropout(self.bsc_embedding(item_seq_id))
        img_embed = self.dropout(self.img_embedding(item_seq_id))
        txt_embed = self.dropout(self.txt_embedding(item_seq_id))
        ent_embed = self.dropout(self.ent_embedding(item_seq_id))

        bsc_ht, _ = self.bsc_gru(bsc_embed)
        fea_ht, _ = self.fea_gru(torch.cat((img_embed, txt_embed, ent_embed), dim=-1))
        ht = torch.cat((bsc_ht, fea_ht), dim=-1)
        out = self.gather_indexes(ht, item_seq_len - 1)
        result = self.dense(out)
        return result

    def get_item_embedding(self):
        return self.bsc_embedding.weight

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def forward(self, item_seq_id, item_seq_len):
        return self.get_user_embedding(item_seq_id, item_seq_len)

    def calculate_loss(self, input):
        item_seq_id = input.item_seq_id
        item_seq_len = input.item_seq_len
        pos_item_id = input.pos_item_id
        neg_item_id = input.neg_item_id

        user_embed = self.forward(item_seq_id, item_seq_len)

        test_item_emb = self.get_item_embedding()

        neg_embed = test_item_emb[neg_item_id]  # [B, E]
        pos_embed = test_item_emb[pos_item_id]

        pos_score = torch.sum(user_embed * pos_embed, dim=-1)  # [B]
        neg_score = torch.sum(user_embed.unsqueeze(1) * neg_embed, dim=-1)  # [B]
        loss = self.loss_fct(pos_score, neg_score)

        return loss

    def predict(self, input):
        item_seq_id = input.item_seq_id
        item_seq_len = input.item_seq_len

        user_embed = self.forward(item_seq_id, item_seq_len)
        item_embed = self.get_item_embedding()
        scores = user_embed @ item_embed.t()
        return scores
