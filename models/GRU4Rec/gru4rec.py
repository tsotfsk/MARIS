import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_


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


class GRU4Rec(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=128, num_layers=1,
                loadpath='', dataset='Beauty'):
        super(GRU4Rec, self).__init__()

        # load parameters info
        self.dataset = dataset
        self.loadpath = loadpath

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.n_items = self.load_info()

        # load embedding
        self.bsc_embedding = nn.Embedding(
            self.n_items + 1, self.embedding_size, padding_idx=0)

        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        self.loss_fct = BPRLoss()

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

    def load_info(self):
        pathname = os.path.join(self.loadpath, f'{self.dataset}_info.pkl')
        with open(pathname, 'rb') as f:
            info = pickle.load(f)
        return info['item_num']

    def gen_user_embedding(self, user_seqs, seq_lens):
        seqs_embed = self.bsc_embedding(user_seqs)
        ht, _ = self.gru_layers(seqs_embed)

        out = self.gather_indexes(ht, seq_lens - 1)
        result = self.dense(out)
        return result

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def gen_item_embedding(self):
        result = self.bsc_embedding.weight
        return result

    def forward(self, user_seqs, seq_lens):
        seqs_embed = self.bsc_embedding(user_seqs)
        return self.gen_user_embedding(seqs_embed, seq_lens)

    def calculate_loss(self, input):
        user_seqs = input.user_seqs
        seq_lens = input.seq_lens
        target_ids = input.target_ids
        neg_ids = input.neg_ids

        user_embed = self.gen_user_embedding(user_seqs, seq_lens)
        item_embed = self.gen_item_embedding()
        scores = torch.matmul(user_embed, item_embed.t())
        neg_embed = self.bsc_embedding(neg_ids)
        pos_embed = self.bsc_embedding(target_ids)
        pos_score = torch.sum(user_embed * pos_embed, dim=-1)  # [B]
        neg_score = torch.sum(user_embed * neg_embed, dim=-1)  # [B]
        loss = self.loss_fct(pos_score, neg_score)
        return loss

    def predict(self, input):
        user_seqs = input.user_seqs
        seq_lens = input.seq_lens

        user_embed = self.gen_user_embedding(user_seqs, seq_lens)
        item_embed = self.gen_item_embedding()
        scores = torch.matmul(user_embed, item_embed.t())
        # scores = torch.matmul(F.normalize(user_embed, dim=1),
        #                       F.normalize(item_embed.t(), dim=0))
        return scores

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def __str__(self):
        return self.__class__.__name__
