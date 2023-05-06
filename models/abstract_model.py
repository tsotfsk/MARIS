import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import BPRLoss
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_
from utils import MODAL_PATH_DICT



class SequentialModel(nn.Module):
    def __init__(self, config, logger):
        super(SequentialModel, self).__init__()

        # load base info
        self.logger = logger
        self.config = config
        
        self.dataset = config.dataset
        
        self.model_param = config.model_param
        
        self.user_num = config.user_num
        self.item_num = config.item_num
        
        self.loss_fct = BPRLoss()

        # self.apply(self._init_weights)
        # self.load_embedding()

    def calculate_loss(self, input):
        raise NotImplementedError

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

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def __str__(self):
        return self.__class__.__name__
