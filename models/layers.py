import torch
import torch.nn as nn
from torch.nn.init import normal_

class RLLoss(nn.Module):
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

    def __init__(self, gamma=1e-10, lamb=0):
        super(RLLoss, self).__init__()
        self.bpr = BPRLoss()
        self.gamma = gamma
        self.lamb = lamb

    def forward(self, prob, sim, pos_score, neg_score):
        pg_loss = - (torch.log(prob) * sim).mean()
        bpr_loss = self.bpr(pos_score, neg_score)
        loss = bpr_loss + self.lamb * pg_loss
        return loss


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
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score.unsqueeze(1) - neg_score)).mean()
        return loss


class MLPLayers(nn.Module):

    def __init__(self, layers, dropout=0, activation='relu', bn=False, init_method=None):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = self.activation_layer(activation_name=self.activation, emb_dim=output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def activation_layer(self, activation_name='relu', emb_dim=None):

        activation = None
        if isinstance(activation_name, str):
            if activation_name.lower() == 'sigmoid':
                activation = nn.Sigmoid()
            elif activation_name.lower() == 'tanh':
                activation = nn.Tanh()
            elif activation_name.lower() == 'relu':
                activation = nn.ReLU()
            elif activation_name.lower() == 'leakyrelu':
                activation = nn.LeakyReLU()
            elif activation_name.lower() == 'none':
                activation = None
        elif issubclass(activation_name, nn.Module):
            activation = activation_name()
        else:
            raise NotImplementedError("activation function {} is not implemented".format(activation_name))

        return activation

    def init_weights(self, module):
        pass
        # We just initialize the module with normal distribution as the paper said
        # if isinstance(module, nn.Linear):
        #     if self.init_method == 'norm':
        #         normal_(module.weight.data, 0, 0.01)
        #     if module.bias is not None:
        #         module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)