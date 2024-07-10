import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
import torch.autograd as autograd
import random
from torch.autograd import grad
from einops import rearrange

	
def projectors(hparams):
    if hparams["resnet18"] == False:
        n_outputs = 2048
    else:
        n_outputs = 512
    
    if hparams['dataset'] == "OfficeHome":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    elif hparams['dataset'] == "PACS":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 256
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )

    elif hparams['dataset'] == "TerraIncognita":
        scale_weights = 12
        pcl_weights = 1
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        encoder = nn.Sequential(
            nn.Linear(n_outputs, hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['hidden_size']),
            nn.BatchNorm1d(hparams['hidden_size']),
            nn.ReLU(inplace=True),
            dropout,
            nn.Linear(hparams['hidden_size'], hparams['out_dim']),
        )
    else:
        pass
    
    return encoder


def fea_proj(hparams):
    if hparams['dataset'] == "OfficeHome":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['hidden_size']),
            dropout,
            nn.Linear(hparams['hidden_size'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    elif hparams['dataset'] == "PACS":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 256
        hparams['out_dim'] = 256
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Linear(hparams['out_dim'], hparams['out_dim'], bias=False)
        
        # fc_proj = nn.Parameter(
        #     torch.FloatTensor(hparams['out_dim'],
        #                       hparams['out_dim'])
        # )

    elif hparams['dataset'] == "TerraIncognita":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 512
        hparams['out_dim'] = 512
        fea_proj = nn.Sequential(
            nn.Linear(hparams['out_dim'],
                      hparams['out_dim']),
        )
        fc_proj = nn.Parameter(
            torch.FloatTensor(hparams['out_dim'],
                              hparams['out_dim'])
        )
    else:
        pass
    
    return fea_proj, fc_proj



class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

class GradReverseKernelSupspaceIndicatorLogit(nn.Module):
    def __init__(self, ensemble_size, dim_in=128, dim_out=2, no_bias=False):
        super(GradReverseKernelSupspaceIndicatorLogit, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ensemble_size =ensemble_size
        if no_bias:
            self.ensemble_fc = EnsembleLinearNoBias(self.ensemble_size, self.dim_in, self.dim_out)
        else:
            self.ensemble_fc = EnsembleLinear(self.ensemble_size, self.dim_in, self.dim_out)
        self.ensemble_fc.apply(init_rf_weight)

        # self.register_buffer('threshold', torch.ones(self.ensemble_size))

    def forward(self, x):
        # x: Ensemble Batch Feature_in
        logit = self.ensemble_fc(GradReverse.apply(x))
        return logit

    def get_weight(self):
        return self.ensemble_fc.weight

    def get_bias(self):
        return self.ensemble_fc.bias


class KernelSupspaceIndicatorLogit(nn.Module):
    def __init__(self, ensemble_size, dim_in=128, dim_out=2, no_bias=False):
        super(KernelSupspaceIndicatorLogit, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ensemble_size =ensemble_size
        if no_bias:
            self.ensemble_fc = EnsembleLinearNoBias(self.ensemble_size, self.dim_in, self.dim_out)
        else:
            self.ensemble_fc = EnsembleLinear(self.ensemble_size, self.dim_in, self.dim_out)
        self.ensemble_fc.apply(init_rf_weight)

        # self.register_buffer('threshold', torch.ones(self.ensemble_size))

    def forward(self, x):
        # x: Ensemble Batch Feature_in
        logit = self.ensemble_fc(x)
        return logit

    def get_weight(self):
        return self.ensemble_fc.weight

    def get_bias(self):
        return self.ensemble_fc.bias



class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x
    



