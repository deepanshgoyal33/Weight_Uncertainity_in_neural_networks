import torch 
import torch.nn as nn
import numpy as np

from Gaussian import *

class BayesianNeuralNetLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = our_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features,in_features).normal_(-0,0.2))
        self.weigt_rho = nn.Prameter(torch.Tensor(out_features,in_features).uniform(-5,-4))
        self.weight = Gaussian(self.weight_mu,self.weigt_rho)
        # bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(0.3,0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu. self.bias_rho)
        # Scaled Distributions 
        self.weight_prior = ScaledMixtureGaussian(PI,SIGMA1, SIGMA2 )
        self.bias_prior = ScaledMixtureGaussian(PI,SIGMA1, SIGMA2 )
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self,input, sample=False, calcualte_log_probs= False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
