import torch 
import torch.nn as nn
import numpy as np

import nn.functional as F


from Gaussian import *

class BayesianNeuralNetLayer(nn.Module):
    """
    Defining the funtionality of 1 Bayesian Layer
    Input:
        infeatures: no.of inputs in a layer 
        outfeatures: output channels desired in a layer
    Output:
        layer with variational inference bundeled in
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = our_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features,in_features).normal_(-0,0.2))
        self.weigt_rho = nn.Prameter(torch.Tensor(out_features,in_features).uniform(-5,-4))
        self.weight = Gaussian(self.weight_mu,self.weigt_rho)
        # bias parameters: uniform distribution with given mean and standard devatiation
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
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)

class BayesianNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = BayesianNeuralNetLayer(feat_in,400)
        self.l2 = BayesianNeuralNetLayer(400,200)
        self.l3 = BayesianNeuralNetLayer(200,10)

    def forward(self,x, sample= False):
        x= x.view(-1,28*28)
        x= 
