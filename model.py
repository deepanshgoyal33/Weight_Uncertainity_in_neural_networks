import torch
import torch.nn as nn
import numpy as np

import nn.functional as F


from Gaussian import *


class BayesianNeuralNetLayer(nn.Module):

    """
Defining the funtionality of 1 Bayesian Layer
Input:
    in_features: no.of inputs in a layer
    out_features: output channels desired in a layer
Output:
    layer with variational inference bundeled in
    """

   def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = our_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(
            out_features, in_features).normal_(-0, 0.2))
        self.weigt_rho = nn.Prameter(torch.Tensor(
            out_features, in_features).uniform(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weigt_rho)
        # bias parameters: uniform distribution with given mean and standard
        # devatiation
        self.bias_mu = nn.Parameter(
    torch.Tensor(out_features).uniform_(
        0.3, 0.2))
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu. self.bias_rho)
        # Scaled Distributions
        self.weight_prior = ScaledMixtureGaussian(PI, SIGMA1, SIGMA2)
        self.bias_prior = ScaledMixtureGaussian(PI, SIGMA1, SIGMA2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calcualte_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(
                weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(
                weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class BayesianNetwork(nn.Module):
    """
    Classic 3 layeres Bayes by Backprop linear neural network
    """
    def __init__(self):
        super().__init__()
        self.l1 = BayesianNeuralNetLayer(feat_in, 400)
        self.l2 = BayesianNeuralNetLayer(400, 400)
        self.l3 = BayesianNeuralNetLayer(400, 10)

    def forward(self, x, sample=False):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.log_sogtmax(self.l3(x, sample),dim =1)
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return self.l1.log_variational_posterior + \
            self.l2.log_variational_posterior + self.l3.log_variational_posterior

    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES)
        log_priors = torch.zeros(samples)
        log_variational_posteriors = torch.zeros(samples)
        # Take a sample from the original distrbution and approximating the
        # nearest curve
        for i in range(samples):
            outputs[i] = self(input, sample= True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(
    outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) /NUM_BATCHES + negative_log_likelihood
        return loss


net = BayesianNetwork()
