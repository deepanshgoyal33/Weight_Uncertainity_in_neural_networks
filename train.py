import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")

from model import *
from Gaussian import *

# TesnorBoard writing
def write_weight_histograms(epoch):
    writer.add_histogram('histogram/w1_mu', net.l1.weight_mu,epoch)
    writer.add_histogram('histogram/w1_rho', net.l1.weight_rho,epoch)
    writer.add_histogram('histogram/w2_mu', net.l2.weight_mu,epoch)
    writer.add_histogram('histogram/w2_rho', net.l2.weight_rho,epoch)
    writer.add_histogram('histogram/w3_mu', net.l3.weight_mu,epoch)
    writer.add_histogram('histogram/w3_rho', net.l3.weight_rho,epoch)
    writer.add_histogram('histogram/b1_mu', net.l1.bias_mu,epoch)
    writer.add_histogram('histogram/b1_rho', net.l1.bias_rho,epoch)
    writer.add_histogram('histogram/b2_mu', net.l2.bias_mu,epoch)
    writer.add_histogram('histogram/b2_rho', net.l2.bias_rho,epoch)
    writer.add_histogram('histogram/b3_mu', net.l3.bias_mu,epoch)
    writer.add_histogram('histogram/b3_rho', net.l3.bias_rho,epoch)

def write_loss_scalars(epoch, batch_idx, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    writer.add_scalar('logs/loss', loss, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/complexity_cost', log_variational_posterior-log_prior, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/log_prior', log_prior, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/log_variational_posterior', log_variational_posterior, epoch*NUM_BATCHES+batch_idx)
    writer.add_scalar('logs/negative_log_likelihood', negative_log_likelihood, epoch*NUM_BATCHES+batch_idx)


def train(model, epoch, optimizer):
    net.train()
    if epoh==0:
        write_weight_histograms(epoch)
    for batch_idx, (data,target) in enumerate (tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo(data,target)
        loss.backward()
        optimizer.step()
        write_loss_scalars(epoch,batch_idx,loss,log_prior, log_variational_posterior,negative_log_likelihood)
    write_weight_histograms(epoch+1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {'num_workers':1,'pin_memory':True} if torch.cuda.is_available() else {}

    B_size = 100
    T_B_size = 5
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./fmnist', train= True, download= True, transform= transforms.ToTensor()),batch_size=B_size, shuffle = True, **loader_kwargs )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            './fmnist', train=False, download=True,
            transform=transforms.ToTensor()),
        batch_size=T_B_size, shuffle=False, **loader_kwargs)

    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    num_batches = len(train_loader)
    num_test_batches = len(test_loader)

    classes =10
    train_epochs = 50
    SAMPLES = 2
    TEST_SAMPLES = 10

    ##Initiating the variables
    PI = 0.5
    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])
    
    model = BayesianNetwork()
    optimizer = optim.Adam(model.parameters())
    for epoch in range(train_epochs):
        train(model,epoch,optimizer)

if __name__ == "__main__":
    main()