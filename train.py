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
samples = 2
test_samples = 10
