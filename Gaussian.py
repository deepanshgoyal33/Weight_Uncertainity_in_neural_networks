import torch 


class Gaussian(object):
    def __init__(self,mu,rho):
        super().__init__()
        self.mu = mu
        self.rho= rho
        self.normal = torch.distributions.Normal(torch.tensor([0]),torch.tensor([1]))
    
    #Properties
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = sell.normal.sample(self.rho.size())
        return self.mu +self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2* math.pi)) - torch.log(self.sigma) - ((input-self.mu)**2)/(2*self.sigma **2)).sum()

#$ Scaled mixture of gaussians
class ScaledMixtureGaussian(object):
    def __init__(self, sigma1, sigma2, p1 ):
        super().__init__()
        self.p1= p1
        self.sigma1= sigma1
        self.sigma2= sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.