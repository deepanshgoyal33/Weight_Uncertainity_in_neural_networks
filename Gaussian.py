import torch


class Gaussian(object):
    """
    Initialises the gaussian with mu and rho of the
    """

    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(
            torch.tensor([0]), torch.tensor([1]))

    # Properties
    def sigma(self):
        """
        Standarad deviation(sigma)= log(1+exp(rho))
        """
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) -
                ((input - self.mu)**2) / (2 * self.sigma ** 2)).sum()

# $ Scaled mixture of gaussians


class ScaledMixtureGaussian(object):
    """
    Scaled mixture of the gaussians = p1(Gaussian1)+ (1-p)(Gaussian2)
    """

    def __init__(self, sigma1, sigma2, p1):
        super().__init__()
        self.p1 = p1
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def scaledgaussian(self):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.p1 * prob1 + (1 - self.p1) * prob2)).sum()
