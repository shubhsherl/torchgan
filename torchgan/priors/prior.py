import torch
import torch.nn as nn
import torch.distributions as dis

# Inherits nn.Module because priors can be learnable in general
class NoisePrior(nn.Module):
    def __init__(self, distribution=None):
        super(NoisePrior, self).__init__()
        if distribution is None:
            device = torch.device('cuda:0')
            mean = torch.Tensor([0.0]).to(device)
            std = torch.Tensor([1.0]).to(device)
            self.distribution = dis.Normal(mean, std)
        else:
            self.distribution = distribution

    def forward(self, batch_size, encoding_dims):
        return self.distribution.sample((batch_size, encoding_dims))

class LabelPrior(nn.Module):
    def __init__(self, distribution=None, num_classes=10):
        super(LabelPrior, self).__init__()
        if distribution is None:
            self.distribution = dis.Categorical(logits=torch.ones((num_classes, ), device=torch.device('cuda:0')))
        else:
            self.distribution = distribution

    def forward(self, batch_size):
        return self.distribution.sample((batch_size, ))
