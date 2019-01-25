import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models import HybridGenerator, AutoEncodingDiscriminator, Discriminator

__all__ = ['AdversarialAutoEncodingGenerator', 'AdversarialAutoEncodingDiscriminator']

class AdversarialAutoEncodingGenerator(HybridGenerator):
    def __init__(self, encoding_dims=100, out_size=32, out_channels=3, step_channels=64, scale_factor=2,
                 batchnorm=True, nonlinearity=None, last_nonlinearity=None, label_type='none'):
        super(AdversarialAutoEncodingGenerator, self).__init__(encoding_dims, label_type)
        model = AutoEncodingDiscriminator(in_size=out_size, in_channels=out_channels, encoding_dims=encoding_dims,
                    step_channels=step_channels, scale_factor=scale_factor, batchnorm=batchnorm,
                    nonlinearity=nonlinearity, last_nonlinearity=last_nonlinearity, energy=False, embeddings=True,
                    label_type=label_type)
        self.embeddings = model.embeddings
        self.init_dim = model.init_dim
        self.decoder = model.decoder
        self.encoder = model.encoder
        self.encoder_fc = model.fc
        del model

    def sample_images(self, noise):
        return self.decoder(noise)

    def train(self, x):
        x = self.encoder(x)
        if self.embeddings:
            return x
        x1 = x.view(-1, (self.init_dim ** 2) * x.size(1))
        x1 = self.decoder(self.fc(x1))
        return x1, x

class AdversarialAutoEncodingDiscriminator(Discriminator):
    def __init__(self, encoding_dims=100, batchnorm=True, nonlinearity=None, last_nonlinearity=None,
                 label_type='none'):
        super(AdversarialAutoEncodingDiscriminator, self).__init__(encoding_dims, label_type)
        input_dims = self.encoding_dims
        nl = nn.LeakyReLU(0.2) if nonlinearity is None else nonlinearity
        last_nl = nn.LeakyReLU(0.2) if last_nonlinearity is None else last_nonlinearity
        model = [nn.Sequential(nn.Linear(input_dims, input_dims // 2), nl)]
        size = input_dims // 2
        while size > 16:
            if batchnorm:
                model.append(nn.Sequential(nn.Linear(size, size // 2), nn.BatchNorm1d(size // 2), nl))
            else:
                model.append(nn.Sequential(nn.Linear(size, size // 2), nl))
            size = size // 2
        model.append(nn.Sequential(nn.Linear(size, 1), last_nl))
        self.model = nn.Sequential(*model)
        self._weight_initializer()

    def forward(self, x):
        return self.model(x)
