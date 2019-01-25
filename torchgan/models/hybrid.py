import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models import Generator

class HybridGenerator(Generator):
    def sample_images(self, x, *args, **kwargs):
        raise NotImplementedError

    def train(self, x, *args, **kwargs):
        raise NotImplementedError

    def forward(self, x, *args, **kwargs):
        if len(x.shape) == 2:
            return self.sample_images(x, *args, **kwargs)
        else:
            return self.train(x, *args, **kwargs)
