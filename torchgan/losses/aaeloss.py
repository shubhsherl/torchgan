import torch
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..models import AdversarialAutoEncodingGenerator
from .functional import minimax_generator_loss, minimax_discriminator_loss

__all__ = ['AdversarialAutoEncoderGeneratorLoss', 'AdversarialAutoEncoderDiscriminatorLoss']

class AdversarialAutoEncoderGeneratorLoss(GeneratorLoss):
    def __init__(self, recon_weight=0.999, gen_weight=0.001, reduction='mean', override_train_ops=None):
        super(AdversarialAutoEncoderGeneratorLoss, self).__init__(reduction, override_train_ops)
        self.gen_weight = gen_weight
        self.recon_weight = recon_weight

    def forward(self, real_inputs, gen_outputs, dgz):
        return self.recon_weight * F.mse_loss(real_inputs, gen_outputs) +\
            self.gen_weight + minimax_generator_loss(dgz, reduction=self.reduction)

    def train_ops(self, generator, discriminator, optimizer_generator, real_inputs,
                  device, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator, optimizer_generator,
                   real_inputs, device, labels)
        else:
            if isinstance(generator, AdversarialAutoEncodingGenerator):
                setattr(generator, "embeddings", False)
            recon, encodings = generator(real_inputs)
            optimizer_generator.zero_grad()
            dgz = discriminator(encodings)
            loss = self.forward(real_inputs, recon, dgz)
            loss.backward()
            optimizer_generator.step()
            return loss.item()

class AdversarialAutoEncoderDiscriminatorLoss(DiscriminatorLoss):
    def forward(self, dx, dgz):
        return minimax_discriminator_loss(dx, dgz)

    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs,
                  device, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator, optimizer_discriminator,
                   real_inputs, device, labels)
        else:
            if isinstance(generator, AdversarialAutoEncodingGenerator):
                setattr(generator, "embeddings", True)
            encodings = generator(real_inputs).detach()
            noise = torch.randn(real_inputs.size(0), generator.encoding_dims, device=device)
            optimizer_discriminator.zero_grad()
            dx = discriminator(noise)
            dgz = discriminator(encodings)
            loss = self.forward(dx, dgz)
            loss.backward()
            optimizer_discriminator.step()
            return loss.item()
