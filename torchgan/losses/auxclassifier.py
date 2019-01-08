import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss import GeneratorLoss, DiscriminatorLoss
from ..utils import reduce

__all__ = ['AuxiliaryClassifierGeneratorLoss', 'AuxiliaryClassifierDiscriminatorLoss']

class AuxiliaryClassifierGeneratorLoss(GeneratorLoss):
    r"""Auxiliary Classifier GAN (ACGAN) loss based on a from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
       reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction=self.reduction)

        r"""Defines the standard ``train_ops`` used by the Auxiliary Classifier generator loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows (label_g and label_d both could be either real labels or generated labels):

        1. :math:`fake = generator(noise, label_g)`
        2. :math:`value_1 = classifier(fake, label_g)`
        3. :math:`value_2 = classifier(real, label_d)`
        4. :math:`loss = loss\_function(value_1, label_g) + loss\_function(value_2, label_d)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized. For ACGAN, it must require
                                                   labels for training
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
    def train_ops(self, generator, discriminator, optimizer_generator,
            noise_prior, label_prior, batch_size, labels=None):
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator,
                    noise_prior, label_prior, batch_size, labels)
        if generator.label_type == 'required' and labels is None:
            raise Exception('GAN model requires label for training')
        if noise_prior is None:
            raise Exception('GAN model cannot be trained without sampling noise')
        if label_prior is None and generator.label_type == 'generated':
            raise Exception('GAN Model cannot be trained without sampling labels')
        if generator.label_type == 'none':
            raise Exception('Incorrect Model: ACGAN generator must require labels for training')

        noise = noise_prior(batch_size, generator.encoding_dims)
        optimizer_generator.zero_grad()
        if generator.label_type == 'required':
            fake = generator(noise, labels)
        elif generator.label_type == 'generated':
            label_gen = label_prior(batch_size)
            fake = generator(noise, label_gen)
        cgz = discriminator(fake, mode='classifier')
        if generator.label_type == 'required':
            loss = self.forward(cgz, labels)
        else:
            loss = self.forward(cgz, label_gen)
        loss.backward()
        optimizer_generator.step()
        return loss.item()

class AuxiliaryClassifierDiscriminatorLoss(DiscriminatorLoss):
    r"""Auxiliary Classifier GAN (ACGAN) loss based on a from
    `"Conditional Image Synthesis With Auxiliary Classifier GANs
    by Odena et. al. " <https://arxiv.org/abs/1610.09585>`_ paper

    Args:
       reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
       override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """
    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction=self.reduction)

    def train_ops(self, generator, discriminator, optimizer_discriminator,
            real_inputs, noise_prior, label_prior, labels=None):
        r"""Defines the standard ``train_ops`` used by the Auxiliary Classifier discriminator loss.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows (label_g and label_d both could be either real labels or generated labels):

        1. :math:`fake = generator(noise, label_g)`
        2. :math:`value_1 = classifier(fake, label_g)`
        3. :math:`value_2 = classifier(real, label_d)`
        4. :math:`loss = loss\_function(value_1, label_g) + loss\_function(value_2, label_d)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized. For ACGAN, it must require labels
                                                   for training
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_discriminator,
                    real_inputs, noise_prior, label_prior, labels)
        if noise_prior is None:
            raise Exception('GAN model cannot be trained without sampling noise')
        if labels is None:
            raise Exception('ACGAN Discriminator requires labels for training')
        if label_prior is None and generator.label_type == 'generated':
            raise Exception('GAN Model cannot be trained without sampling labels')
        if generator.label_type == 'none':
            raise Exception('Incorrect Model: ACGAN generator must require labels for training')

        batch_size = real_inputs.size(0)
        noise = noise_prior(batch_size, generator.encoding_dims)
        optimizer_discriminator.zero_grad()
        cx = discriminator(real_inputs, mode='classifier')
        if generator.label_type == 'required':
            fake = generator(noise, labels)
        elif generator.label_type == 'generated':
            label_gen = label_prior(batch_size)
            fake = generator(noise, label_gen)
        cgz = discriminator(fake, mode='classifier')
        if generator.label_type == 'required':
            loss = self.forward(cgz, labels) + self.forward(cx, labels)
        else:
            loss = self.forward(cgz, label_gen) + self.forward(cx, labels)
        loss.backward()
        optimizer_discriminator.step()
        return loss.item()
