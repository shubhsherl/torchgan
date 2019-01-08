import torch
import torch.nn as nn

__all__ = ['GeneratorLoss', 'DiscriminatorLoss']

class GeneratorLoss(nn.Module):
    r"""Base class for all generator losses.

    .. note:: All Losses meant to be minimized for optimizing the Generator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """
    def __init__(self, reduction='mean', override_train_ops=None):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.arg_map = {}

    def set_arg_map(self, value):
        r"""Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_generator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(self, generator, discriminator, optimizer_generator,
                  noise_prior, label_prior, batch_size, labels=None):
        r"""Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``generator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value = discriminator(fake)`
        3. :math:`loss = loss\_function(value)`
        4. Backpropagate by computing :math:`\nabla loss`
        5. Run a step of the optimizer for generator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator, optimizer_generator,
                    noise_prior, label_prior, batch_size, labels)
        else:
            if noise_prior is None:
                raise Exception('GAN model cannot be trained without sampling noise')
            if labels is None and generator.label_type == 'required':
                raise Exception('GAN model requires labels for training')
            if label_prior is None and generator.label_type == 'generated':
                raise Exception('GAN Model cannot be trained without sampling labels')
            noise = noise_prior(batch_size, generator.encoding_dims)
            optimizer_generator.zero_grad()
            if generator.label_type == 'generated':
                label_gen = label_prior(batch_size)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            elif generator.label_type == 'generated':
                fake = generator(noise, label_gen)

            if discriminator.label_type == 'none':
                dgz = discriminator(fake)
            else:
                # Since discriminator is recieving samples from generator, the labels should be consistent
                # with the ones passed in the generator
                if generator.label_type == 'generated':
                    dgz = discriminator(fake, label_gen)
                elif generator.label_type == 'required':
                    dgz = discriminator(fake, labels)
                else:
                    raise Exception('Discriminator requires labels but Generator generates unlabelled samples')
            loss = self.forward(dgz)
            loss.backward()
            optimizer_generator.step()
            # NOTE(avik-pal): This will error if reduction is is 'none'
            return loss.item()

class DiscriminatorLoss(nn.Module):
    r"""Base class for all discriminator losses.

    .. note:: All Losses meant to be minimized for optimizing the Discriminator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """
    def __init__(self, reduction='mean', override_train_ops=None):
        super(DiscriminatorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.arg_map = {}

    def set_arg_map(self, value):
        r"""Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_discriminator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(self, generator, discriminator, optimizer_discriminator, real_inputs, noise_prior,
                  label_prior, labels=None):
        r"""Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value_1 = discriminator(fake)`
        3. :math:`value_2 = discriminator(real)`
        4. :math:`loss = loss\_function(value_1, value_2)`
        5. Backpropagate by computing :math:`\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
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
            return self.override_train_ops(self, generator, discriminator, optimizer_discriminator,
                   real_inputs, noise_prior, label_prior, labels)
        else:
            if noise_prior is None:
                raise Exception('GAN model cannot be trained without sampling noise')
            if labels is None and (generator.label_type == 'required' or discriminator.label_type == 'required'):
                raise Exception('GAN model requires labels for training')
            if label_prior is None and generator.label_type == 'generated':
                raise Exception('GAN Model cannot be trained without sampling labels')
            batch_size = real_inputs.size(0)
            noise = noise_prior(batch_size, generator.encoding_dims)
            if generator.label_type == 'generated':
                label_gen = label_prior(batch_size)
            optimizer_discriminator.zero_grad()
            if discriminator.label_type == 'none':
                dx = discriminator(real_inputs)
            elif discriminator.label_type == 'required':
                dx = discriminator(real_inputs, labels)
            else:
                dx = discriminator(real_inputs, label_gen)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            else:
                fake = generator(noise, label_gen)

            if discriminator.label_type == 'none':
                dgz = discriminator(fake.detach())
            else:
                if generator.label_type == 'generated':
                    dgz = discriminator(fake.detach(), label_gen)
                elif generator.label_type == 'required':
                    dgz = discriminator(fake.detach(), labels)
                else:
                    raise Exception('Discriminator requires labels but Generator generates unlabelled samples')
            loss = self.forward(dx, dgz)
            loss.backward()
            optimizer_discriminator.step()
            # NOTE(avik-pal): This will error if reduction is is 'none'
            return loss.item()
