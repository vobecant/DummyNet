""" Module implementing various loss functions """

import sys, os

from torch.nn import CrossEntropyLoss, BCELoss

from models.customLayers import BilinearInterpolation

tmp = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(tmp)
import torch as th
from torch import nn
from models.utils import *


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses

        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, height, alpha):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :param height: current height at which training is going on
        :param alpha: current value of the fader alpha
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


# =============================================================
# Normal versions of the Losses:
# =============================================================


class WGAN_GP(GANLoss):
    def __init__(self, dis, drift=0.001, use_gp=False, shape_loss=None):
        super().__init__(dis)
        self.name = 'wgan-gp'

        self.drift = drift
        self.use_gp = use_gp
        self.use_rock = dis.module.use_rock if isinstance(dis, nn.DataParallel) else dis.use_rock
        self.use_pred = dis.module.use_pred if isinstance(dis, nn.DataParallel) else dis.use_pred
        self.shape_loss = shape_loss
        self.downsampler = BilinearInterpolation(0.5)

    def set_rock_loss(self, rock_loss):
        self.shape_loss = rock_loss

    def __gradient_penalty(self, real_samps, fake_samps,
                           height, alpha, reg_lambda=10, mask=None):
        """
        private helper for calculating the gradient penalty
        :param real_samps: real samples
        :param fake_samps: fake samples
        :param height: current depth in the optimization
        :param alpha: current alpha for fade-in
        :param reg_lambda: regularisation lambda
        :return: tensor (gradient penalty)
        """
        batch_size = real_samps.shape[0]

        # if the mask should be used
        if mask is not None:
            # use mask
            fake_samps = fake_samps * mask
            real_samps = real_samps * mask

        # generate random epsilon
        epsilon = th.rand((batch_size, 1, 1, 1)).to(fake_samps.device)

        # create the merge of both real and fake samples
        merged = epsilon * real_samps + ((1 - epsilon) * fake_samps)
        merged.requires_grad_(True)

        # forward pass
        if self.use_rock or self.use_pred:
            op, _ = self.dis(merged, height, alpha)
        else:
            op = self.dis(merged, height, alpha)

        # perform backward pass from op to merged for obtaining the gradients
        gradient = th.autograd.grad(outputs=op, inputs=merged,
                                    grad_outputs=th.ones_like(op), create_graph=True,
                                    retain_graph=True, only_inputs=True)[0]

        # calculate the penalty using these gradients
        gradient = gradient.view(gradient.shape[0], -1)
        penalty = reg_lambda * ((gradient.norm(p=2, dim=1) - 1) ** 2).mean()

        # return the calculated penalty:
        return penalty

    def dis_loss(self, real_samps, fake_samps, height, alpha, mask=None):
        use_rock = self.dis.module.use_rock if isinstance(self.dis, nn.DataParallel) else self.dis.use_rock
        use_pred = self.dis.module.use_pred if isinstance(self.dis, nn.DataParallel) else self.dis.use_pred
        if mask is not None and not use_rock and not use_pred:
            # if the rock is not used and mask is available - use mask
            fake_samps = fake_samps * mask
            real_samps = real_samps * mask

        # define the (Wasserstein) loss
        if self.use_rock or self.use_pred:
            fake_out, fake_pred = self.dis(fake_samps, height, alpha)
            real_out, real_pred = self.dis(real_samps, height, alpha)
            # compute rock loss
            while fake_pred.shape[-1] != mask.shape[-1]:
                mask = self.downsampler(mask)
            mask_pix = torch.sum(mask)
            shape_loss_real = torch.sum(self.shape_loss(mask, real_pred * mask)) / mask_pix
            shape_loss_fake = torch.sum(self.shape_loss(mask, fake_pred * mask)) / mask_pix
            shape_loss = (shape_loss_real + shape_loss_fake)
            if self.use_rock:
                # multiplied by alpha since the alpha determines the level of influence of the ROCK layer
                shape_loss = shape_loss * alpha
        else:
            fake_out = self.dis(fake_samps, height, alpha)
            real_out = self.dis(real_samps, height, alpha)

        wgp_loss = (th.mean(fake_out) - th.mean(real_out) + (self.drift * th.mean(real_out ** 2)))

        if self.use_gp:
            # calculate the WGAN-GP (gradient penalty)
            gp = self.__gradient_penalty(real_samps, fake_samps, height, alpha)
            wgp_loss += gp

        if self.use_rock or self.use_pred:
            return wgp_loss, shape_loss, fake_pred
        else:
            return wgp_loss

    def gen_loss(self, _, fake_samps, height, alpha, mask=None):
        # calculate the WGAN loss for generator
        if mask is not None and not self.use_rock:
            fake_samps = fake_samps * mask

        if self.use_rock or self.use_pred:
            fake_out, fake_pred = self.dis(fake_samps, height, alpha)

            while fake_pred.shape[-1] != mask.shape[-1]:
                mask = self.downsampler(mask)

            mask_pix = torch.sum(mask)

            # it is negative since we want to maximize this value
            shape_loss = - torch.sum(self.shape_loss(mask, fake_pred * mask)) / mask_pix
            if self.use_rock:
                # multiplied by alpha since the alpha determines the level of influence of the ROCK layer
                shape_loss = shape_loss * alpha
        else:
            fake_out = self.dis(fake_samps, height, alpha)
        wgp_loss = -th.mean(fake_out)

        if self.use_rock or self.use_pred:
            return wgp_loss, shape_loss, fake_pred
        else:
            return wgp_loss


class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()

    def forward(self, x0, x1, mask):
        raise NotImplementedError("This is just interface!")


class L1MaskedLoss(MaskedLoss):
    def __init__(self, average, foreground, grayscale=False):
        super(L1MaskedLoss, self).__init__()
        self.name = 'l1_masked_loss'

        self.average = average
        self.foreground = foreground
        self.grayscale = grayscale

    def forward(self, target, generated, mask, batch_nn=None, alpha_channel=None):
        '''
        :param alpha_channel: the value of alpha channel used for mixing original and nearest-neighbor image, not to be mistaken with alha used for fade effect
        '''

        # if the background loss should be computed, invert mask such that 1 is background
        mask = mask if self.foreground else invert_bin_tensor(mask)
        mult = 3
        if self.grayscale:
            mult = 1
            generated = rgb2gray(generated)
            target = rgb2gray(generated)

        # compute L1 absolute difference just on images
        L1_abs_diff_masked = th.abs(generated - target) * mask
        L1_masked_sum = th.sum(L1_abs_diff_masked)

        if not self.average:
            return L1_masked_sum

        # number of pixels for computing average over L1 difference of image
        n_pix = th.sum(mask) * mult
        L1_avg = L1_masked_sum / n_pix

        return L1_avg


class L2MaskedLoss(MaskedLoss):
    def __init__(self, average, foreground, grayscale=False):
        super(L2MaskedLoss, self).__init__()
        self.name = 'l2_masked_loss'

        self.average = average
        self.foreground = foreground
        self.grayscale = grayscale

    def forward(self, target, generated, mask, batch_nn=None, alpha_channel=None):
        '''
        :param alpha_channel: the value of alpha channel used for mixing original and nearest-neighbor image, not to be mistaken with alha used for fade effect
        '''
        # if the background loss should be computed, invert mask such that 1 is background
        mask = mask if self.foreground else invert_bin_tensor(mask)
        if self.grayscale:
            generated = rgb2gray(generated)
            target = rgb2gray(generated)

        # compute L1 absolute difference just on images
        L2_abs_diff_masked = th.pow(generated - target, 2) * mask
        L2_masked_sum = th.sum(L2_abs_diff_masked)

        # number of pixels for computing average over L2 difference of image
        n_pix = th.sum(mask)
        L2_avg = L2_masked_sum / n_pix

        return L2_avg


def rgb2gray(x, r=0.299, g=0.587, b=0.114):
    red = x[:, 0, :, :] * r
    green = x[:, 1, :, :] * g
    blue = x[:, 2, :, :] * b
    gray = (red + green + blue)
    return gray  # return image of size (batch size, height, width)


class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()
        self.name = 'kl_loss'

    def forward(self, x):
        mu = torch.mean(x, dim=0)
        logvar = torch.var(x, dim=0).log()
        KLD = -0.5 * th.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, device):
        from models.networks import VGG19
        super(VGGLoss, self).__init__()
        self.device = device
        if self.device == 'cuda':
            n_avail = torch.cuda.device_count()
            # self.device = 'cuda:{:d}'.format(int(n_avail - 1))
            # print('Loading VGG to device: {}'.format(self.device))
            self.vgg = VGG19().to(device)
            self.vgg = nn.DataParallel(self.vgg)
            print('Loaded with DataParallel.')
        else:
            self.vgg = VGG19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, fake, real):
        fake_vgg, real_vgg = self.vgg(fake), self.vgg(real)
        loss = 0
        for i in range(len(fake_vgg)):
            loss += self.weights[i] * self.criterion(fake_vgg[i], real_vgg[i].detach())
        return loss
