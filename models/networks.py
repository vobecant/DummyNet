import sys, os

import torchvision

tmp = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
sys.path.append(tmp)
from torch import nn
from models.utils import *


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        if h_relu4.shape[-1] == 1:
            out = [h_relu1, h_relu2, h_relu3, h_relu4]
            return out
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VAE_enc_only(nn.Module):
    def __init__(self, in_channels=3, base_features=8, code_dim_base=100):
        super(VAE_enc_only, self).__init__()

        # define building blocks
        encoder = [
            nn.Conv2d(in_channels, base_features, 5, stride=2, padding=2),
            nn.ReLU(inplace=True)
        ]

        in_features = base_features
        out_features = 2 * in_features
        for _ in range(2):
            encoder += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = 2 * in_features

        encoder += [
            nn.Conv2d(in_features, out_features, 3, stride=5, padding=1)
        ]

        self.encoder = nn.Sequential(*encoder)

        out_features *= 4

        encoder1 = [
            nn.Linear(out_features, code_dim_base)
        ]
        self.encoder_mu = nn.Sequential(*encoder1)

        encoder2 = [
            nn.Linear(out_features, code_dim_base)
        ]
        self.encoder_logvar = nn.Sequential(*encoder2)

        self.out_features = out_features

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, return_mu_logvar=True):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        if return_mu_logvar:
            return mu, logvar, z
        return z
