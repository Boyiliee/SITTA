import os
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.serialization import load_lua
import torchfile
from torch import autograd
import torchvision
import torch.nn.init as init
import functools
import numpy as np


############################
###  Model Architecture  ###
############################

# Generator
class TextureLayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(TextureLayer, self).__init__()
        self.avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(9),
            ConvBlock2d(channel, channel, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.ReLU() 
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x = y.expand_as(x)
        return x


class AutoGenerator(nn.Module):
    def __init__(self, opt):
        super(AutoGenerator, self).__init__()

        self.input_dim = opt['input_dim'] 
        self.output_dim = opt['output_dim']
        self.ngf = opt['ngf']
        self.norm_layer = nn.BatchNorm2d
        self.use_dropout = False
        self.n_blocks = opt['n_res']
        self.padding_type = opt['pad_type'] #'reflect'
        self.padding = opt['padding'] # 0
        self.n_downsampling = opt['n_downsampling']
        if type(self.norm_layer) == functools.partial:
            use_bias = self.norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = self.norm_layer == nn.InstanceNorm2d
        
        self.ms = True
        self.pono = True

        reflection_block = []
        reflection_block += [ConvBlock2d(self.input_dim, self.ngf, 5, 1, 2, pad_type='reflect')]
        reflection_block += [ConvBlock2d(self.ngf, self.ngf, 3, 1, 1)]
        reflection_block += [ConvBlock2d(self.ngf, self.ngf, 3, 1, 1)]
        enc_shape = []
        dim = self.ngf
        for _ in range(self.n_downsampling):  # add downsampling layers
            enc_shape += [Conv2dBlock(dim, dim * 2, 'conv', use_bias, pono=self.pono)]
            dim = dim * 2

        ### *** enc texture
        enc_texture = []
        enc_texture += [ConvBlock2d(self.input_dim, self.ngf, 5, 1, 2, pad_type='reflect')]
        dim = self.ngf
        for _ in range(self.n_downsampling):
            enc_texture += [Conv2dBlock(dim, dim * 2, 'conv', use_bias=True, normtype=nn.InstanceNorm2d)]
            dim = dim * 2
        enc_texture += [TextureLayer(dim, reduction=16)]

        ### *** dec
        pre_dec = []
        pre_dec += [ConvBlock2d(dim * 2, dim, 3, 1, 1)]
        for _ in range(self.n_blocks):       # add ResNet blocks
            pre_dec += [ResnetBlock(dim, padding_type=self.padding_type, norm_layer=self.norm_layer, use_dropout=self.use_dropout, use_bias=use_bias)]

        dec = []
        for _ in range(self.n_downsampling):  # add upsampling layers
            dec += [Conv2dBlock(dim, dim // 2, 'deconv', use_bias)]
            dim = dim // 2

        out_block = []
        out_block += [ConvBlock2d(dim, dim, 3, 1, 1)] 
        out_block += [ConvBlock2d(dim, self.output_dim, 5, 1, 2, activation='tanh')] 

        self.reflection_block = nn.Sequential(*reflection_block)
        self.enc_shape = nn.Sequential(*enc_shape)
        self.enc_texture = nn.Sequential(*enc_texture)
        self.pre_dec = nn.Sequential(*pre_dec)
        self.dec = nn.Sequential(*dec)
        self.out_block = nn.Sequential(*out_block)

        self.pono = PONO(affine=False) 
        self.ms = MS()

    def encode_s(self, x):
        x = self.reflection_block(x)
        stats = []
        for block in self.enc_shape:
            x, mean, std = block(x)
            stats.append((mean, std))
        stats.reverse()
        return x, stats

    def encode_t(self, x):
        texture = self.enc_texture(x)
        return texture

    def decode(self, shape, texture, new_stats):
        x = torch.cat((shape, texture), dim=1)
        x = self.pre_dec(x)
        i = 0
        for block in self.dec:
            if isinstance(block, Conv2dBlock):
                beta, gamma = new_stats[i]
                x = x * gamma + beta
                x = block(x)
                i += 1
        x = self.out_block(x)
        return x

    def forward(self, x, y):
        shape, new_stats = self.encode_s(x)
        texture = self.encode_t(y)
        out = self.decode(shape, texture, new_stats)
        return out


# WDiscriminator
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.num_layer = opt['num_layer']
        self.input_dim = opt['input_dim']
        self.ndf = opt['ndf']
        self.kernel_size = opt['kernel_size']
        self.stride = opt['stride']
        self.padding = opt['padding']
        self.gan_type = opt['gan_type']

        #######
        norm_layer=nn.BatchNorm2d
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [nn.Conv2d(self.input_dim, self.ndf, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, self.num_layer):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=use_bias),
                norm_layer(self.ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self.num_layer, 8)
        sequence += [
            nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=use_bias),
            norm_layer(self.ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(self.ndf * nf_mult, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding)]  # output 1 channel prediction map
        self.net = nn.Sequential(*sequence)

    def calc_gen_loss(self, input):
        # calculate the loss to train G
        out = self.forward(input)
        if self.gan_type == 'lsgan':
            loss = torch.mean((out - 1) ** 2)
        elif self.gan_type == 'nsgan':
            all1 = torch.ones_like(out.data).cuda()
            all1.requires_grad = False
            loss = torch.mean(F.binary_cross_entropy(F.sigmoid(out), all1))
        else:
            assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_dis_loss(self, input_fake, input_real):
        out_fake = self.forward(input_fake)
        out_real = self.forward(input_real)
        # Adversarial loss
        if self.gan_type == 'lsgan':
            loss_adv = torch.mean((out_fake - 0) ** 2) + torch.mean((out_real - 1) ** 2)
            loss_adv = loss_adv * 0.5
        elif self.gan_type == 'nsgan':
            all0 = torch.zeros_like(out_fake.data).cuda()
            all1 = torch.ones_like(out_real.data).cuda()
            all0.requires_grad = False
            all1.requires_grad = False
            loss_adv = torch.mean(F.binary_cross_entropy(F.sigmoid(out_fake), all0) +
                               F.binary_cross_entropy(F.sigmoid(out_real), all1))
            loss_adv = loss_adv * 0.5
        else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)

        return loss_adv

    def calc_gradient_penalty(self, input_real):
        gradient_penalty = 0
        input_real = input_real.requires_grad_(True)
        out = self.forward(input_real)
        out = out.mean(3).mean(2)  # average across all patch discriminators
        gradients = autograd.grad(outputs=out, inputs=input_real, grad_outputs=torch.ones(out.size()).cuda(), create_graph=True, retain_graph=True, only_inputs=True)
        for gradient in gradients: # for gradient w.r.t. each input
            if gradient is not None:
                gradient = gradient.view(gradient.size(0), -1)
                gradient_penalty += (gradient.norm(2, dim=1) ** 2).mean()
        return gradient_penalty

    def forward(self, x):
        x = self.net(x)
        return x

#########################
### Model Functions
##########################

def weights_init(init_method='gaussian'):
    def weights_init_method(m):
        classname = m.__class__.__name__
        # if classname.find('Conv2d') != -1:
        #     m.weight.data.normal_(0.0, 0.02)
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_method == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_method == 'xavier':
                init.xavier_normal_(m.weight.data)
            elif init_method == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_method == 'orthogonal':
                init.orthogonal_(m.weight.data)
            elif init_method == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_method)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return weights_init_method


def compute_recon_loss(input, target):
    # criterion = nn.MSELoss()
    # return torch.sqrt(criterion(input, target))
    return torch.mean(torch.abs(input - target))


def scale_function(x, scale_factor):
    main_function = nn.Upsample(scale_factor=scale_factor)
    x = main_function(x)
    return x


# VGG architecter, used for the perceptual loss using a pretrained VGG network
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
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

#########################
### Model Components
##########################
class ConvBlock2d(nn.Sequential):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, norm='None', activation='ReLU', pad_type='zero', use_bias='True'):
        super(ConvBlock2d, self).__init__()
        self.use_bias = use_bias
        # padding initialization
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # convolution setup
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        # normalization initialization
        normalization_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(normalization_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(normalization_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(normalization_dim)
        elif norm == 'None':
            self.norm = None
        else:
            assert 0, "Unsupported Normalization: {}".format(norm)

        # activation initialization
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activateion: {}".format(activation)

    def forward(self, input):
        x = self.conv(self.pad(input))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class PONO(nn.Module):
    def __init__(self, input_size=None, return_stats=False, affine=True, eps=1e-5):
        super(PONO, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std

class MS(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MS, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, inputdim, outputdim, operation, use_bias=True, use_relu=True, relu=True, ms=False, pono=False, norm=True, front=False, normtype=nn.BatchNorm2d, kerneltype=3):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.front = front
        # initialize normalization
        self.use_relu = use_relu
        self.norm = norm
        self.norm_flag = norm
        if operation == 'conv':
            if kerneltype == 3:
                self.conv = nn.Conv2d(inputdim, outputdim, kernel_size=3, stride=2, padding=1, bias=use_bias)
            elif kerneltype == 5:
                self.conv = nn.Conv2d(inputdim, outputdim, kernel_size=5, stride=2, padding=2, bias=use_bias)
            if normtype==nn.GroupNorm:
                self.norm = normtype(32, outputdim)
            else:
                self.norm = normtype(outputdim)
        elif operation == 'deconv':
            if kerneltype == 3:
                self.conv = nn.ConvTranspose2d(inputdim, outputdim, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias)
            elif kerneltype == 5:
                self.conv = nn.ConvTranspose2d(inputdim, outputdim, kernel_size=5, stride=2,
                                               padding=2, output_padding=1, bias=use_bias)
            if normtype==nn.GroupNorm:
                self.norm = normtype(16, outputdim)
            else:
                self.norm = normtype(outputdim)
        elif operation == 'stat_convs':
            if kerneltype == 3:
                self.conv = nn.Conv2d(inputdim, outputdim, 3, 1, 1)
            elif kerneltype == 5:
                self.conv = nn.Conv2d(inputdim, outputdim, 5, 1, 2)
            if normtype==nn.GroupNorm:
                self.norm = normtype(32, outputdim)
            else:
                self.norm = normtype(outputdim)
        # initialize activation
        self.activation = nn.ReLU(inplace=relu)
        # PONO-MS:
        self.pono = PONO(affine=False) if pono else None
        self.ms = MS() if ms else None

    def forward(self, x, beta=None, gamma=None):
        x = self.conv(x)
        mean, std = None, None
        if self.pono:
            x, mean, std = self.pono(x)
        if self.norm and self.norm_flag:
            x = self.norm(x)
        if self.ms:
            x = self.ms(x, beta, gamma)
        if self.use_relu and self.activation:
            x = self.activation(x)
        if mean is None:
            return x
        else:
            return x, mean, std


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out






