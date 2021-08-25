import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import os
import sys
from collections import OrderedDict


class FlowLayer(nn.Module):

    def __init__(self, channels=1, bottleneck=32, params=[1, 1, 1, 1, 1], n_iter=10):
        super(FlowLayer, self).__init__()
        self.bottleneck = nn.Conv3d(channels, bottleneck, stride=1, padding=0, bias=False, kernel_size=1)
        self.unbottleneck = nn.Conv3d(bottleneck * 2, channels, stride=1, padding=0, bias=False, kernel_size=1)
        self.bn = nn.BatchNorm3d(channels)
        channels = bottleneck

        self.n_iter = n_iter
        if params[0]:
            self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5, 0, 0.5]]]]).repeat(channels, channels, 1, 1))
            self.img_grad2 = nn.Parameter(
                torch.FloatTensor([[[[-0.5, 0, 0.5]]]]).transpose(3, 2).repeat(channels, channels, 1, 1))
        else:
            self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5, 0, 0.5]]]]).repeat(channels, channels, 1, 1),
                                         requires_grad=False)
            self.img_grad2 = nn.Parameter(
                torch.FloatTensor([[[[-0.5, 0, 0.5]]]]).transpose(3, 2).repeat(channels, channels, 1, 1),
                requires_grad=False)

        if params[1]:
            self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
            self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
            self.div = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
            self.div2 = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1))
        else:
            self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1),
                                       requires_grad=False)
            self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1),
                                        requires_grad=False)
            self.div = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1),
                                    requires_grad=False)
            self.div2 = nn.Parameter(torch.FloatTensor([[[[-1], [1]]]]).repeat(channels, channels, 1, 1),
                                     requires_grad=False)

        self.channels = channels

        self.t = 0.3
        self.l = 0.15
        self.a = 0.25

        if params[2]:
            self.t = nn.Parameter(torch.FloatTensor([self.t]))
        if params[3]:
            self.l = nn.Parameter(torch.FloatTensor([self.l]))
        if params[4]:
            self.a = nn.Parameter(torch.FloatTensor([self.a]))

    def norm_img(self, x):
        mx = torch.max(x)
        mn = torch.min(x)
        x = 255 * (x - mn) / (mn - mx)
        return x

    def forward_grad(self, x):
        grad_x = F.conv2d(F.pad(x, (0, 0, 0, 1)), self.f_grad)  # , groups=self.channels)
        grad_x[:, :, -1, :] = 0

        grad_y = F.conv2d(F.pad(x, (0, 0, 0, 1)), self.f_grad2)  # , groups=self.channels)
        grad_y[:, :, -1, :] = 0
        return grad_x, grad_y

    def divergence(self, x, y):
        tx = F.pad(x[:, :, :-1, :], (0, 0, 1, 0))
        ty = F.pad(y[:, :, :-1, :], (0, 0, 1, 0))

        grad_x = F.conv2d(F.pad(tx, (0, 0, 0, 1)), self.div)  # , groups=self.channels)
        grad_y = F.conv2d(F.pad(ty, (0, 0, 0, 1)), self.div2)  # , groups=self.channels)
        return grad_x + grad_y

    def forward(self, x):
        residual = x[:, :, :-1]
        x = self.bottleneck(x)
        inp = self.norm_img(x)
        x = inp[:, :, :-1]
        y = inp[:, :, 1:]
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)
        y = y.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, h, w)

        u1 = torch.zeros_like(x)
        u2 = torch.zeros_like(x)
        l_t = self.l * self.t
        taut = self.a / self.t

        grad2_x = F.conv2d(F.pad(y, (1, 1, 0, 0)), self.img_grad, padding=0, stride=1)  # , groups=self.channels)
        grad2_x[:, :, :, 0] = 0.5 * (x[:, :, :, 1] - x[:, :, :, 0])
        grad2_x[:, :, :, -1] = 0.5 * (x[:, :, :, -1] - x[:, :, :, -2])

        grad2_y = F.conv2d(F.pad(y, (0, 0, 1, 1)), self.img_grad2, padding=0, stride=1)  # , groups=self.channels)
        grad2_y[:, :, 0, :] = 0.5 * (x[:, :, 1, :] - x[:, :, 0, :])
        grad2_y[:, :, -1, :] = 0.5 * (x[:, :, -1, :] - x[:, :, -2, :])

        p11 = torch.zeros_like(x.data)
        p12 = torch.zeros_like(x.data)
        p21 = torch.zeros_like(x.data)
        p22 = torch.zeros_like(x.data)

        gsqx = grad2_x ** 2
        gsqy = grad2_y ** 2
        grad = gsqx + gsqy + 1e-12

        rho_c = y - grad2_x * u1 - grad2_y * u2 - x

        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

            v1 = torch.zeros_like(x.data)
            v2 = torch.zeros_like(x.data)
            mask1 = (rho < -l_t * grad).detach()
            v1[mask1] = (l_t * grad2_x)[mask1]
            v2[mask1] = (l_t * grad2_y)[mask1]

            mask2 = (rho > l_t * grad).detach()
            v1[mask2] = (-l_t * grad2_x)[mask2]
            v2[mask2] = (-l_t * grad2_y)[mask2]

            mask3 = ((mask1 ^ 1) & (mask2 ^ 1) & (grad > 1e-12)).detach()
            v1[mask3] = ((-rho / grad) * grad2_x)[mask3]
            v2[mask3] = ((-rho / grad) * grad2_y)[mask3]
            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2

            u1 = v1 + self.t * self.divergence(p11, p12)
            u2 = v2 + self.t * self.divergence(p21, p22)
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)

            p11 = (p11 + taut * u1x) / (1. + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * torch.sqrt(u1x ** 2 + u1y ** 2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * torch.sqrt(u2x ** 2 + u2y ** 2 + 1e-12))
            del u1x
            del u1y
            del u2x
            del u2y

        flow = torch.cat([u1, u2], dim=1)
        flow = flow.view(b, t, c * 2, h, w).contiguous().permute(0, 2, 1, 3, 4)
        flow = self.unbottleneck(flow)
        flow = self.bn(flow)
        return F.relu(residual + flow)


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        # print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        # print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        # print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        # print x.size()

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class ExtensionModule(nn.Module):
    def __init__(self, name):
        super(ExtensionModule, self).__init__()

        self.sobel = Unit3D(in_channels=1, output_channels=3, kernel_shape=[3, 3, 3], padding=0,
                         name=name + '/e_sobel')
        self.laplace = Unit3D(in_channels=1, output_channels=3, kernel_shape=[3, 3, 3], padding=0,
                          name=name + '/e_laplace')
        self.kirsh = Unit3D(in_channels=1, output_channels=3, kernel_shape=[3, 3, 3], padding=0,
                          name=name + '/e_kirsh')
        self.fusion1 = Unit3D(in_channels=9, output_channels=6, kernel_shape=[3, 3, 3], padding=0,
                          name=name + '/e_fusion1')
        self.fusion2 = Unit3D(in_channels=6, output_channels=3, kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/e_fusion2')

        self.name = name

    def forward(self, x):
        x_sobel = x[:, 0, : , :, :]
        x_laplace = x[:, 1, :, :, :]
        x_kirsh = x[:, 2, :, :, :]

        (batch, t, h, w) = x_sobel.size()

        x_sobel = x_sobel.view(batch, 1, t, h, w)
        x_laplace = x_laplace.view(batch, 1, t, h, w)
        x_kirsh = x_kirsh.view(batch, 1, t, h, w)

        x_sobel = self.sobel(x_sobel)
        x_laplace = self.laplace(x_laplace)
        x_kirsh = self.kirsh(x_kirsh)
        x = torch.cat([x_sobel, x_laplace, x_kirsh], dim=1)
        x = self.fusion1(x)
        x = self.fusion2(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionI3d(nn.Module):
    """Inception-v1 I3D architecture.
    The model is introduced in:
        Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
        Joao Carreira, Andrew Zisserman
        https://arxiv.org/pdf/1705.07750v1.pdf.
    See also the Inception architecture, introduced in:
        Going deeper with convolutions
        Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
        Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
        http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d', in_channels=3, dropout_keep_prob=0.5,
                 is_extended=False):
        """Initializes I3D model instance.
        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.
        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__()
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint
        self.logits = None
        self.is_extended = is_extended

        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)


        self.pre_process = ExtensionModule('ext_')

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=(3, 3, 3), name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        if self._final_endpoint == end_point: return

        end_point = 'Logits'
        self.avg_pool = nn.AvgPool3d(kernel_size=[2, 7, 7],
                                     stride=(1, 1, 1))
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.build()

    def replace_logits(self, num_classes):
        self._num_classes = num_classes
        self.logits = Unit3D(in_channels=384 + 384 + 128 + 128, output_channels=self._num_classes,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])

    def forward(self, x):
        x = self.pre_process(x)
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)  # use _modules to work with dataparallel
                if end_point == 'Mixed_5c':
                    mixed_5c = x

        x = self.logits(self.dropout(self.avg_pool(x)))
        if self._spatial_squeeze:
            logits = x.squeeze(3).squeeze(3)
        # logits is batch X time X classes, which is what we want to work with
        return mixed_5c, logits  # , self.end_points['Conv3d_1a_7x7'], self.end_points['Mixed_5c']

    def extract_features(self, x):
        for end_point in self.VALID_ENDPOINTS:
            if end_point in self.end_points:
                x = self._modules[end_point](x)
        return self.avg_pool(x)
