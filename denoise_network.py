import torch
import torch.nn as nn
from torch.nn import functional as F


def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'")

    return modules


class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(SingleConv, self).__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv1_in_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module('SingleConv1', SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        self.add_module('SingleConv2', SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))


class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='cge', num_groups=8, **kwargs):
        super(ExtResNetBlock, self).__init__()
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups)
        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
        self.conv3 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups)

        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        residual = out
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.non_linearity(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='crg',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='crg', num_groups=8):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            self.upsample = None
        else:
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
            x = torch.cat((encoder_features, x), dim=1)
        else:
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)
        return x


class FinalConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8):
        super(FinalConv, self).__init__()
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


class UNet3D(nn.Module):
    """
    Dual-branch 3D U-Net for denoising + anisotropic Y-upsampling.
    Branch 1: denoise input -> same resolution.
    Branch 2: upsample denoised output -> 4x along Y axis.
    """
    def __init__(self, in_channels=1, out_channels=1, f_maps=16, layer_order='cr', num_groups=8):
        super(UNet3D, self).__init__()

        f_maps1 = create_feature_maps(f_maps, number_of_fmaps=4)
        f_maps2 = create_feature_maps(f_maps, number_of_fmaps=4)

        encoders1 = []
        for i, out_feature_num in enumerate(f_maps1):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps1[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders1.append(encoder)
        self.encoders1 = nn.ModuleList(encoders1)

        encoders2 = []
        for i, out_feature_num in enumerate(f_maps2):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps2[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders2.append(encoder)
        self.encoders2 = nn.ModuleList(encoders2)

        decoders1 = []
        reversed_f_maps1 = list(reversed(f_maps1))
        for i in range(len(reversed_f_maps1) - 1):
            in_feature_num = reversed_f_maps1[i] + reversed_f_maps1[i + 1]
            out_feature_num = reversed_f_maps1[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders1.append(decoder)
        self.decoders1 = nn.ModuleList(decoders1)

        decoders2 = []
        reversed_f_maps2 = list(reversed(f_maps2))
        for i in range(len(reversed_f_maps2) - 1):
            in_feature_num = reversed_f_maps2[i] + reversed_f_maps2[i + 1]
            out_feature_num = reversed_f_maps2[i + 1]
            decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            decoders2.append(decoder)
        self.decoders2 = nn.ModuleList(decoders2)

        self.final_conv1 = nn.Sequential(nn.Conv3d(f_maps1[0], out_channels, 1))
        self.final_conv2 = nn.Sequential(nn.Conv3d(f_maps2[0], out_channels, 1))

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 1), mode='trilinear'),
            nn.Conv3d(f_maps2[0], f_maps2[0], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(f_maps2[0]),
            nn.ReLU(),
            nn.Conv3d(f_maps2[0], f_maps2[0], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(f_maps2[0]),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2, 1), mode='trilinear'),
            nn.Conv3d(f_maps2[0], f_maps2[0], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(f_maps2[0]),
            nn.ReLU(),
            nn.Conv3d(f_maps2[0], f_maps2[0], kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(f_maps2[0]),
            nn.ReLU(),
        )

    def forward(self, x):
        # Branch 1: denoise
        encoders_features = []
        for encoder in self.encoders1:
            x = encoder(x)
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders1, encoders_features):
            x = decoder(encoder_features, x)
        x = self.final_conv1(x)

        # Branch 2: upsample from denoised
        y = x
        encoders_features = []
        for encoder in self.encoders2:
            y = encoder(y)
            encoders_features.insert(0, y)
        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.decoders2, encoders_features):
            y = decoder(encoder_features, y)
        y = self.upsample(y)
        y = self.final_conv2(y)

        return x, y
