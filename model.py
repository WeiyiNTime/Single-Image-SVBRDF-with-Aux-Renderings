import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
import math
import utils
from torch.autograd import Variable
from einops import rearrange
import torchvision.ops.deform_conv as DeformConv
import torchvision.models as M


def print_model_structure(model):
    blank = ' '
    print('\t '+'-' * 95)
    print('\t ' + '|' + ' ' * 13 + 'weight name' + ' ' * 13 + '|' + ' ' * 15 + 'weight shape' + ' ' * 15 + '|'
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('\t ' + '-' * 95)
    num_para = 0

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 35:
            key = key + (35 - len(key)) * blank
        else:
            key = key[:32] + '...'
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        else:
            shape = shape[:37] + '...'
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('\t ' + '| {} | {} | {} |'.format(key, shape, str_num))
    print('\t ' + '-' * 95)
    print('\t ' + 'Total number of parameters: ' + str(num_para))
    print('\t CUDA: ' + str(next(model.parameters()).is_cuda))
    print('\t ' + '-' * 95 + '\n')


class DeformableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.padding = padding
        self.stride = stride
        self.org_conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)

        # kernel offset
        self.offset_conv = nn.Conv2d(in_ch, 2*(kernel_size**2), kernel_size=3, stride=stride, padding=1)
        init_offset = torch.zeros([2*(kernel_size**2), in_ch, 3, 3])
        self.offset_conv.weight = torch.nn.Parameter(init_offset)
        # kernel modulation
        self.mask_conv = nn.Conv2d(in_ch, (kernel_size ** 2), kernel_size=3, stride=stride, padding=1)
        init_mask = torch.full([(kernel_size ** 2), in_ch, 3, 3], 0.5)
        self.mask_conv.weight = torch.nn.Parameter(init_mask)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = self.sigmoid(self.mask_conv(x))
        out = DeformConv.deform_conv2d(x, offset=offset, weight=self.org_conv.weight, mask=mask,
                                       padding=(self.padding, self.padding),
                                       stride=(self.stride, self.stride))
        return out


class GatedDeformConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        if stride == 1:
            self.conv_attn = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
            self.conv_feat = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        elif stride == 2:
            self.conv_attn = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
            self.conv_feat = nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch)
        self.deform_conv = DeformableConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_ch)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        attn = self.sigmoid(self.conv_attn(x))
        x_feat = self.lrelu(self.norm(self.conv_feat(x)))
        x_attned = x_feat * attn

        x_valid_feat = self.lrelu(self.norm2(self.deform_conv(x_attned)))
        x_attned = x_attned + x_valid_feat * (1 - attn)
        return x_attned


class DoubleConvOrg(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.lrelu1(x1)

        x2 = self.conv2(x1)
        x2 = self.lrelu2(x2)
        return x2


class GatedDeformDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gated_deform_conv1 = GatedDeformConv(in_channels, out_channels, stride=2)
        self.gated_deform_conv2 = GatedDeformConv(out_channels, out_channels, stride=1)

    def forward(self, x):
        x = self.gated_deform_conv1(x)
        x = self.gated_deform_conv2(x)
        return x


class GatedDeformUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvOrg(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class GatedDeformUNet(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=True):
        super(GatedDeformUNet, self).__init__()
        base_ch = 48
        self.inc = DoubleConvOrg(n_channels, base_ch)
        self.down1 = GatedDeformDown(base_ch, base_ch*2)
        self.down2 = GatedDeformDown(base_ch*2, base_ch*4)
        self.down3 = GatedDeformDown(base_ch*4, base_ch*8)
        self.down4 = GatedDeformDown(base_ch*8, base_ch*8)
        factor = 2 if bilinear else 1
        self.up1 = GatedDeformUp(base_ch*16, base_ch*8 // factor, bilinear)
        self.up2 = GatedDeformUp(base_ch*8, base_ch*4 // factor, bilinear)
        self.up3 = GatedDeformUp(base_ch*4, base_ch*2 // factor, bilinear)
        self.up4 = GatedDeformUp(base_ch*2, base_ch, bilinear)
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        # elif isinstance(m, nn.BatchNorm2d):
        #     nn.init.constant_(m.bias, 0)
        #     nn.init.constant_(m.weight, 1.0)

    def forward(self, x):    # B 3 h w
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.tanh(self.out_conv(x))
        return out


# ======================================================================================================================
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ======================================================================================================================
class MaskNet(nn.Module):
    def __init__(self, out_ch, device):
        super(MaskNet, self).__init__()
        # https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth
        weights = M.EfficientNet_B3_Weights.DEFAULT
        backbone = M.efficientnet_b3(weights=weights)
        modules = list(backbone.features._modules.values())
        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1]).to(device)
        self.std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1]).to(device)

        self.stage1 = nn.Sequential(*modules[0:2])  # [1, 24, 128, 128]
        self.stage2 = modules[2]                    # [1, 32, 64, 64]
        self.stage3 = modules[3]                    # [1, 48, 32, 32]
        self.stage4 = nn.Sequential(*modules[4:6])  # [1, 136, 16, 16]
        self.stage5 = modules[6]                    # [1, 232, 8, 8]

        # upsample
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(232+136, 136, 3, padding=1),
            nn.BatchNorm2d(136),
            nn.SiLU()
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(136+48, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU()
        )
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv3 = nn.Sequential(
            nn.Conv2d(48+32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(32+24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(24, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, out_ch, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        s1 = self.stage1(x)  # [1, 24, 128, 128]
        s2 = self.stage2(s1)  # [1, 32, 64, 64]
        s3 = self.stage3(s2)  # [1, 48, 32, 32]
        s4 = self.stage4(s3)  # [1, 136, 16, 16]
        s5 = self.stage5(s4)  # [1, 232, 8, 8]

        s5 = self.up1(s5)
        s4 = self.conv1(torch.cat([s5, s4], dim=1))
        s4 = self.up2(s4)
        s3 = self.conv2(torch.cat([s4, s3], dim=1))
        s3 = self.up3(s3)
        s2 = self.conv3(torch.cat([s3, s2], dim=1))
        s2 = self.up4(s2)
        s1 = self.conv4(torch.cat([s2, s1], dim=1))
        out = self.up5(s1)
        out = self.conv5(out)
        return out


# ======================================================================================================================
class DownGlobal(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, mean_in_ch, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvGlobal(in_channels, mean_in_ch, out_channels)

    def forward(self, x, mean_feat):
        x = self.pool(x)
        x, mean_feat = self.conv(x, mean_feat)
        return x, mean_feat


class UpGlobal(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, mean_in_ch, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConvGlobal(in_channels, mean_in_ch, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvGlobal(in_channels, mean_in_ch, out_channels)

    def forward(self, x1, x2, mean_feat):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x, mean_feat = self.conv(x, mean_feat)
        return x, mean_feat


class DoubleConvGlobal(nn.Module):
    def __init__(self, in_ch, mean_in_ch, out_ch, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.linear2gen1 = nn.Linear(mean_in_ch, out_ch, bias=False)
        self.linear1 = nn.Linear(mean_in_ch+out_ch, out_ch, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=bias)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.linear2gen2 = nn.Linear(out_ch, out_ch, bias=False)
        self.linear2 = nn.Linear(out_ch*2, out_ch, bias=True)
        self.lrelu = nn.LeakyReLU(0.2)
        self.selu = nn.SELU()

    def forward(self, x, mean_feat):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        mean_1 = x1.mean(dim=3).mean(dim=2)
        b, c, _, _ = x1.shape
        x1 = x1 + self.linear2gen1(mean_feat).view(b, c, 1, 1)
        x1 = self.lrelu(x1)
        mean_feat = self.selu(self.linear1(torch.cat([mean_feat, mean_1], dim=1)))
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        mean_2 = x2.mean(dim=3).mean(dim=2)
        b, c, _, _ = x2.shape
        x2 = x2 + self.linear2gen2(mean_feat).view(b, c, 1, 1)
        x2 = self.lrelu(x2)
        mean_feat = self.selu(self.linear2(torch.cat([mean_feat, mean_2], dim=1)))
        return x2, mean_feat


class UNetGlobal(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=True):
        super(UNetGlobal, self).__init__()
        base_ch = 64
        self.linear_in = nn.Linear(n_channels, base_ch, bias=True)
        self.inc = DoubleConvGlobal(n_channels, base_ch, base_ch)
        self.down1 = DownGlobal(base_ch, base_ch, base_ch*2)
        self.down2 = DownGlobal(base_ch*2, base_ch*2, base_ch*4)
        self.down3 = DownGlobal(base_ch*4, base_ch*4, base_ch*8)
        self.down4 = DownGlobal(base_ch*8, base_ch*8, base_ch*8)
        factor = 2 if bilinear else 1
        self.up1 = UpGlobal(base_ch*16, base_ch*8, base_ch*8 // factor, bilinear)
        self.up2 = UpGlobal(base_ch*8, base_ch*4, base_ch*4 // factor, bilinear)
        self.up3 = UpGlobal(base_ch*4, base_ch*2, base_ch*2 // factor, bilinear)
        self.up4 = UpGlobal(base_ch*2, base_ch, base_ch, bilinear)
        self.selu = nn.SELU(inplace=True)
        self.out_conv = nn.Conv2d(base_ch, out_channels, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        mean_in = x.mean(dim=3).mean(dim=2)
        mean_feat = self.selu(self.linear_in(mean_in))
        x1, mean_feat = self.inc(x, mean_feat)
        x2, mean_feat = self.down1(x1, mean_feat)
        x3, mean_feat = self.down2(x2, mean_feat)
        x4, mean_feat = self.down3(x3, mean_feat)
        x5, mean_feat = self.down4(x4, mean_feat)
        x, mean_feat = self.up1(x5, x4, mean_feat)
        x, mean_feat = self.up2(x, x3, mean_feat)
        x, mean_feat = self.up3(x, x2, mean_feat)
        x, mean_feat = self.up4(x, x1, mean_feat)
        out = self.tanh(self.out_conv(x))
        return out