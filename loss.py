import torch
import torch.nn as nn
from torch import autograd
import torchvision.models as models
from model import Discriminator
import numpy as np
import torch.nn.functional as F
from math import exp
import utils


# VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, device):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=False).to(device)
        vgg16.load_state_dict(torch.load('../vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram
# ======================================================================================================================


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False):
    L = 2  # min_val=-1 max_val=1

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, normalize=False, window=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, window=window, size_average=size_average, full=True)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


class MSSSIM(torch.nn.Module):
    def __init__(self, device, window_size=11, size_average=True, normalize=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.normalize = normalize
        self.window = create_window(window_size, channel=channel).to(device)

    def forward(self, output, gt):
        return 1 - msssim(output, gt, window_size=self.window_size, size_average=self.size_average, window=self.window,
                          normalize=self.normalize)
# ======================================================================================================================


# modified from WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, device, Lambda):
    B = real_data.size()[0]
    H = real_data.size()[2]
    alpha = torch.rand(B, 1, device=device)
    alpha = alpha.expand(B, int(real_data.nelement() / B)).contiguous()
    alpha = alpha.view(B, 6, H, H)

    fake_data = fake_data.view(B, 6, H, H)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty.sum().mean()
# ======================================================================================================================


class AllLoss(nn.Module):
    def __init__(self, D_in_ch, device, Lamda, lr, betas=(0.5, 0.9)):
        super(AllLoss, self).__init__()
        self.device = device
        self.l1 = nn.L1Loss().to(device)
        self.msssim = MSSSIM(device).to(device)
        # self.extractor = VGG16FeatureExtractor(device)
        self.discriminator = Discriminator(D_in_ch).to(device)
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.lamda = Lamda

    def forward(self, input, output, gt, train_G):
        # GAN-D
        self.discriminator.zero_grad()
        D_real = self.discriminator(torch.cat([input, gt], dim=1))
        D_real = D_real.mean().sum() * -1
        D_fake = self.discriminator(torch.cat([input, output.detach()], dim=1))
        D_fake = D_fake.mean().sum() * 1
        gp = calc_gradient_penalty(self.discriminator,
                                   torch.cat([input, gt], dim=1),
                                   torch.cat([input, output.detach()], dim=1),
                                   self.device, self.lamda)
        D_loss = D_fake + D_real + gp
        D_loss_data = D_loss.item()
        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()

        G_loss_data = 0
        l1_loss_data = 0
        msssim_loss_data = 0
        G_loss = 0
        if train_G:
            # GAN-G
            G_fake = self.discriminator(torch.cat([input, output], dim=1))
            G_fake = G_fake.mean().sum() * -1
            G_loss_data = G_fake.item()

            # L1
            l1_loss = self.l1(output, gt)
            l1_loss_data = l1_loss.item()

            # ms-ssim
            msssim_loss = self.msssim(output, gt)
            msssim_loss_data = msssim_loss.item()

            G_loss = l1_loss + 0.88 * msssim_loss + 1e-3 * G_fake

        # # Perceptual & Style
        # feat_output = self.extractor(output)
        # feat_gt = self.extractor(gt)
        #
        # P_loss = 0.0
        # for i in range(3):
        #     P_loss += 0.01 * self.l1(feat_output[i], feat_gt[i])
        # P_loss_data = P_loss.item()
        #
        # S_loss = 0.0
        # for i in range(3):
        #     S_loss += 120 * self.l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
        # S_loss_data = S_loss.item()
        #
        # G_loss = l1_loss + P_loss + S_loss + 0.1 * G_fake
        # return G_loss.sum(), G_loss_data, l1_loss_data, P_loss_data, S_loss_data

        return G_loss, G_loss_data, l1_loss_data, msssim_loss_data, D_loss_data


# ======================================================================================================================
class DSMapLoss(nn.Module):
    def __init__(self):
        super(DSMapLoss, self).__init__()
        self.l1_loss = torch.nn.L1Loss()

    def forward(self, output, target):
        output_D = utils.deprocess(output[:, :3, :, :])
        output_S = utils.deprocess(output[:, 3:6, :, :])
        target_D = utils.deprocess(target[:, 3:6, :, :])
        target_S = utils.deprocess(target[:, 9:12, :, :])
        l1_loss = self.l1_loss(torch.log(output_D + 0.01), torch.log(target_D + 0.01)) + \
                  self.l1_loss(torch.log(output_S + 0.01), torch.log(target_S + 0.01))
        color_loss = 1 - torch.cosine_similarity(output_D, target_D, dim=1) + \
                     1 - torch.cosine_similarity(output_S, target_S, dim=1)
        return l1_loss, torch.mean(color_loss)



