import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.archs.arch_util import Refine
from scipy import signal
from einops import rearrange


class GaussianBlur(nn.Module):

    def __init__(self):
        super(GaussianBlur, self).__init__()

        def get_kernel(size=51, std=3):
            """Returns a 2D Gaussian kernel array."""
            k = signal.gaussian(size, std=std).reshape(size, 1)
            k = np.outer(k, k)
            return k / k.sum()

        kernel = get_kernel(size=31, std=3)
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
        self.kernel = nn.Parameter(data=kernel, requires_grad=True)  # shape [3, 1, 193, 193]

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 3, h, w].
        """
        x = F.conv2d(x, self.kernel, padding=15, groups=1)
        return x


class IlluNet(nn.Module):
    def __init__(self, channel):
        super(IlluNet, self).__init__()
        self.net_convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, 1, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.Sigmoid())

    def forward(self, x):
        out = self.net_convs(x)
        return out


class RIN(nn.Module):
    def __init__(self, inplanes, selected_classes=3):
        super(RIN, self).__init__()

        self.IN = nn.InstanceNorm2d(inplanes, affine=True)
        self.CFR_branches = nn.ModuleList()
        self.CFR_tails = nn.ModuleList()
        for i in range(selected_classes):
            self.CFR_branches.append(
                nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False))
            self.CFR_tails.append(
                nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False))
        self.sigmoid = nn.Sigmoid()

        self.blur = GaussianBlur().cuda()

        self.fuse_1 = nn.Conv2d(inplanes * 2, inplanes, kernel_size=1)
        self.fuse_2 = nn.Conv2d(inplanes * 2, inplanes, kernel_size=1)
        self.alpha = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x, mask):
        outs = []
        mask_comp = torch.ones_like(mask) - mask

        mask1 = self.blur(mask)
        mask2 = self.blur(mask_comp)
        mask3 = torch.ones_like(mask)

        masks = [mask1, mask2, mask3]
        for idx, mask_i in enumerate(masks):
            mid = x * mask_i
            avg_out = torch.mean(mid, dim=1, keepdim=True)
            max_out, _ = torch.max(mid, dim=1, keepdim=True)
            atten = torch.cat([avg_out, max_out, mask_i], dim=1)
            atten = self.sigmoid(self.CFR_branches[idx](atten))
            out = mid * atten
            out = self.IN(out)
            out = self.CFR_tails[idx](out)
            outs.append(out)

        out1 = self.fuse_1(torch.cat([outs[0], outs[2]], 1))
        out2 = self.fuse_2(torch.cat([outs[1], outs[2]], 1))
        out_ = (1 - self.alpha) * out1 + self.alpha * out2

        return out_


class MCSA(nn.Module):
    # Mixed-scale Attention Module
    def __init__(self, dim, ffn_expansion_factor=1, num_heads=2):
        super(MCSA, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.num_heads = num_heads

        ####################################
        # 1. for global self-attention
        self.conv3x3_q = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features)
        self.conv5x5_q = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features)
        self.relu3_q = nn.ReLU(inplace=True)
        self.relu5_q = nn.ReLU(inplace=True)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_ch = nn.Conv2d(hidden_features * 2, hidden_features,
                                    kernel_size=1, padding=0, stride=1)
        ####################################

        ####################################
        # 2. for local convolution
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features * 2)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features * 2)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.relu5_1 = nn.ReLU(inplace=True)

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1)

    def forward(self, x, x_feat):
        ##########################################################################
        # 1. for local convolution: Context-Aware Feature Augmentation
        x_proj = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x_proj)).chunk(2, dim=1)  # also defined as qkv
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x_proj)).chunk(2, dim=1)
        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)
        out_local = self.project_out(x)

        ##########################################################################
        # 2. for channel self-attention
        x3_3 = self.relu3_q(self.conv3x3_q(x_feat))
        x3_5 = self.relu5_q(self.conv5x5_q(x_feat))

        outss_ = []
        for q, k, v in [[x1_3, x2_3, x3_3], [x1_5, x2_5, x3_5]]:
            B, C, H, W = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

            # b, 2x4, 384, 384  -> b, 2, (384x384), 4  -> b, 2, 4, (384x384)
            q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # torch.Size([8, 2, 4, 384^2])
            k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # torch.Size([8, 2, 4, 384^2])
            v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)  # torch.Size([8, 2, 4, 384^2])

            attn = ((q @ k.transpose(-2, -1)) * self.temperature)
            out = (attn.softmax(dim=-1)) @ v
            # ipdb.set_trace()
            out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=H, w=W)
            outss_.append(out)

        out_global = self.project_ch(torch.cat([outss_[0], outss_[1]], 1))

        return out_local, out_global


class DualBlock(nn.Module):
    def __init__(self, nc):
        super(DualBlock, self).__init__()

        self.norm = RIN(nc)
        self.mssa = MCSA(nc)

        self.fuse1 = nn.Conv2d(2 * nc, nc, 1, 1, 0)
        self.fuse2 = nn.Conv2d(2 * nc, nc, 1, 1, 0)
        self.post = nn.Sequential(nn.Conv2d(2 * nc, nc, 3, 1, 1),
                                  nn.LeakyReLU(0.1),
                                  nn.Conv2d(nc, nc, 3, 1, 1))

    def forward(self, x, mask):
        x_norm = self.norm(x, mask)
        x_local, x_global = self.mssa(x_norm, x)

        x_l = self.fuse1(torch.cat([x_norm, x_local], 1))
        x_g = self.fuse2(torch.cat([x_norm, x_global], 1))
        x_out = self.post(torch.cat([x_l, x_g], 1))

        return x_out + x


class DualProcess(nn.Module):
    def __init__(self, nc):
        super(DualProcess, self).__init__()
        self.conv1 = DualBlock(nc)
        self.conv2 = DualBlock(nc)
        self.conv3 = DualBlock(nc)
        self.conv4 = DualBlock(nc)
        self.conv5 = DualBlock(nc)
        self.cat = nn.Conv2d(5 * nc, nc, 1, 1, 0)
        self.refine = Refine(nc, 3)

    def forward(self, x, mask):
        x1 = self.conv1(x, mask)
        x2 = self.conv2(x1, mask)
        x3 = self.conv3(x2, mask)
        x4 = self.conv4(x3, mask)
        x5 = self.conv5(x4, mask)
        xout = self.cat(torch.cat([x1, x2, x3, x4, x5], 1))  # torch.Size([4, 8*5->8, 384, 384])

        # color refine
        xfinal = self.refine(xout)

        return xfinal, xout, x, x1, x2, x3, x4, x5


class InteractNet(nn.Module):
    def __init__(self, nc):
        super(InteractNet, self).__init__()
        self.learned_mask = IlluNet(nc)
        self.extract = nn.Conv2d(3, nc, 3, 1, 1)
        self.dualprocess = DualProcess(nc)

    def forward(self, x):

        x_pre = self.extract(x)

        mask = self.learned_mask(x_pre)
        x_final, xout, x, x1, x2, x3, x4, x5 = self.dualprocess(x_pre, mask)

        return torch.clamp(x_final + 0.00001, 0.0, 1.0), [x1, x2, x3, x4, x5], mask, x_pre
