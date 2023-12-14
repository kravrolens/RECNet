import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np
from torchvision.models import vgg16


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return (-1) * ssim_map.mean()
    else:
        return (-1) * ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class GradientLoss(nn.Module):
    """Gradient Histogram Loss"""

    def __init__(self):
        super(GradientLoss, self).__init__()
        self.bin_num = 64
        self.delta = 0.2
        self.clip_radius = 0.2
        assert (self.clip_radius > 0 and self.clip_radius <= 1)
        self.bin_width = 2 * self.clip_radius / self.bin_num
        if self.bin_width * 255 < 1:
            raise RuntimeError("bin width is too small")
        self.bin_mean = np.arange(-self.clip_radius + self.bin_width * 0.5, self.clip_radius, self.bin_width)
        self.gradient_hist_loss_function = 'L2'
        # default is KL loss
        if self.gradient_hist_loss_function == 'L2':
            self.criterion = nn.MSELoss()
        elif self.gradient_hist_loss_function == 'L1':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.KLDivLoss()

    def get_response(self, gradient, mean):
        # tmp = torch.mul(torch.pow(torch.add(gradient, -mean), 2), self.delta_square_inverse)
        s = (-1) / (self.delta ** 2)
        tmp = ((gradient - mean) ** 2) * s
        return torch.mean(torch.exp(tmp))

    def get_gradient(self, src):
        right_src = src[:, :, 1:, 0:-1]  # shift src image right by one pixel
        down_src = src[:, :, 0:-1, 1:]  # shift src image down by one pixel
        clip_src = src[:, :, 0:-1, 0:-1]  # make src same size as shift version
        d_x = right_src - clip_src
        d_y = down_src - clip_src

        return d_x, d_y

    def get_gradient_hist(self, gradient_x, gradient_y):
        lx = None
        ly = None
        for ind_bin in range(self.bin_num):
            fx = self.get_response(gradient_x, self.bin_mean[ind_bin])
            fy = self.get_response(gradient_y, self.bin_mean[ind_bin])
            fx = torch.cuda.FloatTensor([fx])
            fy = torch.cuda.FloatTensor([fy])

            if lx is None:
                lx = fx
                ly = fy
            else:
                lx = torch.cat((lx, fx), 0)
                ly = torch.cat((ly, fy), 0)
        # lx = torch.div(lx, torch.sum(lx))
        # ly = torch.div(ly, torch.sum(ly))
        return lx, ly

    def forward(self, output, target):
        output_gradient_x, output_gradient_y = self.get_gradient(output)
        target_gradient_x, target_gradient_y = self.get_gradient(target)

        output_gradient_x_hist, output_gradient_y_hist = self.get_gradient_hist(output_gradient_x, output_gradient_y)
        target_gradient_x_hist, target_gradient_y_hist = self.get_gradient_hist(target_gradient_x, target_gradient_y)
        # loss = self.criterion(output_gradient_x_hist, target_gradient_x_hist) +
        # self.criterion(output_gradient_y_hist, target_gradient_y_hist)
        loss = self.criterion(output_gradient_x, target_gradient_x) + \
               self.criterion(output_gradient_y, target_gradient_y)
        return loss


class EXCloss(torch.nn.Module):
    # contrastive color correction loss

    def __init__(self, k=5, patch_res=50):
        super(EXCloss, self).__init__()

        self.D = nn.L1Loss()
        self.k = k  # number of negatives
        self.patch_res = patch_res  # path resolution
        vgg16_model = vgg16(pretrained=True)
        self.color_extractor = vgg16_model.features[:23]

    def forward(self, pred, gt_image, lq_image, mask):
        '''
        composition (N, 3, H, W):           composition image (input image to the network)
        pred        (N, 3, H, W):           predicted image (output of the network)
        gt_image    (N, 3, H, W):           ground truth image
        mask        (N, 1, H, W):           image mask
        '''

        B, C, H, W = pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]
        if self.k > pred.size(0) - 1:
            self.k = pred.size(0) - 1

        # calculate foreground and background style representations
        self.SR_extractor = self.color_extractor.to(pred.get_device())
        self.SR_extractor.eval()

        pooled_pred = F.max_pool2d(pred, (4, 4))
        pooled_gt_image = F.max_pool2d(gt_image, (4, 4))
        pooled_lq_image = F.max_pool2d(lq_image, (4, 4))
        pooled_mask = F.max_pool2d(mask, (4, 4))

        f_f = self.SR_extractor(pooled_pred * pooled_mask)
        f_b = self.SR_extractor(pooled_pred * (1 - pooled_mask))
        g_f_plus = self.SR_extractor(pooled_gt_image * pooled_mask)
        g_b_plus = self.SR_extractor(pooled_gt_image * (1 - pooled_mask))
        l_f_minus = self.SR_extractor(pooled_lq_image * pooled_mask)
        l_b_minus = self.SR_extractor(pooled_lq_image * (1 - pooled_mask))

        l_ss_cr = self.D(f_f, g_f_plus) / (self.D(f_f, g_f_plus) + self.D(l_f_minus, g_f_plus) + 1e-8) + \
                  self.D(f_b, g_b_plus) / (self.D(f_b, g_b_plus) + self.D(l_b_minus, g_b_plus) + 1e-8)

        # calculate Gram matrices based on style representations
        c = self.Gram(f_f, f_b)
        c_plus = self.Gram(g_f_plus, g_b_plus)
        c_minus = self.Gram(l_f_minus, l_b_minus)
        l_cs_cr = self.D(c, c_plus) / (self.D(c, c_plus) + self.D(c, c_minus) + 1e-8)

        return l_ss_cr + l_cs_cr

    def Gram(self, mat1, mat2):
        '''
        caculates the Gram matrix

        mat1 (N, 512, 32, 32):              feature map
        mat2 (N, 512, 32, 32):              feature map

        out (N, 512, 512):                  Gram matrix of both feature maps
        '''

        out = []
        for f1, f2 in zip(mat1, mat2):
            out.append(torch.matmul(f1.view(512, -1).T, f2.view(512, -1)))

        return torch.stack(out)
