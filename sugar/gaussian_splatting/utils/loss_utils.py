#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

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
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def compute_scale_and_shift(prediction, target):
    '''
    Compute the scale and shift between the two depth maps
    Adapted from Instant-NGP-PP (https://github.com/zhihao-lin/instant-ngp-pp)
    '''
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(prediction * prediction)
    a_01 = torch.sum(prediction)
    ones = torch.ones_like(prediction)
    a_11 = torch.sum(ones)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(prediction * target)
    b_1 = torch.sum(target)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    # x_0 = torch.zeros_like(b_0)
    # x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        x_0 = (a_11 * b_0 - a_01 * b_1) / det
        x_1 = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        x_0 = torch.FloatTensor(0).cuda()
        x_1 = torch.FloatTensor(0).cuda()

    return x_0, x_1

def depth_loss(network_output, gt, scene_scale=5.0):
    '''
    Use monocular depth to regularize the depth prediction
    Adapted from Instant-NGP-PP (https://github.com/zhihao-lin/instant-ngp-pp)
    '''
    network_output = network_output.view(-1)
    gt = gt.view(-1)

    # Option 1: use whole GT depth map
    depth_2d = gt / 25
    mask = depth_2d > 0
    weight = torch.zeros_like(depth_2d).cuda()
    weight[mask] = 1.
    scale, shift = compute_scale_and_shift(network_output[mask].detach(), depth_2d[mask])

    # Option 2: use GT depth map but only consider network_output < scene_scale
    # depth_2d = gt / 25
    # mask = torch.logical_and(depth_2d > 0, network_output.detach() < scene_scale)
    # weight = torch.zeros_like(depth_2d).cuda()
    # weight[mask] = 1.
    # scale, shift = compute_scale_and_shift(network_output[mask].detach(), depth_2d[mask])

    return torch.mean(weight * torch.exp(-network_output.detach() / scene_scale) * (scale * network_output + shift - depth_2d) ** 2)

def normal_loss(network_output, gt, depth=None, scene_scale=5.0):
    '''
    Use normal to regularize the normal prediction
    Adapted from Instant-NGP-PP (https://github.com/zhihao-lin/instant-ngp-pp)
    '''
    assert network_output.shape[-1] == 3                                 # expected shape: (H, W, 3)
    normal_pred = F.normalize(network_output, p=2, dim=-1)               # (H, W, 3)
    normal_gt = F.normalize(gt, p=2, dim=-1)                             # (H, W, 3)
    if depth is not None:
        mask = torch.logical_and(depth > 0, depth < scene_scale)
        normal_pred = normal_pred[mask]
        normal_gt = normal_gt[mask]
    l1_loss = torch.abs(normal_pred - normal_gt).mean()                  # L1 loss (H, W, 3)
    cos_loss = -torch.sum(normal_pred * normal_gt, axis=-1).mean()       # Cosine similarity loss (H, W, 3)
    return l1_loss + 0.1 * cos_loss

# def opacity_loss(network_output):
#     '''
#     Use opacity to regularize it to be 0 or 1 to avoid floater
#     Adapted from Instant-NGP-PP (https://github.com/zhihao-lin/instant-ngp-pp)
#     '''
#     # TODO: add image mask (refer to Relightable-3D-Gaussian)
#     o = network_output.clamp(1e-6, 1 - 1e-6)
#     return torch.mean(-o * torch.log(o))

def opacity_loss(network_output):
    '''
    Use opacity to regularize it to be 0 or 1 to avoid floater
    '''
    return torch.mean(network_output)

def sparsity_loss(network_output):
    '''
    Use to regularize opacity values of each gaussian to approach either 0 or 1
    Adapted from GaussianShader (https://github.com/Asparagus15/GaussianShader/blob/main/utils/loss_utils.py)
    '''
    zero_epsilon = 1e-3
    val = torch.clamp(network_output, zero_epsilon, 1 - zero_epsilon)
    loss = torch.mean(torch.log(val) + torch.log(1 - val))
    return loss

def anisotropic_loss(gaussians_scale, r=3):
    '''
    Use to regularize gaussians size to be isotropic (avoid over-stretching gaussians)
    Reference from PhysGaussian (https://arxiv.org/pdf/2311.12198)
    '''
    # L_aniso = mean( max( max(scale)/min(scale), r ) - r)
    eps = 1e-6
    max_scale = torch.max(gaussians_scale, dim=-1).values
    min_scale = torch.min(gaussians_scale, dim=-1).values
    return torch.mean(torch.clamp(max_scale / (min_scale + eps), min=r) - r)