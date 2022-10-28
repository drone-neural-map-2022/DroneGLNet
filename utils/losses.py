
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from utils.transformations import Project3D, BackprojectDepth


def get_smooth_loss(disp: torch.Tensor, img: torch.Tensor):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    from https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    from https://github.com/nianticlabs/monodepth2/blob/master/layers.py
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class MVS3DLoss(nn.Module):
    """
    Multi-view 3D structure consistency loss
    """
    def __init__(self, batch_size: int, height: int, width: int, reduction: str = 'mean'):
        super(MVS3DLoss, self).__init__()
        from .transformations import BackprojectDepth
        self.back_project = BackprojectDepth(batch_size, height, width)
        self.reduction = reduction

    def forward(
            self,
            depth_src: torch.Tensor,
            depth_tgt: torch.Tensor,
            pose: torch.Tensor,
            inv_intrinsics: torch.Tensor
    ):
        cloud_src = self.back_project(depth_src, inv_intrinsics)
        cloud_tgt = self.back_project(depth_tgt, inv_intrinsics)

        if self.reduction == 'mean':
            return (cloud_tgt - pose @ cloud_src).abs().mean()
        elif self.reduction == 'sum':
            return (cloud_tgt - pose @ cloud_src).abs().sum()
        elif self.reduction == 'none':
            return (cloud_tgt - pose @ cloud_src).abs()
        else:
            raise NotImplementedError


class EpipolarLoss(nn.Module):
    """
    Epipolar geometry for loss calculation
    """
    def __init__(self, batch_size: int, height: int, width: int, pix_group_size: int = 128, reduction: str = 'mean'):
        super(EpipolarLoss, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.pix_group_size = pix_group_size
        self.reduction = reduction

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(
            self,
            flow: torch.Tensor,
            pose: torch.Tensor,
            inv_intrinsics: torch.Tensor
    ) -> torch.Tensor:
        rotation = pose[:, :3, :3]
        translation = pose[:, :3, -1].unsqueeze(1)

        # Translation skew-matrix
        t_skew = torch.zeros_like(translation)
        t_skew = t_skew.repeat(1, translation.shape[-1], 1)
        t_skew[:, 0, 1] = -translation[..., 2]
        t_skew[:, 0, 2] = translation[..., 1]
        t_skew[:, 1, 2] = -translation[..., 0]
        t_skew = t_skew - t_skew.transpose(1, 2)

        # Move pixels with flow
        flattened_flow = flow.view(flow.shape[0], flow.shape[1], -1)
        pix_flow = torch.ones_like(self.pix_coords)
        pix_flow[:, :2] = self.pix_coords[:, :2] + flattened_flow

        # Epipolar geometry using the predicted flow as target image
        if self.pix_group_size > 1:
            pix_losses = None
            for idx in range(0, self.pix_coords.shape[-1], self.pix_group_size):
                pix_coords = self.pix_coords[..., idx:idx+self.pix_group_size]
                poseT_mm_invT = pix_coords.transpose(1, 2) @ inv_intrinsics[:, :3, :3].transpose(1, 2)
                rot_tskew_inv = rotation @ t_skew @ inv_intrinsics[:, :3, :3]
                if pix_losses is not None:
                    pix_losses = pix_losses + \
                                 (poseT_mm_invT @ rot_tskew_inv @ pix_flow[..., idx:idx+self.pix_group_size]
                                  ).sum()
                else:
                    pix_losses = (poseT_mm_invT @ rot_tskew_inv @ pix_flow[..., idx:idx+self.pix_group_size]
                                  ).sum()
        else:
            pix_losses = self.pix_coords.transpose(1, 2) @ inv_intrinsics[:, :3, :3].transpose(1, 2) \
                         @ rotation @ t_skew @ inv_intrinsics[:, :3, :3] @ pix_flow

        if self.reduction == 'sum':
            if self.pix_group_size > 1:
                return pix_losses
            else:
                return pix_losses.sum()
        elif self.reduction == 'mean':
            if self.pix_group_size > 1:
                return pix_losses / self.pix_coords.shape[-1]
            else:
                return pix_losses.mean()
        elif self.reduction == 'none':
            return pix_losses
        else:
            raise NotImplementedError


class AdaptivePhotometricLoss(nn.Module):
    def __init__(self, batch_size: int, height: int, width: int, r: float = 0.85, reduction: str = 'mean'):
        super(AdaptivePhotometricLoss, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.reduction = reduction

        self.project = Project3D(height, width)
        self.back_project = BackprojectDepth(batch_size, height, width)
        self.ssim = SSIM()
        self.r = r

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(
            self,
            img_src: torch.Tensor,
            img_tgt: torch.Tensor,
            depth: torch.Tensor,
            flow: torch.Tensor,
            pose: torch.Tensor,
            inv_intrinsics: torch.Tensor
    ):
        # 3D warp target image
        cam_coord_src = self.back_project(depth, inv_intrinsics)
        pix_coords_tgt = self.project(cam_coord_src, inv_intrinsics, pose)
        warped_tgt_3d = F.grid_sample(img_src, pix_coords_tgt,
                                      padding_mode='border', align_corners=True)

        # Flow warp target image
        pix_coords = self.pix_coords[:, :2].view(self.batch_size, 2, self.height, self.width).contiguous()
        pix_flow = pix_coords + flow
        warped_tgt_flow = F.grid_sample(img_src, pix_flow.permute(0, 2, 3, 1).contiguous(),
                                        padding_mode='border', align_corners=True)

        # Pixel-wise minimum of SSIM maps
        ssim_3d = self.ssim(img_tgt, warped_tgt_3d).mean(1, keepdim=True)
        ssim_flow = self.ssim(img_tgt, warped_tgt_flow).mean(1, keepdim=True)

        l1_3d = (img_tgt - warped_tgt_3d).abs().mean(1, keepdim=True)
        l1_flow = (img_tgt - warped_tgt_flow).abs().mean(1, keepdim=True)

        s_3d = self.r * (1 - ssim_3d) / 2 + (1 - self.r) * l1_3d
        s_flow = self.r * (1 - ssim_flow) / 2 + (1 - self.r) * l1_flow

        apc_loss = torch.stack([s_3d, s_flow], dim=1)
        apc_loss = apc_loss.min(dim=1)[0]

        if self.reduction == 'mean':
            return apc_loss.mean()
        elif self.reduction == 'sum':
            return apc_loss.sum()
        elif self.reduction == 'none':
            return apc_loss
        else:
            raise NotImplementedError


class FwdBwdFlowConsistency(nn.Module):
    def __init__(
            self,
            batch_size: int,
            height: int,
            width: int,
            alpha: float = 3.,
            beta: float = 0.05,
            scale: int = 0,
            reduction: str = 'mean'
    ):
        super(FwdBwdFlowConsistency, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.reduction = reduction

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor):
        # Warp
        bwd2fwd = F.grid_sample(flow_bwd, flow_fwd, padding_mode='border', align_corners=True)
        fwd2bwd = F.grid_sample(flow_fwd, flow_bwd, padding_mode='border', align_corners=True)

        # Consistency error
        diff_fwd = (bwd2fwd + flow_fwd).abs()
        diff_bwd = (fwd2bwd + flow_bwd).abs()

        # Condition
        bound_fwd = self.beta * (2 ** self.scale) * flow_fwd.norm(p=2, dim=1, keepdim=True)
        with torch.no_grad:
            bound_fwd = bound_fwd.clamp_min(self.alpha)

        bound_bwd = self.beta * (2 ** self.scale) * flow_bwd.norm(p=2, dim=1, keepdim=True)
        with torch.no_grad:
            bound_bwd = bound_bwd.clamp_min(self.alpha)

        # Mask
        noc_mask_src = ((2 ** self.scale) * diff_bwd.norm(p=2, dim=1, keepdim=True) < bound_bwd)
        noc_mask_tgt = ((2 ** self.scale) * diff_fwd.norm(p=2, dim=1, keepdim=True) < bound_fwd)

        # Consistency loss
        loss_fwd = (diff_fwd.mean(dim=1, keepdim=True) * noc_mask_tgt).sum() / noc_mask_tgt.sum()
        loss_bwd = (diff_bwd.mean(dim=1, keepdim=True) * noc_mask_src).sum() / noc_mask_src.sum()
        consistency_loss = (loss_fwd + loss_bwd) / 2

        if self.reduction == 'mean':
            return consistency_loss.mean()
        elif self.reduction == 'sum':
            return consistency_loss.sum()
        elif self.reduction == 'none':
            return consistency_loss
        else:
            raise NotImplementedError
