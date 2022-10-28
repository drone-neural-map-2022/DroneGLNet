
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from utils.losses import SSIM, get_smooth_loss
from utils.transformations import transformation_from_parameters


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)
    wrapper.has_run = False
    return wrapper


class GLNetLoss(nn.Module):
    def __init__(
            self,
            img_size: tuple,
            scales: int,
            mvs_weight: float,
            epi_weight: float,
            apc_weight: float,
            disp_smooth: float,
            flow_smooth: float,
            flow_cons_params: tuple = None,
            flow_cons_weight: float = 0.,
            ssim_r: float = 0.85,
            reduction: str = 'mean'
    ):
        super(GLNetLoss, self).__init__()
        self.height = img_size[0]
        self.width = img_size[1]
        self.scales = scales
        self.mvs_weight = mvs_weight
        self.epi_weight = epi_weight
        self.apc_weight = apc_weight
        self.disp_smooth = disp_smooth
        self.flow_smooth = flow_smooth
        self.flow_cons_params = flow_cons_params
        self.flow_cons_weight = flow_cons_weight
        self.ssim_r = ssim_r
        self.reduction = reduction

        self.pix_coords_pyramid = {}
        for scale in range(scales):
            self.pix_coords_pyramid[scale] = \
                self.__generate_pix_coords(self.height // (2 ** scale), self.width // (2 ** scale))

        self.ssim_f_3d = SSIM()
        self.ssim_f_flow = SSIM()

    def __generate_pix_coords(self, height: int, width: int):
        meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        id_coords = torch.from_numpy(id_coords)
        ones = torch.ones(1, 1, height * width)
        pix_coords = torch.unsqueeze(
            torch.stack([id_coords[0].view(-1), id_coords[1].view(-1)], 0)
            , 0
        )
        pix_coords = torch.cat([pix_coords, ones], 1)
        return pix_coords

    @run_once
    def __set_batch_size(self, batch_size: int, ref_tensor: torch.Tensor):
        self.batch_size = batch_size
        for i in range(len(self.pix_coords_pyramid)):
            self.pix_coords_pyramid[i] = self.pix_coords_pyramid[i].repeat(batch_size, 1, 1).to(ref_tensor)

    def forward(
            self,
            inputs: dict,
            depths: dict,
            poses: dict,
            flows_fwd: dict,
            scales: int,
            disps: dict = None,
            flows_bwd: dict = None
    ):
        pose_frames = list(poses.keys())
        flow_frames = list(flows_fwd.keys())

        calc_disp_smooth = (disps is not None and self.disp_smooth != 0.)
        calc_flow_smooth = self.flow_smooth != 0.
        calc_flow_consistency = (flows_bwd is not None and self.flow_cons_weight != 0.)

        mvs_loss = 0.
        epipolar_loss = 0.
        apc_loss = 0.
        disp_smooth_loss = 0.
        flow_smooth_loss = 0.
        flow_consistency_loss = 0.

        for scale in range(scales):
            for frame_group in list(set(pose_frames) & set(flow_frames)):
                pose_params = poses[frame_group]
                flow_fwd = flows_fwd[frame_group][('flow', scale)]
                depth_src = depths[frame_group[0]][('depth', scale)]
                depth_tgt = depths[frame_group[-1]][('depth', scale)]
                inv_intrinsics = inputs[('inv_K', scale)]
                intrinsics = inputs[('K', scale)]
                image_src = inputs[('color', frame_group[0], scale)]
                image_tgt = inputs[('color', frame_group[-1], scale)]

                pose = transformation_from_parameters(pose_params['axisangle'], pose_params['translation'])

                # Multi-view 3D structure loss
                mvs_loss += self.mvs3d_loss(depth_src, depth_tgt, pose, inv_intrinsics, scale)

                # Epipolar loss
                epipolar_loss += self.epipolar_loss(flow_fwd, pose, inv_intrinsics, scale, pix_group_size=128)

                # Adaptive Photometric Loss
                apc_loss += self.adaptive_photometric_loss(image_src, image_tgt, depth_src, flow_fwd,
                                                           pose, intrinsics, inv_intrinsics, scale)

            if calc_disp_smooth:
                # Disparity smoothness loss
                for frame_id, disp_i in disps.items():
                    disp_e = disp_i[('disp', scale)]
                    mean_disp = disp_e.mean(2, True).mean(3, True)
                    norm_disp = disp_e / (mean_disp + 1e-7)
                    disp_smooth_loss += \
                        (self.disp_smooth / (2 ** scale) *
                         get_smooth_loss(norm_disp, inputs[('color', frame_id, scale)]))

            if calc_flow_smooth:
                # Flow smoothness loss
                div = (2 ** (scale + 1)) if flows_bwd is not None else (2 ** scale)
                for frame_group, flow_fwd_i in flows_fwd.items():
                    for chan in range(2):
                        flow_smooth_loss += (self.flow_smooth / div *
                                             get_smooth_loss(flow_fwd_i[('flow', scale)][:, chan].unsqueeze(1),
                                                             inputs[('color', frame_group[-1], scale)]))
                        if flows_bwd is not None:
                            flow_bwd_i = flows_bwd[frame_group]
                            flow_smooth_loss += (self.flow_smooth / div *
                                                 get_smooth_loss(flow_bwd_i[('flow', scale)][:, chan].unsqueeze(1),
                                                                 inputs[('color', frame_group[0], scale)]))

            if calc_flow_consistency:
                # Forward-backward flow consistency
                for f_idx, flow_fwd_i in flows_fwd.items():
                    flow_bwd_i = flows_bwd[f_idx]
                    flow_consistency_loss += self.fwd_bwd_flow_consistency(flow_fwd_i[('flow', scale)],
                                                                           flow_bwd_i[('flow', scale)],
                                                                           scale)

        total_loss = self.mvs_weight * mvs_loss + self.epi_weight * epipolar_loss + self.apc_weight * apc_loss
        if calc_disp_smooth:
            total_loss += disp_smooth_loss
        if calc_flow_smooth:
            total_loss += flow_smooth_loss
        if calc_flow_consistency:
            total_loss += (self.flow_cons_weight * flow_consistency_loss)
        total_loss /= self.scales

        loss_parts = {
            'mvs': mvs_loss / self.scales,
            'epi': epipolar_loss / self.scales,
            'apc': apc_loss / self.scales,
            'ds': disp_smooth_loss / self.scales,
            'fs': flow_smooth_loss / self.scales,
            'fc': flow_consistency_loss / self.scales
        }
        return total_loss, loss_parts

    def back_project(self, depth: torch.Tensor, inv_intrinsics: torch.Tensor, scale: int) -> torch.Tensor:
        self.__set_batch_size(depth.shape[0], depth)

        cam_points = torch.matmul(inv_intrinsics[:, :3, :3], self.pix_coords_pyramid[scale])
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.pix_coords_pyramid[scale][:, -1].unsqueeze(1)], 1)

        return cam_points

    def project_3d(self, cloud: torch.Tensor, intrinsics: torch.Tensor, pose: torch.Tensor, scale: int) -> torch.Tensor:
        self.__set_batch_size(cloud.shape[0], cloud)

        P = torch.matmul(intrinsics, pose)[:, :3, :]
        cam_points = torch.matmul(P, cloud)
        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-6)
        pix_coords = pix_coords.view(-1, 2, self.height // (2 ** scale), self.width // (2 ** scale))
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width // (2 ** scale) - 1
        pix_coords[..., 1] /= self.height // (2 ** scale) - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

    def mvs3d_loss(
            self,
            depth_src: torch.Tensor,
            depth_tgt: torch.Tensor,
            pose: torch.Tensor,
            inv_intrinsics: torch.Tensor,
            scale: int
    ):
        self.__set_batch_size(depth_src.shape[0], depth_src)
        cloud_src = self.back_project(depth_src, inv_intrinsics, scale)
        cloud_tgt = self.back_project(depth_tgt, inv_intrinsics, scale)

        if self.reduction == 'mean':
            return (cloud_tgt - pose @ cloud_src).abs().mean()
        elif self.reduction == 'sum':
            return (cloud_tgt - pose @ cloud_src).abs().sum()
        elif self.reduction == 'none':
            return (cloud_tgt - pose @ cloud_src).abs()
        else:
            raise NotImplementedError

    def epipolar_loss(
            self,
            flow: torch.Tensor,
            pose: torch.Tensor,
            inv_intrinsics: torch.Tensor,
            scale: int,
            pix_group_size: int = 1,
    ):
        self.__set_batch_size(pose.shape[0], pose)
        pix_coords = self.pix_coords_pyramid[scale]
        rotation = pose[:, :3, :3]
        translation = pose[:, :3, -1].unsqueeze(1)

        # Translation skew-matrix
        t_skew = torch.zeros_like(translation)
        t_skew = t_skew.repeat(1, translation.shape[-1], 1)
        t_skew[:, 0, 1] = -translation[:, 0, 2]
        t_skew[:, 0, 2] = translation[:, 0, 1]
        t_skew[:, 1, 2] = -translation[:, 0, 0]
        t_skew = t_skew - t_skew.transpose(1, 2)

        # Move pixels with flow
        flattened_flow = flow.view(flow.shape[0], flow.shape[1], -1)
        pix_flow = torch.ones_like(pix_coords)
        pix_flow[:, :2] = pix_coords[:, :2] + flattened_flow

        # Epipolar geometry using the predicted flow as target image
        if pix_group_size > 1:
            pix_losses = 0.
            for idx in range(0, pix_coords.shape[-1], pix_group_size):
                pix_coords_i = pix_coords[..., idx:idx + pix_group_size]
                poseT_mm_invT = pix_coords_i.transpose(1, 2) @ inv_intrinsics[:, :3, :3].transpose(1, 2)
                rot_tskew_inv = rotation @ t_skew @ inv_intrinsics[:, :3, :3]
                pix_losses += (poseT_mm_invT @ rot_tskew_inv @ pix_flow[..., idx:idx + pix_group_size]
                               ).abs().sum()
        else:
            pix_losses = (pix_coords.transpose(1, 2) @ inv_intrinsics[:, :3, :3].transpose(1, 2) @
                          rotation @ t_skew @ inv_intrinsics[:, :3, :3] @ pix_flow
                          ).abs().sum()

        if self.reduction == 'sum':
            if pix_group_size > 1:
                return pix_losses
            else:
                return pix_losses.sum()
        elif self.reduction == 'mean':
            if pix_group_size > 1:
                return pix_losses / pix_coords.shape[-1]
            else:
                return pix_losses.mean()
        elif self.reduction == 'none':
            return pix_losses
        else:
            raise NotImplementedError

    def adaptive_photometric_loss(
            self,
            img_src: torch.Tensor,
            img_tgt: torch.Tensor,
            depth: torch.Tensor,
            flow: torch.Tensor,
            pose: torch.Tensor,
            intrinsics: torch.Tensor,
            inv_intrinsics: torch.Tensor,
            scale: int
    ):
        self.__set_batch_size(pose.shape[0], pose)
        r = self.ssim_r
        pix_coords = self.pix_coords_pyramid[scale]

        # 3D warp target image
        cam_coord_src = self.back_project(depth, inv_intrinsics, scale)
        pix_coords_tgt = self.project_3d(cam_coord_src, intrinsics, pose, scale)
        warped_tgt_3d = F.grid_sample(img_src, pix_coords_tgt, padding_mode='border', align_corners=True)

        # Flow warp target image
        pix_coords_i = pix_coords[:, :2]\
            .view(self.batch_size, 2, self.height // (2 ** scale), self.width // (2 ** scale)).contiguous()
        pix_flow = pix_coords_i + flow
        warped_tgt_flow = F.grid_sample(img_src, pix_flow.permute(0, 2, 3, 1).contiguous(),
                                        padding_mode='border', align_corners=True)

        # Pixel-wise minimum of SSIM maps
        ssim_3d = self.ssim_f_3d(img_tgt, warped_tgt_3d).mean(1, keepdim=True)
        ssim_flow = self.ssim_f_flow(img_tgt, warped_tgt_flow).mean(1, keepdim=True)

        l1_3d = (img_tgt - warped_tgt_3d).abs().mean(1, keepdim=True)
        l1_flow = (img_tgt - warped_tgt_flow).abs().mean(1, keepdim=True)

        s_3d = r * (1 - ssim_3d) / 2 + (1 - r) * l1_3d
        s_flow = r * (1 - ssim_flow) / 2 + (1 - r) * l1_flow

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

    def fwd_bwd_flow_consistency(self, flow_fwd: torch.Tensor, flow_bwd: torch.Tensor, scale: int):
        self.__set_batch_size(flow_fwd.shape[0], flow_fwd)
        alpha = self.flow_cons_params[0]
        beta = self.flow_cons_params[1]

        # Warp
        bwd2fwd = F.grid_sample(flow_bwd, flow_fwd.permute(0, 2, 3, 1), padding_mode='border', align_corners=True)
        fwd2bwd = F.grid_sample(flow_fwd, flow_bwd.permute(0, 2, 3, 1), padding_mode='border', align_corners=True)

        # Consistency error
        diff_fwd = (bwd2fwd + flow_fwd).abs()
        diff_bwd = (fwd2bwd + flow_bwd).abs()

        # Condition
        bound_fwd = beta * (2 ** scale) * flow_fwd.norm(p=2, dim=1, keepdim=True)
        with torch.no_grad():
            bound_fwd = bound_fwd.clamp_min(alpha)

        bound_bwd = beta * (2 ** scale) * flow_bwd.norm(p=2, dim=1, keepdim=True)
        with torch.no_grad():
            bound_bwd = bound_bwd.clamp_min(alpha)

        # Mask
        noc_mask_src = ((2 ** scale) * diff_bwd.norm(p=2, dim=1, keepdim=True) < bound_bwd)
        noc_mask_tgt = ((2 ** scale) * diff_fwd.norm(p=2, dim=1, keepdim=True) < bound_fwd)

        # Consistency loss
        loss_fwd = (diff_fwd.mean(dim=1, keepdim=True) * noc_mask_tgt).sum() / noc_mask_tgt.sum()
        loss_bwd = (diff_bwd.mean(dim=1, keepdim=True) * noc_mask_src).sum() / noc_mask_src.sum()
        consistency_loss = (loss_fwd + loss_bwd) / 2
        return consistency_loss
