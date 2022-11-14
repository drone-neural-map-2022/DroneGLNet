import numpy as np
import torch
from itertools import combinations

from models.glnet.modules.glnet_loss import GLNetLoss
from utils.transformations import transformation_from_parameters, disp2depth
from models.glnet.modules.networks.pose_cnn import PoseCNN
from models.glnet.modules.networks.resnet_encoder import ResnetEncoder
from models.glnet.modules.networks.depth_decoder import DepthDecoder
from models.glnet.modules.networks.flow_decoder import FlowDecoder


class GLNet:
    def __init__(
            self,
            pose_input_num: int,
            pose_output_num: int,
            depth_input_num: int,
            depth_output_num: int,
            flow_input_num: int,
            flow_output_num: int,
            loss_parameters: dict,
            pred_intrinsics: bool = False,
            depth_res_layers: int = 18,
            depth_use_skips: bool = True,
            flow_res_layers: int = 18,
            flow_use_skips: int = True,
            resnet_pretrained: bool = True,
            shared_resnet: bool = True,
            depth_limits: tuple = (0.1, 100.),
            frame_ids: list = None,
            scales: int = 4
    ):
        self.shared_resnet = shared_resnet
        self.intrinsics = pred_intrinsics

        self.pose_input_num = pose_input_num
        self.depth_input_num = depth_input_num
        self.flow_input_num = flow_input_num

        self.scales = scales

        self.frame_ids = np.sort(frame_ids) if frame_ids is not None else [0, 1]
        self.depth_limits = depth_limits

        if shared_resnet and \
                (depth_input_num != flow_input_num or depth_res_layers != flow_res_layers):
            raise AttributeError("In case of shared resnet the flow and depth input nums must match, "
                                 "as well as the num of res layers!")

        self.camera_net = PoseCNN(pose_input_num, pred_intrinsics, pose_output_num)

        self.depth_encoder = ResnetEncoder(depth_res_layers, resnet_pretrained, depth_input_num)
        self.depth_decoder = DepthDecoder(self.depth_encoder.num_ch_enc, range(scales),
                                          depth_output_num, depth_use_skips)

        if self.shared_resnet:
            self.flow_encoder = self.depth_encoder
        else:
            self.flow_encoder = ResnetEncoder(flow_res_layers, resnet_pretrained, flow_input_num)

        self.flow_decoder = FlowDecoder(self.flow_encoder.num_ch_enc, range(scales),
                                        flow_output_num, flow_use_skips)

        self.glnet_loss = GLNetLoss(**loss_parameters)
        parameter_dict_list = [
            {'params': self.depth_decoder.parameters()},
            {'params': self.depth_encoder.parameters()},
            {'params': self.camera_net.parameters()},
            {'params': self.flow_decoder.parameters()},
        ]
        if not self.shared_resnet:
            parameter_dict_list.append({'params': self.flow_encoder.parameters()})
        self.optimizer = torch.optim.Adam(parameter_dict_list, lr=2e-4, betas=(0.9, 0.999))

    def forward(self, input: dict) -> dict:
        raise NotImplementedError

    def training_step(self, batch: dict):
        self.train()

        all_color_aug = torch.stack([batch[('color_aug', i, 0)] for i in self.frame_ids], dim=1)
        disps, poses, flows_fwd, flows_bwd = self.__predict_for_train_val(all_color_aug)

        # Convert disparities to depths
        depths = self.__disps2depths(disps)

        loss, loss_parts = self.glnet_loss(batch, depths, poses, flows_fwd, self.scales, disps, flows_bwd)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_parts['total_loss'] = loss
        return loss_parts

    def validation_step(self, batch: dict):
        self.eval()

        all_color = torch.stack([batch[('color', i, 0)] for i in self.frame_ids], dim=1)
        disps, poses, flows_fwd, flows_bwd = self.__predict_for_train_val(all_color)

        # Convert disparities to depths
        depths = self.__disps2depths(disps)

        loss, loss_parts = self.glnet_loss(batch, depths, poses, flows_fwd, self.scales, disps, flows_bwd)
        loss_parts['total_loss'] = loss
        return loss_parts

    def __disps2depths(self, disps):
        depths = {}
        for frame, disp in disps.items():
            depth_dict = {}
            for scale in range(self.scales):
                depth_dict[('depth', scale)] = \
                    disp2depth(disp[('disp', scale)], self.depth_limits[0], self.depth_limits[1])[1]
            depths[frame] = depth_dict
        return depths

    def __predict_for_train_val(self, image_stack: torch.Tensor, is_train: bool = True):
        inp_shape = image_stack.shape
        key_features = []

        disps = {}
        poses = {}
        flows_fwd = {}
        flows_bwd = {}

        # Depth estimation
        for f_idx in range(len(self.frame_ids) - self.depth_input_num + 1):
            frame_ids = self.frame_ids[f_idx:f_idx + self.depth_input_num]
            key_features.append(f_idx + (self.depth_input_num - 1) // 2)
            disps[frame_ids[(self.depth_input_num - 1) // 2]] = \
                self.depth_decoder(
                    self.depth_encoder(
                        image_stack[:, f_idx:f_idx + self.depth_input_num]
                        .view(inp_shape[0], -1, *inp_shape[-2:]).contiguous(),
                        norm_input=True
                    )
                )

        # Pose estimation
        for feature_group in combinations(key_features, self.pose_input_num):
            fg_list = list(feature_group)
            frame_group = self.frame_ids[fg_list]

            poses[tuple(frame_group)] = self.camera_net(
                image_stack[:, fg_list]
                .view(inp_shape[0], -1, *inp_shape[-2:]).contiguous()
            )

        # Forward and Backward Flow estimation
        for feature_group in combinations(key_features, self.flow_input_num):
            fg_list = list(feature_group)
            frame_group = self.frame_ids[fg_list]

            flows_fwd[tuple(frame_group)] = \
                self.flow_decoder(
                    self.flow_encoder(
                        image_stack[:, fg_list]
                        .view(inp_shape[0], -1, *inp_shape[-2:]).contiguous(),
                        norm_input=True
                    )
                )

            if is_train:
                flows_bwd[tuple(frame_group)] = \
                    self.flow_decoder(
                        self.flow_encoder(
                            image_stack[:, fg_list[::-1]]
                            .view(inp_shape[0], -1, *inp_shape[-2:]).contiguous(),
                            norm_input=True
                        )
                    )

        return disps, poses, flows_fwd, flows_bwd
    
    def train(self):
        self.depth_decoder.train()
        self.depth_encoder.train()
        self.camera_net.train()
        self.flow_decoder.train()
        if not self.shared_resnet:
            self.flow_encoder.train()
            
    def eval(self):
        self.depth_decoder.eval()
        self.depth_encoder.eval()
        self.camera_net.eval()
        self.flow_decoder.eval()
        if not self.shared_resnet:
            self.flow_encoder.eval()
            
    def cuda(self):
        self.depth_decoder.cuda()
        self.depth_encoder.cuda()
        self.camera_net.cuda()
        self.flow_decoder.cuda()
        if not self.shared_resnet:
            self.flow_encoder.cuda()
            
    def cpu(self):
        self.depth_decoder.cpu()
        self.depth_encoder.cpu()
        self.camera_net.cpu()
        self.flow_decoder.cpu()
        if not self.shared_resnet:
            self.flow_encoder.cpu()
