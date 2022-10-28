
import os


DATA_CFG = {
    'batch_size': 1,
    'img_size': (480, 640),
    'frame_idxs': [-1, 0, 1, 2],
    'scales': 4,

    'set_root': r'C:\Users\tormi\Documents\Egyetem\Research\Data\TartanAIR',
    'train_txt': os.path.join(os.getcwd(), "assets", "tartanair_splits", "Easy", "train_files.txt"),
    'val_txt': os.path.join(os.getcwd(), "assets", "tartanair_splits", "Easy", "val_files.txt"),
    'test_txt': os.path.join(os.getcwd(), "assets", "tartanair_splits", "Easy", "test_files_abandonedfactory_P010.txt"),
    'num_workers': 1
}


GLNET_LOSS_CFG = {
    'img_size': DATA_CFG['img_size'],
    'scales': DATA_CFG['scales'],
    'mvs_weight': 1.,
    'epi_weight': 1.,
    'apc_weight': 1.,
    'disp_smooth': 0.5,
    'flow_smooth': 0.2,
    'flow_cons_params': (3.0, 0.05),
    'flow_cons_weight': 1.,
    'ssim_r': 0.85,
    'reduction': 'mean'
}


GLNET_CFG = {
    'pose_input_num': 2,
    'pose_output_num': 1,
    'pred_intrinsics': False,
    'resnet_pretrained': True,

    'depth_input_num': 3,
    'depth_output_num': 1,
    'depth_res_layers': 18,
    'depth_use_skips': True,

    'flow_input_num': 2,
    'flow_output_num': 1,
    'flow_res_layers': 18,
    'flow_use_skips': True,

    'shared_resnet': False,
    'frame_ids': DATA_CFG['frame_idxs'],
    'scales': DATA_CFG['scales'],
    'depth_limits': (0.1, 100.0),
    'loss_parameters': GLNET_LOSS_CFG
}