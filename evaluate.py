
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from dataset.tartanair_dataset import TartanAirDataset
from configs import DATA_CFG


def evaluate():
    test_file_names = readlines(DATA_CFG['test_txt'])
    test_set = TartanAirDataset(
        data_path=DATA_CFG['set_root'],
        filenames=test_file_names,
        height=DATA_CFG['img_size'][0],
        width=DATA_CFG['img_size'][1],
        frame_idxs=DATA_CFG['frame_idxs'],
        num_scales=DATA_CFG['scales'],
        is_train=False
    )
    pose_file = readlines(r"C:\Users\tormi\Documents\Egyetem\Research\Data\TartanAIR\abandonedfactory\Easy\pose_left.txt")
    poses_np = np.loadtxt(pose_file)
    traj = shift0(poses_np)

    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(projection='3d')
    ax.set_xlim(-40, 15)
    ax.set_ylim(-40, 15)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for b_idx, batch in enumerate(test_set):
        ax.plot(traj[:b_idx, 0], traj[:b_idx, 1], traj[:b_idx, 2])
        plt.pause(0.001)
        img_np = batch[('color', 0, 0)].permute(1, 2, 0).numpy()
        img_np = np.asarray(img_np * 255, dtype=np.uint8)
        cv2.imshow("Drone", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    plt.show()


def SO2quat(SO_data):
    rr = R.from_matrix(SO_data)
    return rr.as_quat()


def pos_quats2SE_matrices(quat_datas):
    data_len = quat_datas.shape[0]
    SEs = []
    for quat in quat_datas:
        SO = R.from_quat(quat[3:7]).as_matrix()
        SE = np.eye(4)
        SE[0:3,0:3] = SO
        SE[0:3,3]   = quat[0:3]
        SEs.append(SE)
    return SEs


def SE2pos_quat(SE_data):
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat


def shift0(traj):
    '''
    Traj: a list of [t + quat]
    Return: translate and rotate the traj
    '''
    traj_ses = pos_quats2SE_matrices(np.array(traj))
    traj_init = traj_ses[0]
    traj_init_inv = np.linalg.inv(traj_init)
    new_traj = []
    for tt in traj_ses:
        ttt=traj_init_inv.dot(tt)
        new_traj.append(SE2pos_quat(ttt))
    return np.array(new_traj)


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


if __name__ == '__main__':
    evaluate()
