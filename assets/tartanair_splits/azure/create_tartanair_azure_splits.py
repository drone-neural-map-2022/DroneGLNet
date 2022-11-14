from azure.storage.blob import ContainerClient
import re
import os
import time
import numpy as np


WRK_PTH = r'C:\Users\tormi\Documents\Egyetem\Research\Frameworks\glnet\DroneGLNet\assets\tartanair_splits\azure\Easy'
ENV_MODES = ['Easy']
SIDES = ['Left', 'Right']
NUM_VAL_TRAJECTS = 2
SHUFFLE_TRAJ_DIRS = False
SHUFFLE_IMAGES = False
FRAME_LIMITS = [-1, 2]


# Dataset website: http://theairlab.org/tartanair-dataset/
account_url = 'https://tartanair.blob.core.windows.net/'
container_name = 'tartanair-release1'
container_client = None


def get_environment_list():
    '''
    List all the environments shown in the root directory
    '''
    global container_client
    env_gen = container_client.walk_blobs()
    envlist = []
    for env in env_gen:
        envlist.append(env.name)
    return envlist


def get_trajectory_list(envname, easy_hard='Easy'):
    '''
    List all the trajectory folders, which is named as 'P0XX'
    '''
    assert (easy_hard == 'Easy' or easy_hard == 'Hard')
    global container_client
    traj_gen = container_client.walk_blobs(name_starts_with=envname + '/' + easy_hard + '/')
    trajlist = []
    for traj in traj_gen:
        trajname = traj.name
        trajname_split = trajname.split('/')
        trajname_split = [tt for tt in trajname_split if len(tt) > 0]
        if trajname_split[-1][0] == 'P':
            trajlist.append(trajname)
    return trajlist


def _list_blobs_in_folder(folder_name):
    """
    List all blobs in a virtual folder in an Azure blob container
    """
    global container_client
    files = []
    generator = container_client.list_blobs(name_starts_with=folder_name)
    for blob in generator:
        files.append(blob.name)
    return files


def get_image_list(trajdir, left_right='left'):
    assert (left_right == 'left' or left_right == 'right')
    files = _list_blobs_in_folder(trajdir + '/image_' + left_right + '/')
    files = [fn for fn in files if fn.endswith('.png')]
    return files


def append_paths_srings(env_name, env_mode, traj_dir_list, path_list):
    for traj_dir in traj_dir_list:
        traj_name = traj_dir.split('/')[-2]
        img_pth_list = get_image_list(traj_dir)
        img_pth_list = img_pth_list[abs(FRAME_LIMITS[0]):-FRAME_LIMITS[1]]

        for img_pth in img_pth_list:
            frame_id = re.findall('\d\d\d\d\d\d', img_pth)

            if len(frame_id) > 0:
                frame_id = frame_id[0]
            else:
                continue

            if 'right' in img_pth:
                side = 'r'
            elif 'left' in img_pth:
                side = 'l'
            else:
                raise AttributeError

            path_list += [
                "{} {} {}".format("{}_{}_{}".format(
                    env_mode, env_name, traj_name), frame_id, side
                )
            ]


def write_paths(file_path, paths):
    with open(file_path, 'w') as txt_file:
        for path in paths:
            txt_file.write("{}\n".format(path))


def create_blob_splits():
    envlist = get_environment_list()

    train_paths = []
    val_paths = []

    for env_name in envlist:
        print(f'Collecting environment {env_name}')
        for mode in ENV_MODES:
            env_trajects = get_trajectory_list(env_name, easy_hard=mode)

            if SHUFFLE_TRAJ_DIRS:
                rng = np.random.default_rng(42)
                rng.shuffle(env_trajects)

            train_trajects = env_trajects[:-NUM_VAL_TRAJECTS]
            val_trajects = env_trajects[-NUM_VAL_TRAJECTS:]

            append_paths_srings(env_name[:-1], mode, train_trajects, train_paths)
            append_paths_srings(env_name[:-1], mode, val_trajects, val_paths)

    if SHUFFLE_IMAGES:
        rng = np.random.default_rng(42)
        rng.shuffle(train_paths)

    write_paths(os.path.join(WRK_PTH, 'train_files.txt'), train_paths)
    write_paths(os.path.join(WRK_PTH, 'val_files.txt'), val_paths)


if __name__ == '__main__':
    print('Connect Azure...')
    container_client = ContainerClient(account_url=account_url,
                                       container_name=container_name,
                                       credential=None)
    print('Collect files...')
    start = time.time()
    create_blob_splits()
    print(f'Elapsed time: {time.time() - start} secs')
