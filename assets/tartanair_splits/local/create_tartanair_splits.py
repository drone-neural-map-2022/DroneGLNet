
import os
from zipfile import ZipFile
import re
import numpy as np


OUTPUT_DIR = r"C:\Users\tormi\Documents\Egyetem\Research\Framework\monOdometry\monoSLAM\assets\tartanair_splits"
DATASET_DIR = r"C:\Users\tormi\Documents\Egyetem\Research\Data\TartanAIR"
ENV_MODES = ["Easy"]  #, "Hard"]
NUM_VAL_TEST_DIRS = [1, 2]
SIDES = ["left", "right"]
SHUFFLE = True
SEED = 42


def append_paths_srings(env_name, env_mode, seq_dirs, zip_namelist, path_strings):
    seq_image_side = zip_namelist[0].split('_')[-1].split('.')[0][0]
    for seq_dir in seq_dirs:
        seq_strings = []
        seq_images = [name for name in zip_namelist if seq_dir in name][1:]
        for seq_image in seq_images:
            seq_image_id = re.findall("\d\d\d\d\d\d", seq_image)
            if len(seq_image_id) > 0:
                seq_image_id = seq_image_id[0]
            else:
                continue
            seq_strings.append(
                "{} {} {}".format("{}_{}_{}".format(
                    env_mode, env_name, os.path.split(seq_dir)[-1]), seq_image_id, seq_image_side
                )
            )
        path_strings += seq_strings[:-1]


def write_paths(file_path, paths):
    with open(file_path, 'w') as txt_file:
        for path in paths:
            txt_file.write("{}\n".format(path))


if __name__ == "__main__":
    train_paths = []
    val_paths = []
    test_paths = []

    test_ids = []
    for env_name in os.listdir(DATASET_DIR):
        for env_mode in ENV_MODES:
            for side in SIDES:
                img_side = "image_{}".format(side)
                zip_path = ZipFile("{}.zip".format(os.path.join(DATASET_DIR, env_name, env_mode, img_side)), 'r')
                rel_path = os.path.join(zip_path.filename, env_name, env_name, env_mode)

                zip_namelist = zip_path.namelist()
                # zip_namelist = [name for name in zip_namelist if "000000" not in name]

                seq_dirs = [re.findall(".*P\d\d\d", name) for name in zip_namelist]
                seq_dirs = list(np.unique(np.array(seq_dirs)))

                train_dirs = seq_dirs[:-sum(NUM_VAL_TEST_DIRS)]
                val_dirs = seq_dirs[len(train_dirs):len(train_dirs) + NUM_VAL_TEST_DIRS[0]]
                test_dirs = seq_dirs[-NUM_VAL_TEST_DIRS[1]:]

                for test_dir in test_dirs:
                    test_ids.append((env_name, os.path.split(test_dir)[-1]))

                append_paths_srings(env_name, env_mode, train_dirs, zip_namelist, train_paths)
                append_paths_srings(env_name, env_mode, val_dirs, zip_namelist, val_paths)
                append_paths_srings(env_name, env_mode, test_dirs, zip_namelist, test_paths)

    if SHUFFLE:
        rnd = np.random.RandomState(SEED)
        rnd.shuffle(train_paths)
        rnd.shuffle(val_paths)

    if len(ENV_MODES) > 1:
        env_mode = "{}_{}".format(ENV_MODES[0], ENV_MODES[1])
        out_path = os.path.join(OUTPUT_DIR, env_mode)
        os.makedirs(out_path)

        write_paths(os.path.join(out_path, "train_files.txt"), train_paths)
        write_paths(os.path.join(out_path, "val_files.txt"), val_paths)

        for (test_env, test_seq) in test_ids:
            test_paths_per_seq = [test_path for test_path in test_paths
                                  if (test_env in test_path) and (test_seq in test_path)]
            write_paths(os.path.join(out_path, "test_{}_{}.txt".format(test_env, test_seq)), test_paths_per_seq)
    else:
        for env_mode in ENV_MODES:
            out_path = os.path.join(OUTPUT_DIR, env_mode)
            os.makedirs(out_path)

            write_paths(os.path.join(out_path, "train_files.txt"), train_paths)
            write_paths(os.path.join(out_path, "val_files.txt"), val_paths)

            for (test_env, test_seq) in test_ids:
                test_paths_per_seq = [test_path for test_path in test_paths
                                      if (test_env in test_path) and (test_seq in test_path)]
                write_paths(os.path.join(out_path, "test_files_{}_{}.txt".format(test_env, test_seq)),
                            test_paths_per_seq)

