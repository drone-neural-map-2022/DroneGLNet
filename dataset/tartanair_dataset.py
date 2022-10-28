
import os
import numpy as np
import PIL.Image as pil
from zipfile import ZipFile
import re

from .mono_dataset import MonoDataset


def pil_zip_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    pattern_dir = re.compile(".*\.zip")
    match_dir = re.match(pattern_dir, path)
    img_path = path.split(".zip")[-1]
    img_path = img_path.replace("\\", "/")[1:]
    with ZipFile(match_dir[0], 'r') as zip_dir:
        with zip_dir.open(img_path.replace("\\", "/")) as f:
            with pil.open(f) as img:
                return img.convert('RGB')


class TartanAirDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(TartanAirDataset, self).__init__(*args, **kwargs)
        self.og_shape = (480, 640)  # (height, width)
        self.loader = pil_zip_loader
        # From https://arxiv.org/pdf/2011.00359.pdf
        fx = 320 / self.og_shape[1]
        fy = 320 / self.og_shape[0]
        self.K = np.array([
            [fx, 0., 0.5, 0.],
            [0., fy, 0.5, 0.],
            [0., 0., 1.0, 0.],
            [0., 0., 0.0, 1.],
        ], dtype=np.float32)
        self.fov = 90
        self.side_map = {"l": "left", "r": "right"}

    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        mode, env, seq = folder.split('_')
        side = self.side_map[side]
        f_str = "{:06d}_{}{}".format(frame_index, side, self.img_ext)
        img_folder = "image_{}".format(side)
        image_path = os.path.join(
            self.data_path,
            env, mode, "{}.zip".format(img_folder), env, env, mode, seq, img_folder,
            f_str)
        return image_path
