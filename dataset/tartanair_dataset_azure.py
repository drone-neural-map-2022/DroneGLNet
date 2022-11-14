
import cv2
import PIL.Image as pil
import numpy as np
import io
import os
from azure.storage.blob import ContainerClient

from .tartanair_dataset import TartanAirDataset


class TartanAirAzureDataset(TartanAirDataset):
    def __init__(self, *args, **kwargs):
        super(TartanAirAzureDataset, self).__init__(*args, **kwargs)

        # Dataset website: http://theairlab.org/tartanair-dataset/
        account_url = 'https://tartanair.blob.core.windows.net/'
        container_name = 'tartanair-release1'
        self.container_client = ContainerClient(account_url=account_url,
                                                container_name=container_name,
                                                credential=None)

    def get_color(self, folder, frame_index, side, do_flip):
        image_file = self.get_image_path(folder, frame_index, side)
        bc = self.container_client.get_blob_client(blob=image_file)
        data = bc.download_blob()
        ee = io.BytesIO(data.content_as_bytes())
        img = cv2.imdecode(np.asarray(bytearray(ee.read()),dtype=np.uint8), cv2.IMREAD_COLOR)
        color = img[:, :, [2, 1, 0]] # BGR2RGB
        color = pil.fromarray(color, "RGB")

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        mode, env, traj = folder.split('_')
        side = self.side_map[side]
        img_folder = "image_{}".format(side)
        f_str = "{:06d}_{}{}".format(frame_index, side, self.img_ext)
        image_path = os.path.join(env, mode, traj, img_folder, f_str)
        return image_path

    def check_depth(self):
        return None

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
