
import os

import torch.cuda
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset.tartanair_dataset import TartanAirDataset
from dataset.tartanair_dataset_azure import TartanAirAzureDataset
from models.glnet.glnet import GLNet
from configs import DATA_CFG, GLNET_CFG


def main():
    # Dataloading
    train_loader, val_loader = setup_dataloaders()

    model = GLNet(**GLNET_CFG)
    trainer = pl.Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1,
                         profiler='simple',
                         # check_val_every_n_epoch=None,
                         val_check_interval=1000,
                         limit_val_batches=0.1,
                         log_every_n_steps=50)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def setup_dataloaders():
    train_file_names = readlines(DATA_CFG['train_txt'])
    val_file_names = readlines(DATA_CFG['val_txt'])

    dataset = TartanAirDataset if not DATA_CFG['online'] else TartanAirAzureDataset

    train_set = dataset(
        data_path=DATA_CFG['set_root'],
        filenames=train_file_names,
        height=DATA_CFG['img_size'][0],
        width=DATA_CFG['img_size'][1],
        frame_idxs=DATA_CFG['frame_idxs'],
        num_scales=DATA_CFG['scales'],
        is_train=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=DATA_CFG['batch_size'],
        shuffle=False,
        num_workers=DATA_CFG['num_workers'],
        pin_memory=True
    )

    val_set = dataset(
        data_path=DATA_CFG['set_root'],
        filenames=val_file_names,
        height=DATA_CFG['img_size'][0],
        width=DATA_CFG['img_size'][1],
        frame_idxs=DATA_CFG['frame_idxs'],
        num_scales=DATA_CFG['scales'],
        is_train=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=DATA_CFG['batch_size'],
        shuffle=True,
        num_workers=DATA_CFG['num_workers'],
        pin_memory=True
    )
    return train_loader, val_loader


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


if __name__ == '__main__':
    main()
