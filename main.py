
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from dataset.tartanair_dataset import TartanAirDataset
from dataset.tartanair_dataset_azure import TartanAirAzureDataset
from models.glnet.glnet import GLNet
from configs import DATA_CFG, GLNET_CFG, TRAIN_CFG, ON_GPU


def main():
    log_dir = os.path.join(os.getcwd(), 'logs', datetime.now().strftime('%m%d%Y_%H%M%S'))
    os.makedirs(log_dir)
    logger = SummaryWriter(log_dir)

    # Dataloading
    train_loader, val_loader = setup_dataloaders()

    model = GLNet(**GLNET_CFG)
    if ON_GPU:
        model.cuda()

    train(model, train_loader, TRAIN_CFG['epochs'], val_loader, logger, ON_GPU)


def train(model, train_loader, epochs, val_loader=None, logger=None, on_gpu=True):
    for epoch in range(epochs):
        epoch_train_loss = 0.
        with tqdm(total=len(train_loader), leave=True) as pbar:
            for bidx, batch in enumerate(train_loader):
                if on_gpu:
                    for key, value in batch.items():
                        batch[key] = value.cuda()

                losses = model.training_step(batch)

                epoch_train_loss += losses['total_loss'].item()
                logger.add_scalars('Train/Losses', losses, epoch * len(train_loader) + bidx)
                pbar.set_postfix({
                    'Epoch': f'{epoch+1}/{epochs}',
                    'Mode': 'train',
                    'AvgLoss': epoch_train_loss / (bidx + 1)
                })
                pbar.update(1)

        if val_loader is not None:
            epoch_val_loss = 0.
            with tqdm(total=len(val_loader), leave=True) as pbar:
                for bidx, batch in enumerate(val_loader):
                    if on_gpu:
                        for key, value in batch.items():
                            batch[key] = value.cuda()

                    losses = model.validation_step(batch)

                    logger.add_scalars('Validation/Losses', losses, epoch * len(val_loader) + bidx)
                    epoch_val_loss += losses['total_loss'].item()
                    pbar.set_postfix({
                        'Epoch': f'{epoch + 1}/{epochs}',
                        'Mode': 'validation',
                        'AvgLoss': epoch_val_loss / (bidx + 1)
                    })
                    pbar.update(1)


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
        shuffle=False,
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
