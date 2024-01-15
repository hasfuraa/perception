import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from modules.resnet import ResNet
from jobs.main import Dataloader, LightningModule
import os

CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../training/logs/lightning_logs/version_62/checkpoints/epoch=8-step=33750.ckpt",
)


def main() -> None:
    assert os.path.exists(CKPT_PATH), f"checkpoint path {CKPT_PATH} does not exist."
    model = LightningModule.load_from_checkpoint(CKPT_PATH)

    dataloader = Dataloader(batch_size=16)
    test_dataloader = dataloader.test_dataloader()

    trainer = pl.Trainer(logger=pl.loggers.TensorBoardLogger("training/logs"))
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
