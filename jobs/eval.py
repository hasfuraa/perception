import argparse
import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from modules.resnet import ResNet
from modules.lightning.runner import Dataloader, LightningModule
import os

DEFAULT_CKPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../training/logs/lightning_logs/version_62/checkpoints/epoch=8-step=33750.ckpt",
)


def evaluate(args) -> None:
    assert os.path.exists(
        args.ckpt_path
    ), f"checkpoint path {args.ckpt_path} does not exist."
    model = LightningModule.load_from_checkpoint(args.ckpt_path)

    dataloader = Dataloader(batch_size=16)
    test_dataloader = dataloader.test_dataloader()

    trainer = pl.Trainer(logger=pl.loggers.TensorBoardLogger("training/logs"))
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model.")
    parser.add_argument(
        "--ckpt-path",
        "-c",
        type=str,
        default=DEFAULT_CKPT_PATH,
        help="Absolute path to checkpoint used for evaluation ",
    )
    args = parser.parse_args()

    evaluate(args)
