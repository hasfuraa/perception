import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from modules.resnet import ResNet
from modules.lightning.runner import Dataloader, LightningModule, RunnerConfig


def train() -> None:
    model = LightningModule()

    trainer = pl.Trainer(
        max_epochs=20,
        limit_val_batches=128,
        accelerator="cpu",
        logger=pl.loggers.TensorBoardLogger("training/logs"),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val_loss"),
        ],
        enable_progress_bar=True,
        enable_model_summary=True,
        # # Debug params.
        # limit_train_batches=1,
        # log_every_n_steps=1,
    )

    dataloader = Dataloader(batch_size=16)
    trainer.fit(
        model,
        train_dataloaders=dataloader.train_dataloader(),
        val_dataloaders=dataloader.val_dataloader(),
    )


if __name__ == "__main__":
    train()
