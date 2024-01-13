import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Block1
        self.conv11 = torch.nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1,
        )
        self.relu11 = torch.nn.ReLU()
        self.conv12 = torch.nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1,
        )
        self.relu12 = torch.nn.ReLU()
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)

        # Block2
        self.conv21 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1,
        )
        self.relu21 = torch.nn.ReLU()
        self.conv22 = torch.nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
        )
        self.relu22 = torch.nn.ReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)

        # Head
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(in_features=3136, out_features=10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x: torch.Tensor):
        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.max_pool1(x)

        x = self.conv21(x)
        x = self.relu21(x)
        x = self.conv22(x)
        x = self.relu22(x)
        x = self.max_pool2(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.softmax(x)

        return x


class LightningModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = CNN()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        output = self(inputs)
        loss = torch.nn.functional.nll_loss(output, targets.view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)


class Dataloader(pl.LightningDataModule):
    def __init__(self, batch_size: int = 1) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.train_dataset = datasets.FashionMNIST(
            root="data", train=True, download=True, transform=ToTensor()
        )

        self.val_dataset = datasets.FashionMNIST(
            root="data", train=False, download=True, transform=ToTensor()
        )

        self.test_dataset = datasets.FashionMNIST(
            root="data", train=False, download=True, transform=ToTensor()
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def main() -> None:
    model = LightningModule()  # Replace with your own model

    trainer = pl.Trainer(
        # max_epochs=10,
        val_check_interval=1000,
        max_steps=10000,
        limit_val_batches=100,
        accelerator="cpu",
        logger=pl.loggers.TensorBoardLogger("training/logs"),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=3),
        ],
    )

    dataloader = Dataloader()
    trainer.fit(
        model,
        train_dataloaders=dataloader.train_dataloader(),
        val_dataloaders=dataloader.val_dataloader(),
    )


if __name__ == "__main__":
    main()
