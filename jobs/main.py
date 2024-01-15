import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from modules.resnet import ResNet
from modules.detr import DETR


class LightningModule(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        # self.model = ResNet(stages=[2, 2, 2, 2], x_size=32)
        self.model = DETR()
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        self.reshaped_inputs = None
        self.metrics = []

    def forward(self, inputs):
        return self.model(inputs)

    def execute(self, batch):
        inputs, targets = batch
        self.reshaped_inputs = torch.zeros(
            (inputs.shape[0], inputs.shape[1], 32, 32),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        self.reshaped_inputs[:, :, 2:30, 2:30] = inputs
        logits = self(self.reshaped_inputs)
        assert logits.shape == (len(targets), 10)
        one_hot_targets = torch.nn.functional.one_hot(targets, num_classes=10).to(
            torch.float32
        )
        loss = self.loss(logits, one_hot_targets)

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self.execute(batch)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def on_train_epoch_end(self) -> None:
        assert self.reshaped_inputs is not None
        grid = make_grid(self.reshaped_inputs)
        self.logger.experiment.add_image("training_images", grid, self.current_epoch)
        print("")

    def validation_step(self, batch, batch_idx):
        loss, _ = self.execute(batch)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def evaluate(self, logits, targets):
        outputs = torch.softmax(logits, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        assert len(predictions) == len(targets)
        assert len(predictions) != 0
        accuracy = (predictions == targets).sum() / len(logits)
        return accuracy

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        loss, logits = self.execute(batch)
        accuracy = self.evaluate(logits, targets)
        self.metrics.append({"test_loss": loss, "accuracy": accuracy})
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        avg_loss = torch.stack([x["test_loss"] for x in self.metrics]).mean()
        avg_accuracy = torch.stack([x["accuracy"] for x in self.metrics]).mean()
        self.log("avg_test_loss", avg_loss)
        self.log("avg_accuracy", avg_accuracy)

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3)


class Dataloader(pl.LightningDataModule):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size

        self.train_dataset = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
        )

        self.val_dataset = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

        self.test_dataset = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=7,
            persistent_workers=True,
        )
