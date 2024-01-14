#!/usr/bin/env python3.11

# System imports.
from typing import List

# 3rd party imports.
import torch


class BasicBlock(torch.nn.Module):
    """
    Basic block implementation.
    """

    def __init__(self, in_channels: int, downsample_block: bool) -> None:
        """
        Initialize basic block.

        Parameters
        ----------
        in_channels: int
            Number of input channels to the block.
        downsample_block: bool
            Whether or not the block downsamples resolution (generally first block in stage).

        Returns
        -------
        None
        """
        super().__init__()

        stride = 1 if not downsample_block else 2
        out_channels = in_channels if not downsample_block else in_channels * 2
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding="same",
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)

        self.skip = (
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )
            if downsample_block
            else torch.nn.Identity()
        )

        self.relu2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute the forward pass for a Basic ResNet Block.

        Parameters
        ----------
        x: torch.Tensor
            Input batch.

        Returns
        -------
        y: torch.Tensor
            Output batch.
        """
        x_skip = x

        # First conv.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Second conv.
        x = self.conv2(x)
        x = self.bn2(x)

        # Skip connection.
        x += self.skip(x_skip)

        # Post ReLU.
        x = self.relu2(x)

        return x


class ResNet(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, stages: List[int], x_size) -> None:
        """
        TODO
        """
        super().__init__()

        # Assertions.
        assert len(stages) == 4

        # Params.
        num_channels = 64

        # Initial ops.
        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=num_channels, kernel_size=7, padding="same"
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=num_channels)
        self.relu1 = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stages.
        self.stage1 = self._make_stage(stages[0], num_channels * 2**0)
        self.stage2 = self._make_stage(stages[1], num_channels * 2**1)
        self.stage3 = self._make_stage(stages[2], num_channels * 2**2)
        self.stage4 = self._make_stage(stages[3], num_channels * 2**3)

        # Final ops.
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
        in_features = (num_channels * 16) * round(x_size / 16) * round(x_size / 16)
        # import pdb; pdb.set_trace()
        self.fc = torch.nn.Linear(in_features=in_features, out_features=10)
        self.softmax = torch.nn.Softmax(dim=1)

    def _make_stage(self, num_blocks: int, in_channels: int) -> torch.nn.Sequential:
        """
        TODO
        """
        blocks = []
        for block_idx in range(num_blocks):
            is_downsample = block_idx == 0
            num_channels = in_channels if is_downsample else in_channels * 2
            block = BasicBlock(num_channels, is_downsample)
            blocks.append(block)
        return torch.nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO
        """
        # Initial ops.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Stages.
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        # Final ops.
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        # x = self.softmax(x)

        return x


if __name__ == "__main__":
    # Test Basic Block
    x = torch.rand(2, 64, 128, 128)
    basic_block_downsample = BasicBlock(64, True)
    y = basic_block_downsample(x)
    assert y.shape == (2, 128, 64, 64)

    x = torch.rand(2, 64, 64, 64)
    basic_block = BasicBlock(64, False)
    y = basic_block(x)
    assert y.shape == (2, 64, 64, 64)

    # Test ResNet18
    resnet18_config = [2, 2, 2, 2]
    resnet18 = ResNet(resnet18_config)
    x = torch.rand(2, 1, 28, 28)
    y = resnet18(x)
    assert y.shape == (2, 10)
