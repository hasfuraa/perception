#!/usr/bin/env python3.11

# System imports.
from typing import List

# 3rd party imports.
import torch
from dataclasses import dataclass


@dataclass
class DETRConfig:
    x_size: int = 32


class Tokenizer(torch.nn.Module):
    def __init__(self) -> None:
        """
        TODO
        """
        super().__init__()

        self.num_channels = 32

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=self.num_channels, kernel_size=3, padding="same"
        )
        self.bn1 = torch.nn.BatchNorm2d(num_features=self.num_channels)
        self.relu1 = torch.nn.ReLU()
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = torch.nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.num_channels * 2,
            kernel_size=3,
            padding="same",
        )
        self.bn2 = torch.nn.BatchNorm2d(num_features=self.num_channels * 2)
        self.relu2 = torch.nn.ReLU()
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = torch.nn.Conv2d(
            in_channels=self.num_channels * 2,
            out_channels=self.num_channels * 4,
            kernel_size=3,
            padding="same",
        )
        self.bn3 = torch.nn.BatchNorm2d(num_features=self.num_channels * 4)
        self.relu3 = torch.nn.ReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = x.reshape(x.shape[0], self.num_channels * 4, -1).permute(0, 2, 1)
        return x


class PositionalEncoding(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, d_model: int, max_tokens: int) -> None:
        """
        TODO
        """
        super().__init__()

        self.encoding = torch.zeros(max_tokens, d_model)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_tokens)
        pos = pos.float().unsqueeze(dim=1)

        denom = torch.arange(0, d_model, step=2).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (denom / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (denom / d_model)))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        TODO
        """
        batch_size, num_tokens, d_model = x.size()
        return x + self.encoding[:num_tokens, :]


class Encoder(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, d_model: int, max_tokens: int) -> None:
        """
        TODO
        """
        super().__init__()

        num_heads = 8
        hidden_dim = 256

        self.msa = torch.nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.msa_norm = torch.nn.LayerNorm((max_tokens, d_model))

        self.linear1 = torch.nn.Linear(in_features=d_model, out_features=hidden_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=hidden_dim, out_features=d_model)
        self.linear_norm = torch.nn.LayerNorm((max_tokens, d_model))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        TODO
        """
        _x = x
        x, _ = self.msa(x, x, x)
        x = self.msa_norm(x + _x)

        _x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.linear_norm(x + _x)

        return x


class Decoder(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, d_model: int, max_tokens: int) -> None:
        """
        TODO
        """
        super().__init__()

        num_heads = 1
        d_embedding = 128
        hidden_dim = 256

        self.q = torch.nn.Embedding(10, d_embedding)

        self.self_attention = torch.nn.MultiheadAttention(
            d_embedding, num_heads, batch_first=True
        )
        self.sa_norm = torch.nn.LayerNorm((10, d_embedding))

        self.cross_attention = torch.nn.MultiheadAttention(
            d_embedding, num_heads, batch_first=True
        )
        self.ca_norm = torch.nn.LayerNorm((10, d_model))

        self.linear1 = torch.nn.Linear(in_features=d_embedding, out_features=hidden_dim)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(in_features=hidden_dim, out_features=d_embedding)
        self.linear_norm = torch.nn.LayerNorm((10, d_embedding))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        TODO
        """
        # Self attention.
        _q = self.q.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1)
        q, _ = self.self_attention(self.q.weight, self.q.weight, self.q.weight)
        q = self.sa_norm(q + _q)

        # Cross attention.
        _q = q
        x, _ = self.cross_attention(q, x, x)
        x = self.ca_norm(x + _q)

        # Feed forward.
        _x = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.linear_norm(x + _x)

        return x


class Head(torch.nn.Module):
    def __init__(self) -> None:
        """
        TODO
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features=128, out_features=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        TODO
        """
        x = self.linear(x).squeeze()
        return x


class DETR(torch.nn.Module):
    """
    TODO
    """

    def __init__(self, model_config: DETRConfig) -> None:
        """
        TODO
        """
        super().__init__()
        self.tokenizer = Tokenizer()
        self.positional_encoding = PositionalEncoding(d_model=128, max_tokens=64)
        self.encoder = Encoder(d_model=128, max_tokens=64)
        self.decoder = Decoder(d_model=128, max_tokens=64)
        self.head = Head()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TODO
        """
        x = self.tokenizer(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x


if __name__ == "__main__":
    # Test tokenizer.
    tokenizer_x = torch.rand(2, 1, 32, 32)
    tokenizer = Tokenizer()
    tokenizer_y = tokenizer(tokenizer_x)
    assert tokenizer_y.shape == (2, 64, 128)
    print("Tokenizer passes.")

    # Test positional encoding.
    pe_x = torch.rand(2, 64, 128)
    pe = PositionalEncoding(d_model=128, max_tokens=64)
    pe_y = pe(pe_x)
    assert pe_y.shape == (2, 64, 128)
    print("PositionalEncoding passes.")

    # Test encoder.
    enc_x = torch.rand(2, 64, 128)
    enc = Encoder(d_model=128, max_tokens=64)
    enc_y = enc(enc_x)
    assert enc_y.shape == (2, 64, 128)
    print("Encoder passes.")

    # Test decoder.
    dec_x = torch.rand(2, 64, 128)
    dec = Decoder(d_model=128, max_tokens=64)
    dec_y = dec(dec_x)
    assert dec_y.shape == (2, 10, 128)
    print("Decoder passes.")

    # Test head.
    head_x = torch.rand(2, 10, 128)
    head = Head()
    head_y = head(head_x)
    assert head_y.shape == (2, 10)
    print("Head passes.")

    # Test detr.
    detr_x = torch.rand(2, 1, 32, 32)
    detr = DETR(DETRConfig())
    detr_y = detr(detr_x)
    assert detr_y.shape == (2, 10)
    print("DETR passes.")
