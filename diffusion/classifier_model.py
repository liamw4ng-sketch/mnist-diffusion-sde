
import torch
import torch.nn as nn
import numpy as np


class GaussianRandomFourierFeatures(nn.Module):
    """Gaussian random Fourier features for encoding time steps."""
    def __init__(self, embed_dim: int, scale: float = 30.0):
        super().__init__()
        self.rff_weights = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = x[:, None] * self.rff_weights[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)[..., None, None]


class TimeDependentMNISTClassifier(nn.Module):
    """
    Clasificador dependiente del tiempo para MNIST.

    Entrada:
        x: (B, 1, 28, 28)
        t: (B,)
    Salida:
        logits: (B, 10)
    """
    def __init__(self, channels=(32, 64, 128), embed_dim: int = 128, num_classes: int = 10):
        super().__init__()

        self.embed = nn.Sequential(
            GaussianRandomFourierFeatures(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        c1, c2, c3 = channels

        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.time1 = Dense(embed_dim, c1)
        self.norm1 = nn.GroupNorm(4, c1)

        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1)
        self.time2 = Dense(embed_dim, c2)
        self.norm2 = nn.GroupNorm(8, c2)

        self.conv3 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.time3 = Dense(embed_dim, c3)
        self.norm3 = nn.GroupNorm(8, c3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(c3, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = self.embed(t)

        h = self.conv1(x)
        h = h + self.time1(emb)
        h = self.norm1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = h + self.time2(emb)
        h = self.norm2(h)
        h = self.act(h)

        h = self.conv3(h)
        h = h + self.time3(emb)
        h = self.norm3(h)
        h = self.act(h)

        h = self.pool(h).flatten(1)
        logits = self.head(h)
        return logits
