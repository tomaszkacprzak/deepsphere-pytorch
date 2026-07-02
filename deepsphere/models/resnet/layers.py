"""Reusable layers for Healpix ResNet regression models."""

# pylint: disable=W0221

import torch.nn.functional as F
from torch import nn

from deepsphere.layers.chebyshev import SphericalChebConv
from deepsphere.layers.samplings.healpix_pool_unpool import HealpixAvgPool


class HealpixPseudoConv(nn.Module):
    """Healpix pseudo-convolution implemented as pooling followed by a channel projection.

    The TensorFlow HealpyPseudoConv with ``p=1`` downsamples neighboring Healpix
    pixels and changes the feature dimension. This PyTorch equivalent applies
    average pooling over Healpix pixel groups of four, then a learned linear
    projection independently at each remaining pixel.
    """

    def __init__(self, in_channels, out_channels, activation=F.relu, bias=True):
        """Initialize the pseudo-convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            activation (callable, optional): Activation function. Defaults to ``torch.nn.functional.relu``.
            bias (bool, optional): Whether to use a bias in the channel projection. Defaults to ``True``.
        """
        super().__init__()
        self.pooling = HealpixAvgPool()
        self.projection = nn.Linear(in_channels, out_channels, bias=bias)
        self.activation = activation

    def forward(self, x):
        """Forward pass for tensors shaped ``[batch, pixels, channels]``."""
        x = self.pooling(x)
        x = self.projection(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class HealpixChebyshev(nn.Module):
    """Healpix Chebyshev graph convolution with optional activation."""

    def __init__(self, in_channels, out_channels, laplacian, kernel_size, activation=F.relu, bias=True):
        """Initialize the Chebyshev layer."""
        super().__init__()
        self.chebyshev = SphericalChebConv(in_channels, out_channels, laplacian, kernel_size)
        if not bias:
            self.chebyshev.chebconv.register_parameter("bias", None)
        self.activation = activation

    def forward(self, x):
        """Forward pass for tensors shaped ``[batch, pixels, channels]``."""
        x = self.chebyshev(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class HealpixResidualLayer(nn.Module):
    """Residual block made of a Chebyshev convolution and layer normalization."""

    def __init__(self, channels, laplacian, kernel_size, activation=F.relu, norm_kwargs=None, bias=True):
        """Initialize the residual block."""
        super().__init__()
        self.chebyshev = SphericalChebConv(channels, channels, laplacian, kernel_size)
        if not bias:
            self.chebyshev.chebconv.register_parameter("bias", None)
        self.norm = nn.LayerNorm(channels, **(norm_kwargs or {}))
        self.activation = activation

    def forward(self, x):
        """Forward pass for tensors shaped ``[batch, pixels, channels]``."""
        residual = x
        x = self.chebyshev(x)
        x = self.norm(x)
        x = x + residual
        if self.activation is not None:
            x = self.activation(x)
        return x


class RegressionHead(nn.Module):
    """Regression head for multivariate outputs."""

    def __init__(self, in_channels, in_pixels, out_features, head_type="dense", dense_layers=None, activation=F.relu, dropout_rate=None):
        """Initialize a dense or convolutional regression head."""
        super().__init__()
        self.head_type = head_type
        self.activation = activation
        if head_type == "dense":
            layers = [nn.Flatten()]
            previous_features = in_channels * in_pixels
            for features in dense_layers or []:
                layers.append(nn.Linear(previous_features, features))
                layers.append(nn.ReLU() if activation is F.relu else nn.Identity())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
                previous_features = features
            layers.append(nn.Linear(previous_features, out_features))
            self.layers = nn.Sequential(*layers)
        elif head_type == "conv":
            self.layers = nn.Sequential(nn.Linear(in_channels, out_features), nn.AdaptiveAvgPool1d(1))
        else:
            raise ValueError("head_type must be 'dense' or 'conv'.")

    def forward(self, x):
        """Forward pass returning ``[batch, out_features]``."""
        if self.head_type == "conv":
            x = self.layers[0](x).permute(0, 2, 1)
            return self.layers[1](x).squeeze(-1)
        return self.layers(x)
