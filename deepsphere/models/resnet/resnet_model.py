"""Residual Healpix graph convolutional network for multivariate regression."""

# pylint: disable=W0221

import logging

import torch.nn.functional as F
from torch import nn

from deepsphere.models.resnet.layers import HealpixChebyshev, HealpixPseudoConv, HealpixResidualLayer, RegressionHead

LOGGER = logging.getLogger(__name__)


class HealpixResNetRegression(nn.Module):
    """Residual convolutional neural network for Healpix multivariate regression."""

    def __init__(
        self,
        in_channels,
        out_features,
        n_pixels,
        base_channels=32,
        downsampling_layers=3,
        cheby_layers=2,
        residual_layers=6,
        head_type="dense",
        dense_layers=None,
        dropout_rate=None,
        poly_degree=5,
        norm_kwargs=None,
        activation=F.relu,
        laplacians=None,
        laplacian_type="normalized",
        smoothing=None,
    ):
        """Initialize the ResNet regression model.

        Args:
            in_channels (int): Number of input features per Healpix pixel.
            out_features (int): Number of regression outputs.
            n_pixels (int): Number of input Healpix pixels.
            base_channels (int, optional): Number of channels after the first downsampling layer.
            downsampling_layers (int, optional): Number of pseudo-convolution downsampling layers.
            cheby_layers (int, optional): Number of Chebyshev downsampling stages.
            residual_layers (int, optional): Number of residual Chebyshev blocks.
            head_type (str, optional): Regression head type, either ``"dense"`` or ``"conv"``.
            dense_layers (list[int], optional): Hidden layer sizes for the dense head.
            dropout_rate (float, optional): Dropout rate in the dense head.
            poly_degree (int, optional): Chebyshev polynomial degree / kernel size.
            norm_kwargs (dict, optional): Keyword arguments for layer normalization.
            activation (callable, optional): Activation used throughout the model.
            laplacians (list[torch.sparse.Tensor], optional): Precomputed Laplacians keyed by their vertex count.
            laplacian_type (str, optional): Laplacian type used when ``laplacians`` is not supplied.
            smoothing (nn.Module, optional): Optional smoothing layer applied before the ResNet body.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        self.smoothing = smoothing
        if self.smoothing is None:
            LOGGER.warning("No smoothing layer is included in the network")

        final_pixels = self._validate_pixels(n_pixels, downsampling_layers + cheby_layers)
        if laplacians is None:
            from deepsphere.utils.laplacian_funcs import get_healpix_laplacians

            laplacians = get_healpix_laplacians(n_pixels, downsampling_layers + cheby_layers + 1, laplacian_type)
        laps_by_size = {lap.size(0): lap for lap in laplacians}

        current_channels = in_channels
        current_pixels = n_pixels
        next_channels = base_channels
        for _ in range(downsampling_layers):
            self.layers.append(HealpixPseudoConv(current_channels, next_channels, activation=activation))
            current_channels = next_channels
            current_pixels //= 4
            next_channels *= 2

        for _ in range(cheby_layers):
            self.layers.append(HealpixChebyshev(current_channels, next_channels, laps_by_size[current_pixels], poly_degree, activation=activation))
            self.layers.append(nn.LayerNorm(next_channels, **(norm_kwargs or {})))
            self.layers.append(HealpixPseudoConv(next_channels, next_channels, activation=activation))
            current_channels = next_channels
            current_pixels //= 4

        for _ in range(residual_layers):
            self.layers.append(HealpixResidualLayer(current_channels, laps_by_size[current_pixels], poly_degree, activation=activation, norm_kwargs=norm_kwargs))

        self.conv_layers = nn.ModuleList(self.layers)
        self.regression_head = RegressionHead(current_channels, final_pixels, out_features, head_type, dense_layers, activation, dropout_rate)

    @staticmethod
    def _validate_pixels(n_pixels, pooling_layers):
        pooled_pixels = n_pixels
        for _ in range(pooling_layers):
            if pooled_pixels % 4 != 0:
                raise ValueError("n_pixels must remain divisible by 4 for each Healpix downsampling layer.")
            pooled_pixels //= 4
        return pooled_pixels

    def get_conv_layers(self):
        """Return graph-convolution layers without the regression head."""
        return self.conv_layers

    def get_head_layers_no_flatten(self):
        """Return dense head layers without the leading flatten layer when present."""
        if getattr(self.regression_head, "head_type", None) == "dense":
            return self.regression_head.layers[1:]
        return self.regression_head.layers

    def forward(self, x):
        """Forward pass for inputs shaped ``[batch, pixels, channels]``."""
        if self.smoothing is not None:
            x = self.smoothing(x)
        for layer in self.layers:
            x = layer(x)
        return self.regression_head(x)
