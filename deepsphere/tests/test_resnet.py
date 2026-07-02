"""Tests for the Healpix ResNet regression model."""

import unittest

import torch

from deepsphere.models.resnet import HealpixResNetRegression


def sparse_identity(size):
    """Create a sparse identity matrix for lightweight graph convolution tests."""
    indices = torch.arange(size, dtype=torch.long).repeat(2, 1)
    values = torch.ones(size)
    return torch.sparse_coo_tensor(indices, values, (size, size)).coalesce()


class TestHealpixResNetRegression(unittest.TestCase):
    """Test the Healpix ResNet regression model."""

    def test_forward_dense_head_returns_multivariate_regression_output(self):
        """The model downsamples Healpix maps and returns one vector per sample."""
        model = HealpixResNetRegression(
            in_channels=2,
            out_features=6,
            n_pixels=48,
            base_channels=4,
            downsampling_layers=1,
            cheby_layers=1,
            residual_layers=2,
            dense_layers=[8],
            poly_degree=2,
            laplacians=[sparse_identity(48), sparse_identity(12), sparse_identity(3)],
        )

        output = model(torch.randn(5, 48, 2))

        self.assertEqual(output.shape, (5, 6))

    def test_conv_layers_exclude_regression_head(self):
        """The convolutional body is available separately from the regression head."""
        model = HealpixResNetRegression(
            in_channels=1,
            out_features=3,
            n_pixels=48,
            base_channels=2,
            downsampling_layers=1,
            cheby_layers=0,
            residual_layers=1,
            poly_degree=2,
            laplacians=[sparse_identity(12)],
        )

        self.assertEqual(len(model.get_conv_layers()), 2)
        self.assertEqual(len(model.get_head_layers_no_flatten()), 1)


if __name__ == "__main__":
    unittest.main()
