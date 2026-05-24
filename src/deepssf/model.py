"""Model architecture: the deepSSF network definition.

The joint model (``ConvJointModel``) combines two sub-networks:

* **Habitat sub-network** – a stack of 2-D convolutions that produces a
  log-normalised probability surface over the local landscape patch.
* **Movement sub-network** – convolutions followed by fully connected layers
  that output parameters for a mixture-of-Gamma × mixture-of-von-Mises
  movement kernel, converted to the same spatial grid.

The final output is the element-wise sum of both log-probability grids, which
is the joint log-likelihood of the next observed step.

Usage::

    from deepssf.model import ConvJointModel, ModelParams

    params = ModelParams(params_dict)
    model  = ConvJointModel(params)
    output = model((spatial, scalars, bearing))   # (B, H, W, 2)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from deepssf.utils import get_device

# ---------------------------------------------------------------------------
# Habitat sub-network
# ---------------------------------------------------------------------------

class Conv2d_block_spatial(nn.Module):
    """CNN block that outputs a log-normalised habitat-selection surface.

    Four successive conv layers (3 with ReLU + 1 final) collapse the
    multi-band spatial input to a single log-probability map of shape
    [B, H, W].
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        ic = params.input_channels
        oc = params.output_channels
        k  = params.kernel_size
        s  = params.stride
        p  = params.padding

        self.conv2d = nn.Sequential(
            nn.Conv2d(ic, oc, k, s, p), nn.ReLU(),
            nn.Conv2d(oc, oc, k, s, p), nn.ReLU(),
            nn.Conv2d(oc, oc, k, s, p), nn.ReLU(),
            nn.Conv2d(oc,  1, k, s, p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(x).squeeze(dim=1)
        return out - torch.logsumexp(out, dim=(1, 2), keepdim=True)


# ---------------------------------------------------------------------------
# Movement CNN → FC bridge
# ---------------------------------------------------------------------------

class Conv2d_block_toFC(nn.Module):
    """CNN block with max-pooling that flattens the spatial input for the FCN."""

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        ic  = params.input_channels
        oc  = params.output_channels
        k   = params.kernel_size
        s   = params.stride
        p   = params.padding
        kmp = params.kernel_size_mp
        smp = params.stride_mp

        self.conv2d = nn.Sequential(
            nn.Conv2d(ic, oc, k, s, p), nn.ReLU(), nn.MaxPool2d(kmp, smp),
            nn.Conv2d(oc, oc, k, s, p), nn.ReLU(), nn.MaxPool2d(kmp, smp),
            nn.Conv2d(oc, oc, k, s, p), nn.ReLU(), nn.MaxPool2d(kmp, smp),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2d(x)


# ---------------------------------------------------------------------------
# Movement fully-connected block
# ---------------------------------------------------------------------------

class FCN_block_all_movement(nn.Module):
    """Three-layer FCN that maps flattened spatial features to movement parameters."""

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        dim_in  = params.dense_dim_in_all
        dim_h   = params.dense_dim_hidden
        n_out   = params.num_movement_params
        dropout = params.dropout

        self.ffn = nn.Sequential(
            nn.Linear(dim_in, dim_h), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(dim_h,  dim_h), nn.Dropout(dropout), nn.ReLU(),
            nn.Linear(dim_h,  n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# ---------------------------------------------------------------------------
# Movement parameters → 2-D probability grid
# ---------------------------------------------------------------------------

class Params_to_Grid_Block(nn.Module):
    """Convert FCN movement parameters to a log-normalised 2-D movement grid.

    Models step-length with a 2-component Gamma mixture and turning-angle with
    a 2-component von Mises mixture.  All densities are computed on the
    log-scale for numerical stability.

    No change-of-variables Jacobian is applied (polar → Cartesian).
    See :class:`Params_to_Grid_Block_ChV` for the Jacobian-corrected version.
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.image_dim  = params.image_dim
        self.pixel_size = params.pixel_size
        self.device     = params.device

        center = self.image_dim // 2
        y_idx, x_idx = np.indices((self.image_dim, self.image_dim))
        dist = np.sqrt((self.pixel_size * (x_idx - center)) ** 2 +
                       (self.pixel_size * (y_idx - center)) ** 2)
        dist[center, center] = 0.3826 * self.pixel_size  # E[r] within centre pixel

        self.distance_layer = torch.from_numpy(dist).float()
        self.bearing_layer = torch.from_numpy(
            np.arctan2(center - y_idx, x_idx - center)
        ).float()

    def _gamma_log(self, r, shape, scale):
        shape, scale = shape.to(r.device), scale.to(r.device)
        return (
            -torch.lgamma(shape) - shape * torch.log(scale)
            + (shape - 1) * torch.log(r) - r / scale
        )

    def _vonmises_log(self, theta, kappa, mu):
        kappa, mu = kappa.to(theta.device), mu.to(theta.device)
        # torch.special.i0 is unsupported on MPS; compute on CPU and move back
        i0_val = torch.special.i0(kappa.cpu()).to(kappa.device)
        log_norm = np.log(2 * torch.pi) + torch.log(i0_val)
        return kappa * torch.cos(theta - mu) - log_norm

    def _expand(self, scalar, dim):
        return scalar.unsqueeze(0).unsqueeze(0).repeat(dim, dim, 1).permute(2, 0, 1)

    def forward(self, x: torch.Tensor, bearing: torch.Tensor) -> torch.Tensor:
        D = self.image_dim
        E = self._expand

        gs1 = E(torch.exp(x[:, 0]), D)
        gc1 = E(torch.exp(x[:, 1]), D)
        gw1 = E(torch.exp(x[:, 2]), D)
        gs2 = E(torch.exp(x[:, 3]), D)
        gc2 = E(torch.exp(x[:, 4]) * 500, D)
        gw2 = E(torch.exp(x[:, 5]), D)
        gw  = torch.nn.functional.softmax(torch.stack([gw1, gw2], dim=0), dim=0)
        gw1, gw2 = gw[0], gw[1]

        dist = self.distance_layer.to(x.device)
        gl1 = self._gamma_log(dist, gs1, gc1)
        gl2 = self._gamma_log(dist, gs2, gc2)
        lse = torch.max(gl1, gl2)
        gamma_grid = lse + torch.log(
            gw1 * torch.exp(gl1 - lse) + gw2 * torch.exp(gl2 - lse)
        )

        brg = self.bearing_layer.to(x.device)
        mu1 = E(x[:, 6]  + bearing[:, 0], D)
        k1  = E(torch.exp(x[:, 7]), D)
        vw1 = E(torch.exp(x[:, 8]), D)
        mu2 = E(x[:, 9]  + bearing[:, 0], D)
        k2  = E(torch.exp(x[:, 10]), D)
        vw2 = E(torch.exp(x[:, 11]), D)
        vw  = torch.nn.functional.softmax(torch.stack([vw1, vw2], dim=0), dim=0)
        vw1, vw2 = vw[0], vw[1]

        vl1 = self._vonmises_log(brg, k1, mu1)
        vl2 = self._vonmises_log(brg, k2, mu2)
        lse = torch.max(vl1, vl2)
        vm_grid = lse + torch.log(
            vw1 * torch.exp(vl1 - lse) + vw2 * torch.exp(vl2 - lse)
        )

        grid = gamma_grid + vm_grid
        return grid - torch.logsumexp(grid, dim=(1, 2), keepdim=True)


class Params_to_Grid_Block_ChV(Params_to_Grid_Block):
    """Same as :class:`Params_to_Grid_Block` but with a change-of-variables
    Jacobian correction (polar → Cartesian) applied to the Gamma density.

    Divides the log-Gamma density by ``log(r)`` (i.e. subtracts it) to
    account for the polar-to-Cartesian area element ``r dr dθ``.
    """

    def _gamma_log(self, r, shape, scale):
        shape, scale = shape.to(r.device), scale.to(r.device)
        return (
            -torch.lgamma(shape) - shape * torch.log(scale)
            + (shape - 1) * torch.log(r) - r / scale
            - torch.log(r)
        )


# ---------------------------------------------------------------------------
# Scalar → 2-D grid broadcast
# ---------------------------------------------------------------------------

class Scalar_to_Grid_Block(nn.Module):
    """Broadcast scalar features into constant-valued spatial maps.

    Converts a [B, S] tensor of scalar values into a [B, S, H, W] tensor
    where every pixel in each map carries the same scalar value.  This lets
    scalar predictors (e.g., time-of-day) enter the convolutional stream.
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.image_dim = params.image_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S = x.shape
        return x.view(B, S, 1, 1).expand(B, S, self.image_dim, self.image_dim)


# ---------------------------------------------------------------------------
# Full joint model
# ---------------------------------------------------------------------------

class ConvJointModel(nn.Module):
    """The deepSSF joint model combining habitat and movement sub-networks.

    Forward input
    -------------
    x : tuple of three tensors
        * ``x[0]`` – spatial covariates, shape [B, C_spatial, H, W]
        * ``x[1]`` – scalar features to broadcast, shape [B, S]
        * ``x[2]`` – previous bearing, shape [B, 1]

    Forward output
    --------------
    torch.Tensor of shape [B, H, W, 2]
        Stack of log-probability grids: index 0 = habitat, index 1 = movement.
        Sum over the last dim gives the joint log-density.
    """

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.scalar_grid_output  = Scalar_to_Grid_Block(params)
        self.conv_habitat        = Conv2d_block_spatial(params)
        self.conv_movement       = Conv2d_block_toFC(params)
        self.fcn_movement_all    = FCN_block_all_movement(params)
        self.movement_grid_output = Params_to_Grid_Block_ChV(params)
        self.device = params.device

    def forward(self, x: tuple) -> torch.Tensor:
        spatial, scalars, bearing = x[0], x[1], x[2]
        scalar_maps  = self.scalar_grid_output(scalars)
        all_spatial  = torch.cat([spatial, scalar_maps], dim=1)

        habitat_out  = self.conv_habitat(all_spatial)
        move_conv    = self.conv_movement(all_spatial)
        move_params  = self.fcn_movement_all(move_conv)
        move_out     = self.movement_grid_output(move_params, bearing)

        return torch.stack((habitat_out, move_out), dim=-1)


# ---------------------------------------------------------------------------
# Parameter container
# ---------------------------------------------------------------------------

class ModelParams:
    """Lightweight container for all model hyper-parameters.

    Construct from a dictionary::

        params = ModelParams({
            "batch_size": 32,
            "image_dim": 101,
            "pixel_size": 25,
            "dim_in_nonspatial_to_grid": 4,
            "dense_dim_in_nonspatial": 4,
            "dense_dim_hidden": 128,
            "dense_dim_in_all": 2500,
            "input_channels": 8,      # spatial layers + scalar layers
            "output_channels": 4,
            "kernel_size": 3,
            "stride": 1,
            "kernel_size_mp": 2,
            "stride_mp": 2,
            "padding": 1,
            "num_movement_params": 12,
            "dropout": 0.1,
            "device": "cpu",
        })
    """

    def __init__(self, d: dict) -> None:
        self.batch_size                = d["batch_size"]
        self.image_dim                 = d["image_dim"]
        self.pixel_size                = d["pixel_size"]
        self.dim_in_nonspatial_to_grid = d["dim_in_nonspatial_to_grid"]
        self.dense_dim_in_nonspatial   = d["dense_dim_in_nonspatial"]
        self.dense_dim_hidden          = d["dense_dim_hidden"]
        self.dense_dim_in_all          = d["dense_dim_in_all"]
        self.input_channels            = d["input_channels"]
        self.output_channels           = d["output_channels"]
        self.kernel_size               = d["kernel_size"]
        self.stride                    = d["stride"]
        self.kernel_size_mp            = d["kernel_size_mp"]
        self.stride_mp                 = d["stride_mp"]
        self.padding                   = d["padding"]
        self.num_movement_params       = d["num_movement_params"]
        self.dropout                   = d["dropout"]
        self.device                    = d.get("device", get_device())