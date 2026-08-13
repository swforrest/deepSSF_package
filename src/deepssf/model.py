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
        # conv stack collapses C channels → 1; squeeze removes singleton → [B, H, W]
        out = self.conv2d(x).squeeze(dim=1)
        # subtract log-sum-exp so the surface integrates to 1 in probability space
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
    log-scale, in forms chosen so that no intermediate term can overflow in
    float32 (see :meth:`_vonmises_log` and :attr:`raw_clamp`).

    No change-of-variables Jacobian is applied (polar → Cartesian).
    See :class:`Params_to_Grid_Block_ChV` for the Jacobian-corrected version.

    Notes
    -----
    Mixture weights are obtained with ``log_softmax`` over the *raw* FCN
    outputs (parameters 2, 5, 8 and 11).  Versions ≤ 0.2.3 applied a softmax to
    ``exp(raw)``, a double exponential that saturated to exactly ``(1, 0)`` by
    a raw value of ≈ 4.75 — zeroing the weight gradient and putting
    ``log(0) = -inf`` into the surface.  Checkpoints written by those versions
    therefore interpret these four parameters differently.
    """

    #: Raw FCN outputs are clamped to ±this before being exponentiated, so
    #: shape/scale/concentration stay in ``[exp(-10), exp(10)]`` no matter what
    #: the network emits.  Without it, ``exp`` overflows to ``inf`` at a raw
    #: value of ≈ 88 and ``lgamma(inf) - inf`` produces NaN.
    raw_clamp: float = 10.0

    def __init__(self, params: ModelParams) -> None:
        super().__init__()
        self.image_dim  = params.image_dim
        self.pixel_size = params.pixel_size
        self.device     = params.device

        center = self.image_dim // 2
        # Build pixel-index grids: y_idx increases downward, x_idx increases rightward
        y_idx, x_idx = np.indices((self.image_dim, self.image_dim))
        # Euclidean distance from the centre pixel, in CRS units (e.g. metres)
        dist = np.sqrt((self.pixel_size * (x_idx - center)) ** 2 +
                       (self.pixel_size * (y_idx - center)) ** 2)
        # Centre pixel has distance 0 → log(0) is undefined in the Gamma PDF.
        # Replace with E[r] for a uniform distribution within one pixel cell.
        dist[center, center] = 0.3826 * self.pixel_size  # E[r] within centre pixel

        # Registered as buffers (not plain attributes) so `model.to(device)`
        # moves them once, instead of copying them host→device every forward.
        # persistent=False keeps them out of state_dict: they are constants
        # derived from image_dim/pixel_size, so storing them would bloat every
        # checkpoint and break strict loading of checkpoints from earlier versions.
        self.register_buffer(
            "distance_layer", torch.from_numpy(dist).float(), persistent=False
        )
        # arctan2 with (center - y, x - center) gives bearing measured from east,
        # increasing counter-clockwise — consistent with the movement bearing convention
        self.register_buffer(
            "bearing_layer",
            torch.from_numpy(np.arctan2(center - y_idx, x_idx - center)).float(),
            persistent=False,
        )

    def _positive(self, raw: torch.Tensor) -> torch.Tensor:
        """Map a raw FCN output to a strictly positive, finite parameter."""
        return torch.exp(raw.clamp(-self.raw_clamp, self.raw_clamp))

    def _gamma_log(self, r, shape, scale):
        # Log-PDF of Gamma(shape, scale) evaluated at each pixel distance r
        shape, scale = shape.to(r.device), scale.to(r.device)
        return (
            -torch.lgamma(shape) - shape * torch.log(scale)
            + (shape - 1) * torch.log(r) - r / scale
        )

    def _vonmises_log(self, theta, kappa, mu):
        """Log-PDF of the von Mises distribution, written to avoid overflow.

        The direct form ``kappa*cos(θ-mu) - log(2π·I₀(kappa))`` breaks down in
        float32: ``I₀`` grows like ``e^κ/√(2πκ)`` and overflows to ``inf`` at
        κ ≈ 89, i.e. a raw FCN output of only ``log(89) ≈ 4.5``.  ``log_norm``
        then becomes ``inf`` and the component log-density ``-inf``; if both
        mixture components overflow the log-sum-exp evaluates ``-inf - (-inf)``
        and the whole surface turns to NaN.  Even one overflowing component
        leaves the forward pass finite but yields NaN gradients, which silently
        poison the weights.

        Using the exponentially-scaled Bessel function ``i0e(κ) = e^-κ·I₀(κ)``
        gives ``log I₀(κ) = κ + log(i0e(κ))``, so the ``κ`` terms cancel::

            κ·cos(θ-μ) - log(2π·I₀(κ)) = κ·(cos(θ-μ) - 1) - log(2π) - log(i0e(κ))

        Since ``cos(θ-μ) - 1 ∈ [-2, 0]``, no term can overflow for any κ.
        ``i0e`` has native CPU, CUDA and MPS kernels, so this also removes the
        host round-trip the previous implementation needed for MPS.
        """
        kappa, mu = kappa.to(theta.device), mu.to(theta.device)
        return (
            kappa * (torch.cos(theta - mu) - 1.0)
            - float(np.log(2 * np.pi))
            - torch.log(torch.special.i0e(kappa))
        )

    def _expand(self, scalar, dim):
        # Broadcast a per-sample scalar (shape [B]) into a [B, dim, dim] spatial grid
        # so each pixel in the grid carries the same value for that batch item.
        # expand() returns a view rather than materialising the full grid.
        return scalar.reshape(-1, 1, 1).expand(-1, dim, dim)

    def forward(self, x: torch.Tensor, bearing: torch.Tensor) -> torch.Tensor:
        D = self.image_dim
        E = self._expand

        # Mixture weights: log_softmax over the RAW outputs, giving log-weights
        # that can be added directly to the component log-densities below.
        log_gw = torch.log_softmax(torch.stack([x[:, 2], x[:, 5]], dim=0), dim=0)
        log_vw = torch.log_softmax(torch.stack([x[:, 8], x[:, 11]], dim=0), dim=0)

        # --- Step-length kernel: 2-component Gamma mixture ---
        # FCN outputs raw scalars; _positive() bounds and exponentiates them.
        # gc2 scaled ×500 so the second component spans longer distances (heavy tail).
        gs1 = E(self._positive(x[:, 0]), D)   # shape of Gamma component 1
        gc1 = E(self._positive(x[:, 1]), D)   # scale of Gamma component 1
        gs2 = E(self._positive(x[:, 3]), D)   # shape of Gamma component 2
        gc2 = E(self._positive(x[:, 4]) * 500, D)  # scale of component 2 (long-range)

        dist = self.distance_layer
        gl1 = E(log_gw[0], D) + self._gamma_log(dist, gs1, gc1)
        gl2 = E(log_gw[1], D) + self._gamma_log(dist, gs2, gc2)
        # logsumexp over the weighted components: stable for any component value,
        # including -inf, unlike the manual max/exp/log formulation it replaces.
        gamma_grid = torch.logsumexp(torch.stack([gl1, gl2], dim=0), dim=0)

        # --- Turning-angle kernel: 2-component von Mises mixture ---
        brg = self.bearing_layer
        # mu is learned as an offset from the previous bearing so the animal can
        # encode forward persistence or a preferred turn relative to its last heading.
        # mu is an offset from prev bearing; k is concentration
        mu1 = E(x[:, 6]  + bearing[:, 0], D)
        k1  = E(self._positive(x[:, 7]), D)
        mu2 = E(x[:, 9]  + bearing[:, 0], D)
        k2  = E(self._positive(x[:, 10]), D)

        vl1 = E(log_vw[0], D) + self._vonmises_log(brg, k1, mu1)
        vl2 = E(log_vw[1], D) + self._vonmises_log(brg, k2, mu2)
        vm_grid = torch.logsumexp(torch.stack([vl1, vl2], dim=0), dim=0)

        # Joint log-density: add step-length and turning-angle log-probs, then normalise
        grid = gamma_grid + vm_grid
        return grid - torch.logsumexp(grid, dim=(1, 2), keepdim=True)


class Params_to_Grid_Block_ChV(Params_to_Grid_Block):
    """Same as :class:`Params_to_Grid_Block` but with a change-of-variables
    Jacobian correction (polar → Cartesian) applied to the Gamma density.

    Subtracts ``log(r)`` from the log-density (i.e. divides the density by
    ``r``) to account for the polar-to-Cartesian area element ``r dr dθ``.
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
        # Broadcast each scalar to a constant spatial map and concatenate with
        # the raster channels so the CNN sees both spatial and temporal context.
        scalar_maps  = self.scalar_grid_output(scalars)
        all_spatial  = torch.cat([spatial, scalar_maps], dim=1)

        # Habitat branch: CNN → log-normalised [B, H, W] surface
        habitat_out  = self.conv_habitat(all_spatial)
        # Movement branch: CNN → flatten → FCN → 12 parameters → [B, H, W] surface
        move_conv    = self.conv_movement(all_spatial)
        move_params  = self.fcn_movement_all(move_conv)
        move_out     = self.movement_grid_output(move_params, bearing)

        # Stack as last dim: [..., 0] = habitat log-prob, [..., 1] = movement log-prob.
        # Summing over this dim gives the joint log-density used by the loss function.
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