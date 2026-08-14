"""Landscape-scale prediction: apply the trained habitat filters to a whole raster.

During training and simulation the habitat sub-network only ever sees a
``window_size`` crop centred on the animal.  But it is *fully convolutional* —
four 3x3 convolutions with stride 1 and padding 1, no pooling and no dense
layer — so the same weights can be run over the entire landscape in one pass.
Because a convolution is translation-equivariant, the value at a pixel is
identical to what the model would produce for a window centred there, as long
as the pixel sits further than the receptive field from the raster edge (see
*edge_buffer*).

The result is a landscape-wide habitat-selection surface.  Read it as a
*relative* ranking: it answers "given the animal is somewhere on this
landscape, how attractive is each cell", ignoring whether the animal could
actually reach it in one step.  The movement kernel is what supplies that
constraint, and it is deliberately absent here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


def habitat_edge_buffer(model: nn.Module) -> int:
    """Width in pixels of the border contaminated by zero-padding.

    Each padded convolution pulls in ``kernel_size // 2`` columns of zeros at
    the raster edge, and the contamination accumulates down the stack.  Pixels
    within this many cells of the edge saw padding rather than landscape and
    are not comparable with the interior.
    """
    return sum(
        conv.kernel_size[0] // 2
        for conv in model.conv_habitat.conv2d
        if isinstance(conv, nn.Conv2d)
    )


def predict_habitat_landscape(
    model: nn.Module,
    landscape_rasters: Sequence[torch.Tensor] | torch.Tensor,
    scalars: Sequence[float] | np.ndarray | torch.Tensor,
    normalise: bool = True,
    edge_buffer: int | None = None,
    chunk_rows: int | None = 1024,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Habitat-selection surface for the full landscape at one point in time.

    Parameters
    ----------
    model:
        Fitted ConvJointModel.  Only ``model.conv_habitat`` is used — the
        movement head plays no part in a landscape-scale prediction.
    landscape_rasters:
        The spatial covariates: a sequence of 2-D tensors (one per channel, as
        returned by ``get_landscape``) or a single stacked [C, H, W] tensor.
        These must be the *same layers, in the same order and on the same
        scaling* as the model was trained on.
    scalars:
        The scalar covariates for the moment being predicted, in the order of
        ``SCALAR_COLS`` used in training — e.g. the row that
        :func:`deepssf.make_simulation_inputs` returns,
        ``[sin_hour, cos_hour, sin_yday, cos_yday, dt_hour]``.  Each is
        broadcast to a constant layer covering the whole landscape, exactly as
        ``Scalar_to_Grid_Block`` does for a window (that block cannot be reused
        here because its output size is fixed to the training window).
    normalise:
        Subtract the log-sum-exp over the whole landscape, so the surface is a
        log-probability that sums to 1 across all valid cells.  This is the
        landscape-scale analogue of what ``Conv2d_block_spatial`` does over a
        window.  Set False to get the raw conv output (an unnormalised
        log-intensity — still fine for ranking cells, and comparable in scale
        between two runs).
    edge_buffer:
        Number of border pixels to mark as NaN.  Defaults to
        :func:`habitat_edge_buffer`, the exact width the zero-padding reaches.
        NaN — rather than the -inf used in earlier scripts — keeps the array
        usable in a GeoTIFF and makes matplotlib leave those cells blank.
    chunk_rows:
        Process this many raster rows at a time (with an overlap of
        *edge_buffer*, so the result is identical to a single pass).  A full
        landscape at float32 needs several hundred MB per intermediate
        activation, which is enough to exhaust a GPU.  Pass None to force one
        pass.
    device:
        Where to run.  Defaults to the device the model is on.

    Returns
    -------
    np.ndarray, shape [H, W]
        Log-probability (or log-intensity when ``normalise=False``) per cell,
        with NaN in the *edge_buffer* border.  Take ``np.exp`` for
        probabilities.
    """
    model.eval()
    device = device if device is not None else next(model.parameters()).device

    if isinstance(landscape_rasters, torch.Tensor):
        stack = landscape_rasters
        if stack.ndim != 3:
            raise ValueError(
                f"Expected a [C, H, W] tensor, got shape {tuple(stack.shape)}"
            )
    else:
        stack = torch.stack(
            [torch.as_tensor(r, dtype=torch.float32) for r in landscape_rasters], dim=0
        )
    stack = stack.to(torch.float32)
    n_channels, height, width = stack.shape

    scalar_values = torch.as_tensor(
        np.asarray(scalars, dtype=np.float32).ravel(), dtype=torch.float32
    )

    expected = model.conv_habitat.conv2d[0].in_channels
    if n_channels + len(scalar_values) != expected:
        raise ValueError(
            f"{n_channels} raster channels + {len(scalar_values)} scalars = "
            f"{n_channels + len(scalar_values)}, but the habitat CNN expects "
            f"{expected} input channels. Check that landscape_rasters matches "
            "the layers used for training and that scalars matches SCALAR_COLS."
        )

    buffer = habitat_edge_buffer(model) if edge_buffer is None else int(edge_buffer)
    if 2 * buffer >= min(height, width):
        raise ValueError(
            f"edge_buffer of {buffer} px leaves nothing of a "
            f"{height}x{width} raster."
        )

    out = np.empty((height, width), dtype=np.float32)
    step = height if chunk_rows is None else max(int(chunk_rows), 1)

    with torch.no_grad():
        for r0 in range(0, height, step):
            r1 = min(r0 + step, height)
            # Grow the strip by the receptive field on each side so the rows we
            # keep never see the strip's own zero-padded boundary; clipped at
            # the true raster edge, where that padding is unavoidable (and is
            # what `buffer` masks off below).
            pad0 = max(r0 - buffer, 0)
            pad1 = min(r1 + buffer, height)

            strip = stack[:, pad0:pad1, :].unsqueeze(0).to(device)
            scalar_maps = (
                scalar_values.to(device)
                .view(1, -1, 1, 1)
                .expand(1, -1, strip.shape[2], strip.shape[3])
            )
            full_stack = torch.cat([strip, scalar_maps], dim=1)

            # The raw conv stack, not conv_habitat(...): the block's forward
            # normalises over whatever extent it is handed, which for a strip
            # would be that strip alone.  Normalisation happens once, over the
            # whole landscape, below.
            logits = model.conv_habitat.conv2d(full_stack).squeeze(1).squeeze(0)
            out[r0:r1, :] = logits[r0 - pad0 : r1 - pad0, :].cpu().numpy()

    # Mask the border the padding reached
    if buffer > 0:
        out[:buffer, :] = np.nan
        out[-buffer:, :] = np.nan
        out[:, :buffer] = np.nan
        out[:, -buffer:] = np.nan

    if normalise:
        valid = ~np.isnan(out)
        # log-sum-exp over valid cells only, shifted by the max for stability
        peak = out[valid].max()
        log_z = peak + np.log(np.exp(out[valid] - peak).sum())
        out[valid] -= log_z

    return out
