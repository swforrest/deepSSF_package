"""Simulation / inference: generate trajectories and next-step predictions."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from deepssf.utils import subset_raster_with_padding_torch


def _day_to_month_index(day_of_year: float, base_year: int = 2018) -> int:
    """0-based month index from a day-of-year value (for numpy array indexing).

    Uses base_year as month-0. Different from data.day_to_month_index which
    is 1-based and uses base 2019 for S2 dict key lookup.
    """
    base = datetime(base_year, 1, 1)
    date = base + timedelta(days=int(day_of_year) - 1)
    return (date.month - 1) + (date.year - base.year) * 12


def make_simulation_inputs(
    n_steps: int,
    starting_yday: float,
    starting_hour: float = 0.0,
    time_between_steps: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute cyclic scalar inputs for a simulated trajectory.

    Parameters
    ----------
    n_steps:
        Number of steps to simulate.
    starting_yday:
        Day of year at the first step (1–365).
    starting_hour:
        Hour of day at the first step (0–24).
    time_between_steps:
        Time interval between consecutive steps in hours.  Used to advance the
        clock and compute yday for each step.  Defaults to 1.0.
    Returns
    -------
    x2_full : ndarray, shape (n_steps, 5)
        Rows are ``[sin_hour, cos_hour, sin_yday, cos_yday, dt]`` for each step.
    hour_t2 : ndarray, shape (n_steps,)
        Hour value at each step.
    yday_t2 : ndarray, shape (n_steps,)
        Day-of-year value at each step.
    """
    hour_t2 = np.zeros(n_steps)
    yday_t2 = np.zeros(n_steps)
    x2_full = np.zeros((n_steps, 5))

    for i in range(n_steps):
        # Advance clock by time_between_steps hours; wrap at 24 h and 365 days
        hour = (starting_hour + i * time_between_steps) % 24
        yday = ((starting_yday - 1 + i * time_between_steps/24) % 365) + 1
        hour_t2[i] = hour
        yday_t2[i] = yday
        # Cyclic (sine/cosine) encoding preserves continuity at midnight and year-end
        x2_full[i, 0] = np.sin(2 * np.pi * hour / 24)
        x2_full[i, 1] = np.cos(2 * np.pi * hour / 24)
        x2_full[i, 2] = np.sin(2 * np.pi * yday / 365.25)
        x2_full[i, 3] = np.cos(2 * np.pi * yday / 365.25)
        x2_full[i, 4] = time_between_steps

    return x2_full, hour_t2, yday_t2


def simulate_next_step(
    model: torch.nn.Module,
    landscape_rasters: list[torch.Tensor],
    scalars_to_grid: torch.Tensor,
    bearing: torch.Tensor,
    window_size: int,
    x_loc: float,
    y_loc: float,
    transform: object,
) -> tuple[float, 
           float, 
        #    torch.Tensor, 
        #    torch.Tensor, 
        #    torch.Tensor, 
           int, 
           int]:
    """Sample the next location from the model's predicted step distribution.

    Parameters
    ----------
    model:
        Fitted ConvJointModel (in eval mode).
    landscape_rasters:
        List of 2-D tensors — one per spatial channel — covering the full landscape.
    scalars_to_grid:
        Scalar inputs broadcast to a spatial grid, shape (1, S).
    bearing:
        Previous step bearing, shape (1, 1).
    window_size:
        Side length of the spatial crop (pixels), e.g. 101.
    x_loc:
        Geographic x coordinate of the current location.
    y_loc:
        Geographic y coordinate of the current location.
    transform:
        Rasterio ``Affine`` transform for the landscape rasters.

    Returns
    -------
    new_x, new_y : float
        Sampled geographic coordinates (with sub-pixel jitter applied).
    # hab_log_prob : Tensor, shape (H, W)
    # move_log_prob : Tensor, shape (H, W)
    # step_log_prob : Tensor, shape (H, W)  (masked; NaN outside raster extent)
    px, py : int
        Sampled pixel column and row within the local crop.
    """
    device = next(model.parameters()).device

    # Crop a window_size × window_size patch from every raster channel at the
    # current location; returns the patch plus its top-left pixel coordinates.
    results = [
        subset_raster_with_padding_torch(
            rt, x=x_loc, y=y_loc, window_size=window_size, transform=transform  # type: ignore[arg-type]
        )
        for rt in landscape_rasters
    ]
    subset_tensors, origin_xs, origin_ys = zip(*results, strict=True)

    # Stack channels into [1, C, H, W] and move to the active device
    x1 = torch.stack(list(subset_tensors), dim=0).unsqueeze(0).to(device)
    scalars_to_grid = scalars_to_grid.to(device)
    bearing = bearing.to(device)

    # Cells padded with -1 lie outside the raster extent; replace with NaN so
    # they receive zero probability after the softmax and are never sampled.
    first_channel = x1[0, 0, :, :]
    mask = torch.where(
        first_channel == -1, torch.tensor(float("nan")), torch.ones_like(first_channel)
    )

    out = model((x1, scalars_to_grid, bearing))
    hab_log_prob = out[:, :, :, 0]
    move_log_prob = out[:, :, :, 1]
    # Multiply by mask: NaN * 1 = NaN for out-of-bounds cells
    step_log_prob = (hab_log_prob + move_log_prob) * mask

    # Convert to probability, zero out NaN cells, then renormalise to sum to 1
    step_prob = torch.exp(step_log_prob.squeeze())
    step_prob = torch.nan_to_num(step_prob, nan=0.0)
    step_prob_norm = step_prob / torch.sum(step_prob)

    # Sample one pixel index from the discrete probability distribution
    flat = step_prob_norm.flatten().detach().cpu().numpy()
    sampled_index = np.random.choice(flat.size, p=flat)
    sampled_row, sampled_col = np.unravel_index(sampled_index, step_prob_norm.shape)

    # Convert sampled local pixel to global pixel, then to geographic coordinates
    new_px = origin_xs[0] + sampled_col
    new_py = origin_ys[0] + sampled_row
    new_x, new_y = transform * (new_px, new_py)  # type: ignore[operator]

    # Sub-pixel jitter: uniform-ish within one cell (~95% within [0,25] / [-25,0])
    # Adds positional uncertainty below the pixel resolution to avoid all simulated
    # locations snapping to pixel-centre coordinates.
    while True:
        jitter_x = np.random.normal(12.5, 6.5)
        if 0.0 <= jitter_x <= 25.0:
            break
    while True:
        jitter_y = np.random.normal(-12.5, 6.5)
        if -25.0 <= jitter_y <= 0.0:
            break

    return (
        float(new_x) + jitter_x,
        float(new_y) + jitter_y,
        # hab_log_prob.squeeze().cpu(),
        # move_log_prob.squeeze().cpu(),
        # step_log_prob.squeeze().cpu(),
        int(sampled_col),
        int(sampled_row),
    )


def simulate_trajectory(
    model: torch.nn.Module,
    get_landscape: Callable[[int], list[torch.Tensor]],
    transform: object,
    start_x: float,
    start_y: float,
    n_steps: int,
    starting_yday: float = 1.0,
    starting_hour: float = 0.0,
    time_between_steps: float = 1.0,
    window_size: int = 101,
    base_year: int = 2018,
    month_index_fn: Callable[[float], int] | None = None,
) -> pd.DataFrame:
    """Simulate a trajectory by rolling the model forward.

    Parameters
    ----------
    model:
        Fitted ConvJointModel (in eval mode).
    get_landscape:
        Callable ``(month_index: int) -> list[Tensor]``.  The caller is
        responsible for returning the correct set of raster tensors for the
        given month index.
    transform:
        Rasterio ``Affine`` transform shared by all landscape rasters.
    start_x, start_y:
        Starting geographic coordinates.
    n_steps:
        Number of steps to simulate.
    starting_yday:
        Day of year at step 0 (1–365).
    starting_hour:
        Hour of day at step 0 (0–24).
    window_size:
        Side length of the spatial crop in pixels.
    base_year:
        Base year for the default 0-based month index (default 2018).
        Ignored when *month_index_fn* is provided.
    month_index_fn:
        Optional callable ``(yday: float) -> int`` converting day-of-year to
        the month index passed to *get_landscape*.  Defaults to
        ``_day_to_month_index(yday, base_year)`` (0-based).  Pass the same
        function used in ``validate_next_step_probs`` to share a single
        *get_landscape* callable across both.

    Returns
    -------
    pd.DataFrame with columns:
        x, 
        y, 
        hour, 
        yday, 
        month_index, 
        # hab_log_prob, 
        # move_log_prob, 
        # step_log_prob
    """
    model.eval()
    # Pre-compute cyclic time encodings for every step up front
    x2_full, hour_t2, yday_t2 = make_simulation_inputs(
        n_steps, starting_yday, starting_hour, time_between_steps
    )

    _month_fn = month_index_fn if month_index_fn is not None else (
        lambda yday: _day_to_month_index(yday, base_year)
    )

    rows: list[dict] = []
    x_loc, y_loc = start_x, start_y
    # No previous bearing at the start of the trajectory
    bearing = torch.zeros(1, 1)

    # Load the landscape for the starting month; only reload when the month changes
    # to avoid re-reading large rasters on every step.
    previous_yday: float | None = None
    month_index = _month_fn(starting_yday)
    landscape_rasters = get_landscape(month_index)

    with torch.no_grad():
        for i in range(n_steps):
            yday = float(yday_t2[i])
            # Reload landscape rasters only when the month changes
            if yday != previous_yday:
                month_index = _month_fn(yday)
                landscape_rasters = get_landscape(month_index)
                previous_yday = yday

            # Wrap precomputed scalar row as a [1, 5] tensor for the model
            scalars_to_grid = torch.tensor(x2_full[i], dtype=torch.float32).unsqueeze(0)

            new_x, new_y, px, py = simulate_next_step( #hab_lp, move_lp, step_lp, 
                model,
                landscape_rasters,
                scalars_to_grid,
                bearing,
                window_size,
                x_loc,
                y_loc,
                transform,
            )

            rows.append(
                {
                    "x": new_x,
                    "y": new_y,
                    "hour": float(hour_t2[i]),
                    "yday": yday,
                    "month_index": month_index,
                    # "hab_log_prob": hab_lp.numpy(),
                    # "move_log_prob": move_lp.numpy(),
                    # "step_log_prob": step_lp.numpy(),
                }
            )

            # Update bearing from the sampled displacement; used as input to the
            # movement sub-network on the next step (directional persistence).
            dx = new_x - x_loc
            dy = new_y - y_loc
            raw_bearing = float(np.arctan2(dy, dx))
            bearing = torch.tensor([[raw_bearing]], dtype=torch.float32)

            x_loc, y_loc = new_x, new_y

    return pd.DataFrame(rows)