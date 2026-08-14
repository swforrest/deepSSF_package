"""Simulation / inference: generate trajectories and next-step predictions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Multiple trajectories
# ---------------------------------------------------------------------------

def _pixel_from_coords(
    transform, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``~transform * (x, y)`` returning integer (col, row) arrays.

    The Affine coefficients are applied directly because ``~transform * (x, y)``
    only accepts one point at a time, and the per-point Python call dominates
    the loop once several trajectories are stepped together.
    """
    inv = ~transform
    col = inv.a * x + inv.b * y + inv.c
    row = inv.d * x + inv.e * y + inv.f
    return np.floor(col).astype(np.int64), np.floor(row).astype(np.int64)


def _coords_from_pixel(
    transform, col: np.ndarray, row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``transform * (col, row)`` (upper-left corner of each pixel)."""
    x = transform.a * col + transform.b * row + transform.c
    y = transform.d * col + transform.e * row + transform.f
    return x, y


def _pad_landscape(
    landscape_rasters: list[torch.Tensor], window_size: int, device: str | torch.device
) -> torch.Tensor:
    """Stack channels into [C, H + 2h, W + 2h], padded with -1.

    Padding once per landscape (rather than clipping per window, as
    ``subset_raster_with_padding_torch`` does) turns window extraction into a
    single advanced-indexing operation over the whole batch.  ``-1`` marks
    out-of-extent cells, which are masked to zero probability before sampling.
    """
    stack = torch.stack([t.to(torch.float32) for t in landscape_rasters], dim=0)
    half = window_size // 2
    return torch.nn.functional.pad(
        stack, (half, half, half, half), mode="constant", value=-1.0
    ).to(device)


def _crop_windows(
    padded: torch.Tensor, cols: torch.Tensor, rows: torch.Tensor, window_size: int
) -> torch.Tensor:
    """Extract one window per (col, row) centre from a padded landscape.

    Returns [N, C, window_size, window_size].  ``cols``/``rows`` are centres in
    *unpadded* pixel coordinates; the padding offset makes them the top-left
    corner of the window in padded coordinates.
    """
    idx = torch.arange(window_size, device=padded.device)
    row_idx = rows[:, None] + idx  # [N, ws]
    col_idx = cols[:, None] + idx
    # [C, N, ws, ws] -> [N, C, ws, ws]
    return padded[:, row_idx[:, :, None], col_idx[:, None, :]].permute(1, 0, 2, 3)


def _truncated_normal(
    n: int, loc: float, scale: float, low: float, high: float
) -> np.ndarray:
    """Rejection-sample a normal truncated to [low, high] (vectorised).

    Deliberately NumPy rather than torch: this runs once per simulation step on
    an array of a few dozen values, and each rejection round on an accelerator
    would need a device→host sync to test the loop condition — enough to
    dominate the step on MPS/CUDA.
    """
    out = np.empty(n)
    invalid = np.ones(n, dtype=bool)
    while invalid.any():
        out[invalid] = np.random.normal(loc, scale, size=int(invalid.sum()))
        invalid = (out < low) | (out > high)
    return out


def _broadcast_start(value, n: int, name: str) -> np.ndarray:
    """Accept a scalar or a length-*n* sequence and return a length-*n* array."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.repeat(arr[0], n)
    if arr.size != n:
        raise ValueError(
            f"{name} has {arr.size} values but n_trajectories is {n}; "
            "pass a scalar or one value per trajectory."
        )
    return arr


def simulate_trajectories_batched(
    model: torch.nn.Module,
    get_landscape: Callable[[int], list[torch.Tensor]],
    transform: object,
    start_x,
    start_y,
    n_steps: int,
    n_trajectories: int | None = None,
    starting_yday: float = 1.0,
    starting_hour: float = 0.0,
    time_between_steps: float = 1.0,
    window_size: int = 101,
    base_year: int = 2018,
    month_index_fn: Callable[[float], int] | None = None,
    device: str | torch.device | None = None,
    pixel_size: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Simulate many trajectories at once, stepping them all in lockstep.

    Every trajectory advances one step per iteration, so the model sees a batch
    of ``n_trajectories`` windows per forward pass instead of one.  The forward
    pass is the entire cost of the simulation, and it is far better utilised at
    batch size 50 than at batch size 1 — this is normally an order of magnitude
    faster than looping over :func:`simulate_trajectory`, and the gap widens on
    a GPU.

    The trade-off is that all trajectories share one clock: ``starting_yday``
    and ``starting_hour`` are scalars, because a single landscape stack is
    cropped for the whole batch each step.  For per-trajectory start *times*,
    use :func:`simulate_trajectories` with ``method="sequential"`` or
    ``"parallel"``.  Per-trajectory start *locations* are supported here.

    Parameters
    ----------
    model:
        Fitted ConvJointModel.
    get_landscape:
        Callable ``(month_index: int) -> list[Tensor]``, one 2-D tensor per
        spatial channel.  Called only when the month index changes.
    transform:
        Rasterio ``Affine`` transform shared by all landscape rasters.
    start_x, start_y:
        Starting coordinates — scalars (all trajectories start together) or
        sequences of length *n_trajectories*.
    n_steps:
        Number of steps per trajectory.
    n_trajectories:
        Number of trajectories.  Inferred from the length of *start_x* when
        that is a sequence; required when *start_x* is a scalar.
    starting_yday, starting_hour, time_between_steps, window_size, base_year,
    month_index_fn:
        As in :func:`simulate_trajectory`.
    device:
        Device to run on.  Defaults to the device the model is already on.
    pixel_size:
        ``(size_x, size_y)`` in CRS units, used to scale the sub-pixel jitter.
        Defaults to the resolution implied by *transform*.  (Note that
        :func:`simulate_next_step` hard-codes a 25 m cell, so a batched run and
        a sequential run differ slightly on rasters of any other resolution.)

    Returns
    -------
    pd.DataFrame
        Long format, one row per trajectory-step, with columns
        ``trajectory_id, step, x, y, hour, yday, month_index``.
    """
    if n_trajectories is None:
        n_trajectories = int(np.atleast_1d(np.asarray(start_x)).size)
    n_traj = int(n_trajectories)
    if n_traj < 1:
        raise ValueError("n_trajectories must be >= 1")

    x_loc = _broadcast_start(start_x, n_traj, "start_x")
    y_loc = _broadcast_start(start_y, n_traj, "start_y")

    model.eval()
    device = device if device is not None else next(model.parameters()).device
    model.to(device)

    # Jitter is applied from the pixel's upper-left corner, so its sign follows
    # the transform: +x eastwards, and -y for the usual north-up raster.
    if pixel_size is None:
        size_x, size_y = abs(transform.a), abs(transform.e)  # type: ignore[attr-defined]
    else:
        size_x, size_y = float(pixel_size[0]), float(pixel_size[1])
    y_sign = -1.0 if transform.e < 0 else 1.0  # type: ignore[attr-defined]

    x2_full, hour_t2, yday_t2 = make_simulation_inputs(
        n_steps, starting_yday, starting_hour, time_between_steps
    )
    scalars_all = torch.tensor(x2_full, dtype=torch.float32, device=device)

    _month_fn = month_index_fn if month_index_fn is not None else (
        lambda yday: _day_to_month_index(yday, base_year)
    )

    # Landscape extent, used to keep window centres inside the padded array
    probe = get_landscape(_month_fn(float(yday_t2[0])))
    height, width = probe[0].shape[-2], probe[0].shape[-1]

    xs_out = np.empty((n_steps, n_traj))
    ys_out = np.empty((n_steps, n_traj))
    month_out = np.empty(n_steps, dtype=np.int64)

    bearing = torch.zeros(n_traj, 1, device=device)
    padded: torch.Tensor | None = None
    previous_month: int | None = None

    with torch.no_grad():
        for i in range(n_steps):
            yday = float(yday_t2[i])
            month_index = _month_fn(yday)
            # Re-pad only when the month (and so the landscape) actually changes
            if month_index != previous_month:
                padded = _pad_landscape(
                    get_landscape(month_index), window_size, device
                )
                previous_month = month_index
            month_out[i] = month_index

            cols, rows = _pixel_from_coords(transform, x_loc, y_loc)
            # Clamp centres to the raster extent.  A centre can only leave the
            # extent through the sub-pixel jitter (out-of-extent cells carry
            # zero probability and are never sampled), so this bites only for
            # animals sitting on the very edge of the raster.
            cols = np.clip(cols, 0, width - 1)
            rows = np.clip(rows, 0, height - 1)
            col_t = torch.as_tensor(cols, device=device)
            row_t = torch.as_tensor(rows, device=device)

            x1 = _crop_windows(padded, col_t, row_t, window_size)
            scalars = scalars_all[i].expand(n_traj, -1)

            out = model((x1, scalars, bearing))
            joint = out[..., 0] + out[..., 1]  # [N, ws, ws]

            # Cells padded with -1 lie outside the raster; -inf removes them
            # from the softmax rather than merely down-weighting them.
            invalid = x1[:, 0] == -1.0
            joint = joint.masked_fill(invalid, float("-inf"))
            probs = torch.softmax(joint.reshape(n_traj, -1), dim=1)

            # One device→host transfer per step; everything downstream (the
            # coordinate arithmetic and the jitter) stays on the CPU.
            sampled = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()
            local_row, local_col = np.divmod(sampled, window_size)

            half = window_size // 2
            new_col = cols - half + local_col
            new_row = rows - half + local_row
            new_x, new_y = _coords_from_pixel(transform, new_col, new_row)

            # Sub-pixel jitter: keeps simulated locations off the pixel-corner
            # lattice.  Mean at the half-cell, sd ~ a quarter of a cell,
            # truncated to the cell (matches simulate_next_step's shape).
            new_x = new_x + _truncated_normal(
                n_traj, size_x / 2, size_x * 0.26, 0.0, size_x
            )
            new_y = new_y + y_sign * _truncated_normal(
                n_traj, size_y / 2, size_y * 0.26, 0.0, size_y
            )

            bearing = torch.tensor(
                np.arctan2(new_y - y_loc, new_x - x_loc),
                dtype=torch.float32, device=device,
            ).unsqueeze(1)

            x_loc, y_loc = new_x, new_y
            xs_out[i], ys_out[i] = new_x, new_y

    # Column-major ravel gives trajectory-major ordering (all steps of
    # trajectory 0, then trajectory 1, ...), matching the id/step columns.
    return pd.DataFrame(
        {
            "trajectory_id": np.repeat(np.arange(n_traj), n_steps),
            "step": np.tile(np.arange(n_steps), n_traj),
            "x": xs_out.ravel(order="F"),
            "y": ys_out.ravel(order="F"),
            "hour": np.tile(hour_t2, n_traj),
            "yday": np.tile(yday_t2, n_traj),
            "month_index": np.tile(month_out, n_traj),
        }
    )


def simulate_trajectories(
    model: torch.nn.Module,
    get_landscape: Callable[[int], list[torch.Tensor]],
    transform: object,
    start_x,
    start_y,
    n_steps: int,
    n_trajectories: int = 1,
    method: str = "batched",
    starting_yday=1.0,
    starting_hour=0.0,
    time_between_steps: float = 1.0,
    window_size: int = 101,
    base_year: int = 2018,
    month_index_fn: Callable[[float], int] | None = None,
    n_workers: int | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Simulate *n_trajectories* trajectories and return them in one DataFrame.

    Parameters
    ----------
    method:
        ``"batched"`` (default)
            All trajectories step together, one batched forward pass per step.
            Fastest by a wide margin, but *starting_yday* and *starting_hour*
            must be scalars.  See :func:`simulate_trajectories_batched`.
        ``"sequential"``
            Plain loop over :func:`simulate_trajectory`.  Slowest, but the
            reference behaviour, and start times may vary per trajectory.
        ``"parallel"``
            Same loop spread over a thread pool.  Threads (not processes)
            because the model, the landscape tensors and any notebook-defined
            ``get_landscape`` closure are shared rather than pickled; the
            speed-up comes from PyTorch releasing the GIL inside its ops, so it
            is modest and depends on how many intra-op threads torch is already
            using.  Results are not reproducible from a seed under this method.
    start_x, start_y, starting_yday, starting_hour:
        Scalars, or sequences of length *n_trajectories* (start times may vary
        only under ``"sequential"``/``"parallel"``).
    n_workers:
        Thread count for ``method="parallel"``.  Defaults to
        ``min(n_trajectories, os.cpu_count())``.
    progress:
        Print a line as each trajectory finishes (sequential/parallel only).

    Returns
    -------
    pd.DataFrame
        Long format with ``trajectory_id`` and ``step`` columns; see
        :func:`simulate_trajectories_batched`.
    """
    n_traj = int(n_trajectories)
    xs = _broadcast_start(start_x, n_traj, "start_x")
    ys = _broadcast_start(start_y, n_traj, "start_y")
    ydays = _broadcast_start(starting_yday, n_traj, "starting_yday")
    hours = _broadcast_start(starting_hour, n_traj, "starting_hour")

    if method == "batched":
        if len(np.unique(ydays)) > 1 or len(np.unique(hours)) > 1:
            raise ValueError(
                "method='batched' steps every trajectory on a shared clock, so "
                "starting_yday/starting_hour must be scalars. Use "
                "method='sequential' or 'parallel' for per-trajectory times."
            )
        return simulate_trajectories_batched(
            model,
            get_landscape=get_landscape,
            transform=transform,
            start_x=xs,
            start_y=ys,
            n_steps=n_steps,
            n_trajectories=n_traj,
            starting_yday=float(ydays[0]),
            starting_hour=float(hours[0]),
            time_between_steps=time_between_steps,
            window_size=window_size,
            base_year=base_year,
            month_index_fn=month_index_fn,
        )

    if method not in ("sequential", "parallel"):
        raise ValueError(
            f"Unknown method {method!r}; expected 'batched', 'sequential' or "
            "'parallel'."
        )

    def _one(j: int) -> pd.DataFrame:
        df = simulate_trajectory(
            model,
            get_landscape=get_landscape,
            transform=transform,
            start_x=float(xs[j]),
            start_y=float(ys[j]),
            n_steps=n_steps,
            starting_yday=float(ydays[j]),
            starting_hour=float(hours[j]),
            time_between_steps=time_between_steps,
            window_size=window_size,
            base_year=base_year,
            month_index_fn=month_index_fn,
        )
        df.insert(0, "step", np.arange(len(df)))
        df.insert(0, "trajectory_id", j)
        if progress:
            print(f"  trajectory {j + 1}/{n_traj} done ({len(df)} steps)")
        return df

    if method == "sequential":
        frames = [_one(j) for j in range(n_traj)]
    else:
        import os

        workers = n_workers or min(n_traj, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_one, range(n_traj)))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_trajectories(
    trajectories: pd.DataFrame | Sequence[pd.DataFrame],
    output_dir: str | Path,
    prefix: str = "deepSSF_sim",
    id_col: str = "trajectory_id",
    split: bool = False,
    date: str | None = None,
    index: bool = False,
) -> list[Path]:
    """Write simulated trajectories to CSV without ever overwriting a file.

    The filename encodes the number of trajectories, the number of steps and
    the date; if that name is already taken a ``_2``, ``_3``, ... run counter is
    appended, so re-running the cell adds files rather than clobbering them.

    Parameters
    ----------
    trajectories:
        A DataFrame (optionally with an *id_col*) or a sequence of DataFrames.
    output_dir:
        Directory to write into; created if it does not exist.
    prefix:
        Leading part of the filename, e.g. ``"deepSSF_pigs"``.
    split:
        Write one file per trajectory instead of a single combined file.
    date:
        Date stamp in the filename; defaults to today.
    index:
        Passed through to ``DataFrame.to_csv``.

    Returns
    -------
    list of Path
        The files written.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    date = date or datetime.now().strftime("%Y-%m-%d")

    if not isinstance(trajectories, pd.DataFrame):
        combined = pd.concat(
            [
                df.assign(**{id_col: df.get(id_col, j)})
                for j, df in enumerate(trajectories)
            ],
            ignore_index=True,
        )
    else:
        combined = trajectories

    has_ids = id_col in combined.columns
    n_traj = int(combined[id_col].nunique()) if has_ids else 1
    n_steps = (
        int(combined.groupby(id_col).size().max()) if has_ids else len(combined)
    )

    def _unique_path(stem: str) -> Path:
        path = out_dir / f"{stem}.csv"
        run = 2
        while path.exists():
            path = out_dir / f"{stem}_{run}.csv"
            run += 1
        return path

    written: list[Path] = []
    if split and has_ids:
        for tid, sub in combined.groupby(id_col, sort=True):
            path = _unique_path(f"{prefix}_id{tid}_{n_steps}steps_{date}")
            sub.to_csv(path, index=index)
            written.append(path)
    else:
        path = _unique_path(f"{prefix}_{n_traj}traj_{n_steps}steps_{date}")
        combined.to_csv(path, index=index)
        written.append(path)

    for path in written:
        print(f"Saved: {path}")
    return written
