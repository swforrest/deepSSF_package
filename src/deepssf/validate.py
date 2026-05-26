"""Validation & diagnostics: assess next-step predictions against observed data."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

from deepssf.utils import subset_raster_with_padding_torch


def _day_to_s2_month(day_of_year: float) -> int:
    """Convert day-of-year to a 1-based month index (1–12), wrapping across years.

    Uses base 2019-01-01.  Multi-year yday values (e.g. 400 = day 35 of year 2)
    are wrapped back into 1–12 so the result can be used directly as an S2 dict
    key: ``f'2019_{month_index:02d}'``.
    """
    base = datetime(2019, 1, 1)
    date = base + timedelta(days=int(day_of_year) - 1)
    month_index = date.month + (date.year - base.year) * 12
    month_index = max(month_index, 1)
    return (month_index - 1) % 12 + 1


def validate_next_step_probs(
    model: torch.nn.Module,
    movement_df: pd.DataFrame,
    get_landscape: Callable[[int], list[torch.Tensor]],
    transform: object,
    window_size: int = 101,
    month_index_fn: Callable[[float], int] | None = None,
    scalar_cols: tuple[str, str, str, str] = (
        "hour_t2_sin",
        "hour_t2_cos",
        "yday_t2_sin",
        "yday_t2_cos",
    ),
    yday_col: str = "yday_t2",
    bearing_col: str = "bearing_tm1",
) -> pd.DataFrame:
    """Evaluate next-step prediction probabilities for an observed trajectory.

    For each observed step the model predicts habitat-selection, movement, and
    joint log-probability surfaces over a local ``window_size × window_size``
    crop.  The probability at the *observed* next location is extracted and
    returned.

    Parameters
    ----------
    model:
        Fitted ConvJointModel (in eval mode).
    movement_df:
        DataFrame with columns ``x1_``, ``y1_``, ``x2_``, ``y2_``,
        ``<scalar_cols>``, ``<yday_col>``, ``<bearing_col>``.
    get_landscape:
        Callable ``(month_index: int) -> list[Tensor]`` returning one 2-D
        tensor per spatial channel for the given month.  The month index
        convention is determined by *month_index_fn* (default: 1-based 1–12).
        Pass the same *get_landscape* and *month_index_fn* as used in
        ``simulate_trajectory`` to share a single landscape loader.
    transform:
        Rasterio ``Affine`` transform shared by all landscape rasters.
    window_size:
        Side length of the spatial crop in pixels (should match training).
    month_index_fn:
        Callable ``(yday: float) -> int``.  Defaults to ``_day_to_s2_month``
        which returns 1-based month indices 1–12 (suitable for S2 dicts keyed
        by ``'2019_MM'``).
    scalar_cols:
        Column names for the four cyclic scalar inputs to the model:
        ``(sin_hour, cos_hour, sin_yday, cos_yday)``.
    yday_col:
        Column containing day-of-year values (for month lookup).
    bearing_col:
        Column containing the bearing of the previous step (radians).

    Returns
    -------
    pd.DataFrame
        Copy of *movement_df* with three new columns appended:
        ``habitat_prob``, ``move_prob``, ``next_step_prob``.
        Row 0 is always 0.0 (no previous bearing).  Rows where the observed
        next step falls outside the local crop are NaN.
    """
    _month_fn = month_index_fn if month_index_fn is not None else _day_to_s2_month

    n = len(movement_df)
    # Initialise all probabilities to zero; row 0 is never updated (no prev bearing)
    habitat_probs = np.zeros(n)
    move_probs = np.zeros(n)
    next_step_probs = np.zeros(n)

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Start at row 1: row 0 has no previous bearing, so its prob stays 0.0
        for i in range(1, n):
            sample = movement_df.iloc[i]

            x, y = float(sample["x1_"]), float(sample["y1_"])
            x2_geo, y2_geo = float(sample["x2_"]), float(sample["y2_"])

            # Geographic → pixel coords for the next step
            px2, py2 = ~transform * (x2_geo, y2_geo)  # type: ignore[operator]

            # Select the correct monthly landscape based on the departure date
            yday = float(sample[yday_col])
            month_index = _month_fn(yday)
            landscape_rasters = get_landscape(month_index)

            # Crop a spatial patch at the departure location for each raster channel
            results = [
                subset_raster_with_padding_torch(
                    rt, x=x, y=y, window_size=window_size, transform=transform  # type: ignore[arg-type]
                )
                for rt in landscape_rasters
            ]
            subset_tensors = [r[0] for r in results]
            origin_xs = [r[1] for r in results]  # top-left column of each patch
            origin_ys = [r[2] for r in results]  # top-left row of each patch

            # Stack channels → [1, C, H, W]
            x1 = torch.stack(list(subset_tensors), dim=0).unsqueeze(0).to(device)

            # Scalar inputs: [1, 4]
            scalars = torch.tensor(
                [float(sample[c]) for c in scalar_cols], dtype=torch.float32
            ).unsqueeze(0).to(device)

            # Previous bearing: [1, 1]
            bearing = torch.tensor(
                [[float(sample[bearing_col])]], dtype=torch.float32
            ).to(device)

            out = model((x1, scalars, bearing))

            # Extract log-prob surfaces [H, W] (already log-normalised by the model)
            hab_log = out[0, :, :, 0].detach().cpu().numpy()
            move_log = out[0, :, :, 1].detach().cpu().numpy()

            # Convert the observed next-step global pixel to local crop coordinates
            px2_local = int(round(float(px2) - origin_xs[0]))
            py2_local = int(round(float(py2) - origin_ys[0]))

            # If the next step falls outside the crop window, record NaN and skip
            if not (0 <= px2_local < window_size and 0 <= py2_local < window_size):
                habitat_probs[i] = np.nan
                move_probs[i] = np.nan
                next_step_probs[i] = np.nan
                continue

            # Convert log-probs to probs; renormalise the joint surface before indexing
            hab_exp = np.exp(hab_log)
            move_exp = np.exp(move_log)
            step_exp = np.exp(hab_log + move_log)
            step_exp_norm = step_exp / np.sum(step_exp)

            # Record the probability assigned to the observed next-step pixel
            habitat_probs[i] = hab_exp[py2_local, px2_local]
            move_probs[i] = move_exp[py2_local, px2_local]
            next_step_probs[i] = step_exp_norm[py2_local, px2_local]

    result = movement_df.copy()
    result["habitat_prob"] = habitat_probs
    result["move_prob"] = move_probs
    result["next_step_prob"] = next_step_probs
    return result