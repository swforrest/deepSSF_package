"""Data preparation: load environmental layers and build PyTorch datasets.

This module replaces the R-based data-prep workflow.  It covers:

* Loading raster layers (Sentinel-2 monthly composites + static covariates).
* Pre-computing pixel coordinates and temporal indices for every GPS fix.
* ``MovementDataset`` – a ``torch.utils.data.Dataset`` that yields spatial
  patches, scalar features, and bearing for each observed movement step.

Typical usage::

    from deepssf.data import MovementDataset, load_environmental_layers

    env_layers, transform = load_environmental_layers(layer_paths)
    dataset = MovementDataset(movement_df, layer_paths, window_size=101)
"""

from __future__ import annotations

import glob
import os
import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

from deepssf.utils import subset_layer_vectorized

# ---------------------------------------------------------------------------
# Filename / date helpers
# ---------------------------------------------------------------------------

def extract_year_month_regex(filename: str) -> str | None:
    """Return the ``'YYYY_MM'`` substring from *filename*, or ``None``."""
    match = re.search(r'(\d{4}_\d{2})', filename)
    return match.group(1) if match else None


def extract_datetime_regex(filename: str) -> datetime | str:
    """Return a ``datetime`` (day 15) for the ``'YYYY_MM'`` in *filename*."""
    match = re.search(r'(\d{4}_\d{2})', filename)
    if match:
        year, month = match.group(1).split("_")
        return datetime(int(year), int(month), 15)
    return "Datetime string not found in filename"


def day_to_month_index(day_of_year: int) -> int:
    """Convert day-of-year to a 1-based month index for S2 data selection.

    Parameters
    ----------
    day_of_year:
        Integer day of the year (1–365).

    Returns
    -------
    int
        Month number (1–12+) relative to 2019-01-01.
    """
    base = datetime(2019, 1, 1)
    date = base + timedelta(days=int(day_of_year % 365) - 1)
    year_diff = date.year - base.year
    return max(date.month + year_diff * 12, 1)


# ---------------------------------------------------------------------------
# Raster loading
# ---------------------------------------------------------------------------

def load_s2_data(s2_dir: str) -> tuple[dict[str, np.ndarray], Any]:
    """Load all Sentinel-2 monthly composite TIFFs from *s2_dir*.

    Each file must match ``S2_*.tif`` and contain a ``YYYY_MM`` date string.
    Values are divided by 10 000 (DN → surface reflectance); NaNs → 0.

    Parameters
    ----------
    s2_dir:
        Directory containing S2 TIFF files.

    Returns
    -------
    s2_data_dict:
        ``{'YYYY_MM': ndarray([bands, H, W]), …}``
    raster_transform:
        Affine transform of the last file opened.
    """
    print(f"Loading S2 data from {s2_dir}")
    tif_files = glob.glob(os.path.join(s2_dir, "S2_*.tif"))
    print(f"Found {len(tif_files)} S2 TIFF files")

    s2_data_dict: dict[str, np.ndarray] = {}
    raster_transform: Any = None

    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        date_str = extract_year_month_regex(filename)
        if date_str is None:
            print(f"  Skipping {filename}: no YYYY_MM pattern found")
            continue
        print(f"  Loading {filename} → {date_str}")

        with rasterio.open(tif_file) as src:
            data = src.read()
            n_nan = np.isnan(data).sum()
            print(f"    NaN values: {n_nan} ({n_nan / data.size:.4%})")
            # Sentinel-2 DNs are stored as integers ×10 000; divide to get
            # surface reflectance in [0, 1] (approximately).
            data = np.nan_to_num(data, nan=0) / 10_000.0
            s2_data_dict[date_str] = data
            raster_transform = src.transform

    if raster_transform is None:
        raise ValueError(
            "No S2 TIFF files found or readable.  "
            "Check s2_dir and filename pattern (S2_*.tif)."
        )

    print(f"Loaded {len(s2_data_dict)} S2 datasets")
    return s2_data_dict, raster_transform


def load_environmental_layers(
    layer_paths: dict,
) -> tuple[dict, Any]:
    """Load all environmental layers from a paths dictionary.

    Parameters
    ----------
    layer_paths:
        Dictionary of ``{name: path}``.  Special keys:

        * ``'s2_dir'`` – passed to :func:`load_s2_data`.
        * Any other ``.tif`` path – loaded via rasterio and scaled to [0, 1].
        * Any other path – loaded as a NumPy ``.npy`` file (memory-mapped).

    Returns
    -------
    environmental_layers:
        Dict of layer names → arrays (S2 gets a nested ``{'YYYY_MM': arr}``).
    raster_transform:
        Affine transform from the first TIFF or S2 directory processed.
    """
    layers: dict = {}
    raster_transform: Any = None

    # S2 directory gets its own loader that handles multi-date file naming
    if "s2_dir" in layer_paths:
        layers["s2"], raster_transform = load_s2_data(layer_paths["s2_dir"])

    for name, path in layer_paths.items():
        if name in ("s2", "s2_dir"):
            continue

        if isinstance(path, str) and path.endswith(".tif"):
            with rasterio.open(path) as src:
                data = src.read()
                data = np.nan_to_num(data, nan=0)
                lo, hi = float(np.nanmin(data)), float(np.nanmax(data))
                print(f"Layer '{name}': min={lo}, max={hi} → scaled to [0, 1]")
                # Min-max scale to [0, 1] so all static layers share the same range
                data = (data - lo) / hi
                # Drop the band dimension for single-band TIFFs → [H, W]
                if data.shape[0] == 1:
                    data = data[0]
                layers[name] = data
                if raster_transform is None:
                    raster_transform = src.transform
        else:
            # .npy files loaded with memory-mapping to avoid loading the full
            # array into RAM before the crop step in __getitem__
            layers[name] = np.load(path, mmap_mode="r")

    if raster_transform is None:
        raise ValueError(
            "raster_transform could not be set.  "
            "Provide at least one TIFF path in layer_paths."
        )

    return layers, raster_transform


# ---------------------------------------------------------------------------
# Coordinate pre-computation
# ---------------------------------------------------------------------------

def precompute_coordinates_and_months(
    movement_df: pd.DataFrame,
    raster_transform: Any,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Pre-compute pixel coordinates and S2 month keys for every GPS fix.

    Parameters
    ----------
    movement_df:
        DataFrame with columns ``x1_``, ``y1_``, and ``t1_`` (ISO-8601
        timestamp string).
    raster_transform:
        Rasterio affine transform for geographic → pixel conversion.

    Returns
    -------
    pixel_coords:
        List of ``(px, py)`` column/row tuples.
    s2_months:
        List of ``'YYYY_MM'`` strings aligned to *pixel_coords*.
    """
    pixel_coords: list[tuple[int, int]] = []
    s2_months: list[str] = []

    for _, row in movement_df.iterrows():
        px, py = ~raster_transform * (row["x1_"], row["y1_"])  # type: ignore[operator]
        pixel_coords.append((int(np.floor(px)), int(np.floor(py))))
        dt = datetime.fromisoformat(str(row["t1_"]).replace("Z", "+00:00"))
        s2_months.append(f"{dt.year}_{dt.month:02d}")

    return pixel_coords, s2_months


# ---------------------------------------------------------------------------
# Landscape cropping utility
# ---------------------------------------------------------------------------

def landscape_crop(
    movement_df: pd.DataFrame,
    environmental_layers: dict,
    raster_transform: Any,
    window_size: int,
) -> dict[str, torch.Tensor]:
    """Crop all environmental layers to the bounding box of the GPS track.

    Adds a half-window-size buffer so patches near the track edge are covered.

    Parameters
    ----------
    movement_df:
        DataFrame with columns ``x1_`` and ``y1_``.
    environmental_layers:
        Output of :func:`load_environmental_layers`.
    raster_transform:
        Rasterio affine transform for coordinate conversion.
    window_size:
        Patch size used during training (determines the buffer).

    Returns
    -------
    dict[str, torch.Tensor]
        Cropped layers as float tensors.
    """
    coords = []
    for _, row in movement_df.iterrows():
        px, py = ~raster_transform * (row["x1_"], row["y1_"])  # type: ignore[operator]
        coords.append((int(np.floor(px)), int(np.floor(py))))

    min_px = min(c[0] for c in coords)
    max_px = max(c[0] for c in coords)
    min_py = min(c[1] for c in coords)
    max_py = max(c[1] for c in coords)
    buf = window_size // 2

    cropped: dict[str, torch.Tensor] = {}
    for name, arr in environmental_layers.items():
        if arr.ndim == 2:
            patch = arr[
                min_py - buf : max_py + buf + 1,
                min_px - buf : max_px + buf + 1,
            ]
        else:
            patch = arr[
                :,
                min_py - buf : max_py + buf + 1,
                min_px - buf : max_px + buf + 1,
            ]
        cropped[name] = torch.from_numpy(patch.copy())

    print(
        f"Cropped layers to extent: "
        f"({min_px - buf}, {min_py - buf}) → ({max_px + buf}, {max_py + buf})"
    )
    return cropped


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

def _select_s2_month(s2_dict: dict, requested: str) -> np.ndarray:
    """Return the S2 array for *requested* month, with year ±1 fallbacks."""
    if requested in s2_dict:
        return s2_dict[requested]

    year_str, month_str = requested.split("_")
    year = int(year_str)

    for candidate in (year - 1, year + 1):
        key = f"{candidate}_{month_str}"
        if key in s2_dict:
            return s2_dict[key]

    return next(iter(s2_dict.values()))


_DEFAULT_SCALAR_COLS = [
    "hour_t1_sin1",
    "hour_t1_cos1",
    "yday_t1_sin1",
    "yday_t1_cos1",
    "dt_hour",
]


class MovementDataset(Dataset):
    """Dataset of GPS movement steps with matched spatial patches.

    Each item is a 5-tuple:

    * ``spatial_data``    – [C, H, W] float tensor centred on (x1, y1).
    * ``scalar_to_grid``  – [S] float tensor (scalar covariates) for
      broadcasting into spatial maps.
    * ``bearing_tm1``     – [1] float tensor of the previous step's bearing.
    * ``next_step_pixel`` – ``(col, row)`` of the next fix within the patch.
    * ``raster_transform`` – affine transform (passed through for downstream
      use).

    Parameters
    ----------
    movement_df:
        DataFrame with columns: ``x1_``, ``y1_``, ``x2_``, ``y2_``,
        ``t1_``, *bearing_col*, and all *scalar_cols*.
    layer_paths:
        Passed directly to :func:`load_environmental_layers`.
    window_size:
        Edge length of the spatial patch in pixels (typically 101).
    scalar_cols:
        Column names to use as scalar inputs to the model.  Defaults to
        ``['hour_t1_sin1', 'hour_t1_cos1', 'yday_t1_sin1', 'yday_t1_cos1',
        'dt_hour']``.
    bearing_col:
        Column name of the step bearing (radians).  Shifted by one row to
        give the *previous* step's bearing.  Default: ``'bearing'``.
    """

    def __init__(
        self,
        movement_df: pd.DataFrame,
        layer_paths: dict,
        window_size: int,
        scalar_cols: list[str] | None = None,
        bearing_col: str = "bearing",
    ) -> None:
        self.movement_df = movement_df.reset_index(drop=True)
        self.window_size = window_size

        self.environmental_layers, self.raster_transform = load_environmental_layers(
            layer_paths
        )

        cols = scalar_cols if scalar_cols is not None else _DEFAULT_SCALAR_COLS
        # Pre-extract scalar columns as a tensor: indexed by row in __getitem__
        self.scalar_to_grid_data = torch.from_numpy(
            movement_df[cols].values
        ).float()

        # Shift bearing by one row so each step receives the *previous* bearing
        # as input (directional persistence). Row 0 gets 0 (no prior heading).
        bearing_raw: pd.Series = (
            movement_df[bearing_col].astype(float).shift(1).fillna(0)
        )
        self.bearing_tm1 = torch.from_numpy(
            bearing_raw.to_numpy()
        ).unsqueeze(1).float()

        # Pre-compute pixel coordinates for every departure and arrival location
        # so __getitem__ avoids per-sample geo-transform lookups at batch time.
        self.x1y1_pixel_coords: list[tuple[int, int]] = []
        self.x2y2_pixel_coords_raw: list[tuple[int, int]] = []
        self.s2_months: list[str] = []

        for _, row in self.movement_df.iterrows():
            px1, py1 = ~self.raster_transform * (  # type: ignore[operator]
                row["x1_"], row["y1_"]
            )
            self.x1y1_pixel_coords.append(
                (int(np.floor(px1)), int(np.floor(py1)))
            )

            px2, py2 = ~self.raster_transform * (  # type: ignore[operator]
                row["x2_"], row["y2_"]
            )
            self.x2y2_pixel_coords_raw.append(
                (int(np.floor(px2)), int(np.floor(py2)))
            )

            # Store 'YYYY_MM' key for selecting the correct S2 monthly composite
            dt = datetime.fromisoformat(str(row["t1_"]).replace("Z", "+00:00"))
            self.s2_months.append(f"{dt.year}_{dt.month:02d}")

        self.n_samples = len(self.movement_df)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> tuple:
        px1, py1 = self.x1y1_pixel_coords[index]
        px2, py2 = self.x2y2_pixel_coords_raw[index]
        selected_month = self.s2_months[index]

        cropped_layers = []
        col_start = row_start = 0

        # S2 is processed first so col_start / row_start reflect its crop origin,
        # which is used below to compute the next-step pixel in local coordinates.
        if "s2" in self.environmental_layers:
            # Select the best-matching monthly composite (with year ±1 fallback)
            s2_arr = _select_s2_month(
                self.environmental_layers["s2"], selected_month
            )
            s2_crop, col_start, row_start = subset_layer_vectorized(
                s2_arr, px1, py1, self.window_size
            )
            cropped_layers.append(s2_crop)

        for name, layer_data in self.environmental_layers.items():
            if name == "s2":
                continue
            crop, col_start, row_start = subset_layer_vectorized(
                layer_data, px1, py1, self.window_size
            )
            # Ensure 2-D layers become [1, H, W] so all layers have a channel dim
            if crop.ndim == 2:
                crop = crop.unsqueeze(0)
            cropped_layers.append(crop)

        # Concatenate all channel groups into a single [C, H, W] tensor
        spatial_data = (
            torch.cat(cropped_layers, dim=0)
            if len(cropped_layers) > 1
            else cropped_layers[0]
        )

        # Express the arrival pixel in the coordinate frame of the local crop.
        # The loss function indexes into the [H, W] probability surface using these
        # values, so they must be in [0, window_size). Steps outside that range
        # will raise an out-of-bounds error — call filter_steps_by_window first.
        next_step_pixel = (px2 - col_start, py2 - row_start)

        return (
            spatial_data,
            self.scalar_to_grid_data[index],
            self.bearing_tm1[index],
            next_step_pixel,
            self.raster_transform,
        )


# ---------------------------------------------------------------------------
# Raw telemetry → step format
# ---------------------------------------------------------------------------

def prepare_movement_df(
    df: pd.DataFrame,
    id_col: str = "id",
    time_col: str = "time",
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Convert a raw telemetry DataFrame to the step format expected by
    :class:`MovementDataset`.

    The raw format is one row per GPS fix with columns ``id``, ``time``
    (ISO-8601), ``x``, ``y``.  This function computes:

    * ``x1_, y1_`` – departure location (current fix).
    * ``x2_, y2_`` – arrival location (next fix).
    * ``t1_`` – ISO-8601 timestamp of the departure (passed to
      :class:`MovementDataset` for S2 month look-up).
    * ``bearing`` – step bearing in radians (arctan2 of dy/dx in projected
      CRS).  Shifted inside :class:`MovementDataset` to give the *previous*
      step's bearing.
    * ``dt_hour`` – time elapsed between consecutive fixes (hours).
    * ``hour_t1_sin1``, ``hour_t1_cos1`` – cyclic encoding of the hour of
      ``t1_``.
    * ``yday_t1_sin1``, ``yday_t1_cos1`` – cyclic encoding of the day-of-year
      of ``t1_``.

    The last GPS fix for each individual is dropped (no arrival location).
    Individuals are processed independently and the results are concatenated.

    Parameters
    ----------
    df:
        Raw telemetry DataFrame.
    id_col, time_col, x_col, y_col:
        Column names (defaults: ``'id'``, ``'time'``, ``'x'``, ``'y'``).

    Returns
    -------
    pd.DataFrame
        Ready to pass directly to :class:`MovementDataset`.
    """
    frames = []

    for _, group in df.groupby(id_col, sort=False):
        g = group.sort_values(time_col).reset_index(drop=True)
        n = len(g)
        if n < 2:
            continue

        # Each consecutive pair of fixes defines one step: dep → arr
        dep = g.iloc[:-1]       # departure rows (all but last)
        arr = g.iloc[1:]        # arrival rows  (all but first)

        # Parse timestamps with UTC to handle mixed timezone strings
        times_dep = pd.to_datetime(dep[time_col].values, utc=True)
        times_arr = pd.to_datetime(arr[time_col].values, utc=True)

        # Raw displacements in CRS units (e.g. metres for a projected CRS)
        dx = arr[x_col].values - dep[x_col].values
        dy = arr[y_col].values - dep[y_col].values

        # Fractional hour (e.g. 14.5 = 14:30) and integer day-of-year at departure
        hours = times_dep.hour + times_dep.minute / 60.0
        ydays = times_dep.day_of_year.astype(float)

        # Step bearing in radians: 0 = east, π/2 = north (standard arctan2 convention)
        bearings = np.arctan2(dy, dx)
        # Previous bearing: row 0 has no predecessor so it is set to 0 (east).
        # MovementDataset shifts this column by one row again, so in practice
        # the model never sees a meaningful bearing for the very first step.
        bearing_tm1 = np.empty_like(bearings)
        bearing_tm1[0] = 0.0
        bearing_tm1[1:] = bearings[:-1]

        out = pd.DataFrame(
            {
                "id":            dep[id_col].values,
                "t1_":           dep[time_col].values,
                "x1_":           dep[x_col].values,
                "y1_":           dep[y_col].values,
                "t2_":           arr[time_col].values,
                "x2_":           arr[x_col].values,
                "y2_":           arr[y_col].values,
                "dx":            dx,
                "dy":            dy,
                "bearing":       bearings,
                "bearing_tm1":   bearing_tm1,
                "dt_hour":       (times_arr - times_dep).total_seconds() / 3600.0,
                "hour_t1":       hours,
                "yday_t1":       ydays,
                "hour_t1_sin1":  np.sin(2 * np.pi * hours / 24),
                "hour_t1_cos1":  np.cos(2 * np.pi * hours / 24),
                "yday_t1_sin1":  np.sin(2 * np.pi * ydays / 365.25),
                "yday_t1_cos1":  np.cos(2 * np.pi * ydays / 365.25),
            }
        )
        frames.append(out)

    return pd.concat(frames, ignore_index=True)


def filter_steps_by_window(
    df: pd.DataFrame,
    window_size: int,
    pixel_size: float,
) -> pd.DataFrame:
    """Drop steps whose displacement exceeds the spatial window.

    Steps that land outside the ``window_size × window_size`` crop centred on
    the departure location will produce an out-of-bounds index in the loss
    function.  Call this after :func:`prepare_movement_df` and before
    :func:`make_dataloaders`.

    Parameters
    ----------
    df:
        Output of :func:`prepare_movement_df`; must contain ``dx`` and ``dy``
        columns (displacement in CRS units, typically metres).
    window_size:
        Side length of the spatial crop in pixels (must match the value passed
        to :func:`make_dataloaders`).
    pixel_size:
        Size of one raster pixel in the same CRS units as ``dx``/``dy``
        (e.g. 25 for a 25 m resolution raster).

    Returns
    -------
    pd.DataFrame
        Filtered copy of *df* with out-of-range steps removed.
    """
    # The crop is centred on the departure pixel; the farthest reachable pixel
    # in each direction is (window_size - 1) / 2 pixels = half_extent metres.
    # Steps beyond this cannot be indexed within the [H, W] probability surface.
    half_extent = (window_size - 1) * pixel_size / 2
    mask = (
        (df["dx"].abs() < half_extent) &
        (df["dy"].abs() < half_extent)
    )
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Convenience: CSV → split DataLoaders
# ---------------------------------------------------------------------------

def make_dataloaders(
    csv_path: str | None = None,
    layer_paths: dict | None = None,
    window_size: int = 101,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 0,
    scalar_cols: list[str] | None = None,
    bearing_col: str = "bearing",
    prepare: bool = False,
    df: pd.DataFrame | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train / val / test DataLoaders from a CSV path or prepared DataFrame.

    Mirrors the notebook pattern::

        dataset = MovementDataset(df, layer_paths, window_size)
        train_ds, val_ds, test_ds = random_split(dataset, [0.8, 0.1, 0.1])
        dl_train = DataLoader(train_ds, batch_size=32, shuffle=True)

    Parameters
    ----------
    csv_path:
        Path to the movement CSV file.  Either *csv_path* or *df* must be
        provided.
    layer_paths:
        Dict passed to :func:`load_environmental_layers`.
    window_size:
        Spatial patch size in pixels.
    batch_size:
        Number of samples per batch.
    train_split:
        Fraction of data for training (default 0.8).
    val_split:
        Fraction of data for validation (default 0.1).  The remaining
        ``1 - train_split - val_split`` goes to the test set.
    num_workers:
        Worker processes for data loading.  Use 0 inside Jupyter notebooks.
    scalar_cols:
        Forwarded to :class:`MovementDataset`.
    bearing_col:
        Forwarded to :class:`MovementDataset`.
    prepare:
        If ``True``, call :func:`prepare_movement_df` on the source data
        before building the dataset.
    df:
        Pre-loaded DataFrame.  If provided, *csv_path* is ignored.

    Returns
    -------
    dataloader_train, dataloader_val, dataloader_test : DataLoader
    """
    if df is None and csv_path is None:
        raise ValueError("Provide either csv_path or df.")
    if layer_paths is None:
        raise ValueError("layer_paths is required.")
    source_df = df if df is not None else pd.read_csv(csv_path)
    # prepare=True is a convenience flag for raw CSVs; for finer control
    # (e.g. calling filter_steps_by_window between steps) pass a pre-processed df
    if prepare:
        source_df = prepare_movement_df(source_df)
    dataset = MovementDataset(
        source_df,
        layer_paths,
        window_size,
        scalar_cols=scalar_cols,
        bearing_col=bearing_col,
    )

    # Remainder after train + val goes to the test set
    test_split = 1.0 - train_split - val_split
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_split, val_split, test_split]
    )

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    # Training set is shuffled each epoch; val/test are not
    dl_train = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    dl_val   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    dl_test  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    return dl_train, dl_val, dl_test