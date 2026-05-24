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
                data = (data - lo) / hi
                if data.shape[0] == 1:
                    data = data[0]
                layers[name] = data
                if raster_transform is None:
                    raster_transform = src.transform
        else:
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
        self.scalar_to_grid_data = torch.from_numpy(
            movement_df[cols].values
        ).float()

        bearing_raw: pd.Series = (
            movement_df[bearing_col].astype(float).shift(1).fillna(0)
        )
        self.bearing_tm1 = torch.from_numpy(
            bearing_raw.to_numpy()
        ).unsqueeze(1).float()

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

        if "s2" in self.environmental_layers:
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
            if crop.ndim == 2:
                crop = crop.unsqueeze(0)
            cropped_layers.append(crop)

        spatial_data = (
            torch.cat(cropped_layers, dim=0)
            if len(cropped_layers) > 1
            else cropped_layers[0]
        )

        next_step_pixel = (px2 - col_start, py2 - row_start)

        return (
            spatial_data,
            self.scalar_to_grid_data[index],
            self.bearing_tm1[index],
            next_step_pixel,
            self.raster_transform,
        )


# ---------------------------------------------------------------------------
# Convenience: CSV → split DataLoaders
# ---------------------------------------------------------------------------

def make_dataloaders(
    csv_path: str,
    layer_paths: dict,
    window_size: int = 101,
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    num_workers: int = 0,
    scalar_cols: list[str] | None = None,
    bearing_col: str = "bearing",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Read a movement CSV and return train / val / test DataLoaders.

    Mirrors the notebook pattern::

        dataset = MovementDataset(df, layer_paths, window_size)
        train_ds, val_ds, test_ds = random_split(dataset, [0.8, 0.1, 0.1])
        dl_train = DataLoader(train_ds, batch_size=32, shuffle=True)

    Parameters
    ----------
    csv_path:
        Path to the movement CSV file.
    layer_paths:
        Dict passed to :func:`load_environmental_layers`, e.g.::

            {
                's2_dir': 'path/to/s2/',
                'slope': 'path/to/slope.tif',
            }

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

    Returns
    -------
    dataloader_train, dataloader_val, dataloader_test : DataLoader
    """
    df = pd.read_csv(csv_path)
    dataset = MovementDataset(
        df,
        layer_paths,
        window_size,
        scalar_cols=scalar_cols,
        bearing_col=bearing_col,
    )

    test_split = 1.0 - train_split - val_split
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_split, val_split, test_split]
    )

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    dl_train = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    dl_val   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    dl_test  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    return dl_train, dl_val, dl_test