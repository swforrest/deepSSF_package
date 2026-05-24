"""Shared internal helpers used across the package.

Things that don't belong to one stage (device detection, raster-window
extraction, circular-time decoding, GIF creation).
Not part of the public API by default.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on hardware availability."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Raster window extraction
# ---------------------------------------------------------------------------

def subset_raster_with_padding_torch(
    raster_tensor: torch.Tensor,
    x: float,
    y: float,
    window_size: int,
    transform,
) -> tuple[torch.Tensor, int, int]:
    """Extract a square window from a 2-D raster tensor, padding with -1.

    Parameters
    ----------
    raster_tensor:
        2-D float tensor [height, width].
    x, y:
        Geographic coordinates.
    window_size:
        Edge length of the square window (typically odd).
    transform:
        Affine transform supporting inversion via ``~transform * (x, y)``.

    Returns
    -------
    subset:
        [window_size, window_size] tensor padded with -1 where out-of-bounds.
    col_start, row_start:
        Top-left pixel coordinates of the window in the source raster.
    """
    px, py = ~transform * (x, y)
    px, py = int(np.floor(px)), int(np.floor(py))

    half = window_size // 2
    row_start = py - half
    row_stop = py + half + 1
    col_start = px - half
    col_stop = px + half + 1

    subset = torch.full((window_size, window_size), -1.0, dtype=raster_tensor.dtype)

    vr0 = max(0, row_start)
    vr1 = min(raster_tensor.shape[0], row_stop)
    vc0 = max(0, col_start)
    vc1 = min(raster_tensor.shape[1], col_stop)

    sr0 = vr0 - row_start
    sr1 = sr0 + (vr1 - vr0)
    sc0 = vc0 - col_start
    sc1 = sc0 + (vc1 - vc0)

    subset[sr0:sr1, sc0:sc1] = raster_tensor[vr0:vr1, vc0:vc1]
    return subset, col_start, row_start


def subset_raster_all_bands_torch(
    raster_tensor: torch.Tensor,
    x: float,
    y: float,
    window_size: int,
    transform,
) -> tuple[torch.Tensor, int, int]:
    """Extract a square window from a multi-band raster tensor, padding with -1.

    Parameters
    ----------
    raster_tensor:
        3-D float tensor [bands, height, width].
    x, y:
        Geographic coordinates.
    window_size:
        Edge length of the square window.
    transform:
        Affine transform.

    Returns
    -------
    subset:
        [bands, window_size, window_size] tensor padded with -1 out-of-bounds.
    col_start, row_start:
        Top-left pixel coordinates of the window in the source raster.
    """
    px, py = ~transform * (x, y)
    px, py = int(np.floor(px)), int(np.floor(py))

    half = window_size // 2
    row_start = py - half
    row_stop = py + half + 1
    col_start = px - half
    col_stop = px + half + 1

    n_bands = raster_tensor.shape[0]
    subset = torch.full((n_bands, window_size, window_size), -1.0, dtype=raster_tensor.dtype)

    vr0 = max(0, row_start)
    vr1 = min(raster_tensor.shape[1], row_stop)
    vc0 = max(0, col_start)
    vc1 = min(raster_tensor.shape[2], col_stop)

    sr0 = vr0 - row_start
    sr1 = sr0 + (vr1 - vr0)
    sc0 = vc0 - col_start
    sc1 = sc0 + (vc1 - vc0)

    subset[:, sr0:sr1, sc0:sc1] = raster_tensor[:, vr0:vr1, vc0:vc1]
    return subset, col_start, row_start


def subset_raster_with_padding_npy(
    raster_npy: np.ndarray,
    x: float,
    y: float,
    window_size: int,
    transform,
) -> tuple[np.ndarray, int, int]:
    """Extract a square window from a 2-D NumPy raster array, padding with -1.

    Parameters
    ----------
    raster_npy:
        2-D array [height, width].
    x, y:
        Geographic coordinates.
    window_size:
        Edge length of the square window.
    transform:
        Affine transform.

    Returns
    -------
    subset:
        [window_size, window_size] array padded with -1 where out-of-bounds.
    col_start, row_start:
        Top-left pixel coordinates of the window in the source raster.
    """
    px, py = ~transform * (x, y)
    px, py = int(np.floor(px)), int(np.floor(py))

    half = window_size // 2
    row_start = py - half
    row_stop = py + half + 1
    col_start = px - half
    col_stop = px + half + 1

    subset = np.full((window_size, window_size), -1.0, dtype=raster_npy.dtype)

    vr0 = max(0, row_start)
    vr1 = min(raster_npy.shape[0], row_stop)
    vc0 = max(0, col_start)
    vc1 = min(raster_npy.shape[1], col_stop)

    sr0 = vr0 - row_start
    sr1 = sr0 + (vr1 - vr0)
    sc0 = vc0 - col_start
    sc1 = sc0 + (vc1 - vc0)

    subset[sr0:sr1, sc0:sc1] = raster_npy[vr0:vr1, vc0:vc1]
    return subset, col_start, row_start


def subset_layer_vectorized(
    layer_data: np.ndarray,
    px: int,
    py: int,
    window_size: int,
) -> tuple[torch.Tensor, int, int]:
    """Efficient patch extraction from a 2-D or 3-D NumPy array.

    Accepts pixel coordinates directly (no geo transform needed), pads with -1
    where the window extends outside the array, and returns a float tensor.

    Parameters
    ----------
    layer_data:
        2-D [H, W] or 3-D [bands, H, W] array.
    px, py:
        Pixel column and row (already converted from geographic coordinates).
    window_size:
        Edge length of the square window.

    Returns
    -------
    patch:
        Float tensor of shape [window_size, window_size] or
        [bands, window_size, window_size].
    col_start, row_start:
        Top-left pixel coordinates of the window.
    """
    half = window_size // 2
    row_start = py - half
    row_stop = py + half + 1
    col_start = px - half
    col_stop = px + half + 1

    if layer_data.ndim == 2:
        height, width = layer_data.shape
        subset = np.full((window_size, window_size), -1.0, dtype=layer_data.dtype)
        vr0, vr1 = max(0, row_start), min(height, row_stop)
        vc0, vc1 = max(0, col_start), min(width, col_stop)
        if vr0 < vr1 and vc0 < vc1:
            sr0 = vr0 - row_start
            sc0 = vc0 - col_start
            subset[sr0:sr0 + (vr1 - vr0), sc0:sc0 + (vc1 - vc0)] = layer_data[vr0:vr1, vc0:vc1]
    else:
        n_bands, height, width = layer_data.shape
        subset = np.full((n_bands, window_size, window_size), -1.0, dtype=layer_data.dtype)
        vr0, vr1 = max(0, row_start), min(height, row_stop)
        vc0, vc1 = max(0, col_start), min(width, col_stop)
        if vr0 < vr1 and vc0 < vc1:
            sr0 = vr0 - row_start
            sc0 = vc0 - col_start
            subset[:, sr0:sr0 + (vr1 - vr0), sc0:sc0 + (vc1 - vc0)] = layer_data[:, vr0:vr1, vc0:vc1]

    return torch.from_numpy(subset.copy()).float(), col_start, row_start


# ---------------------------------------------------------------------------
# Circular time decoding
# ---------------------------------------------------------------------------

def recover_hour(sin_term: float | np.ndarray, cos_term: float | np.ndarray) -> float | np.ndarray:
    """Recover hour of day (0–24) from its sine/cosine circular encoding."""
    theta = np.arctan2(sin_term, cos_term)
    return (24 * theta) / (2 * np.pi) % 24


def recover_yday(sin_term: float | np.ndarray, cos_term: float | np.ndarray) -> float | np.ndarray:
    """Recover day of year (0–365) from its sine/cosine circular encoding."""
    theta = np.arctan2(sin_term, cos_term)
    return (365.25 * theta) / (2 * np.pi) % 365.25


# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def clear_memory(device: str | None = None) -> None:
    """Clear CUDA or MPS memory cache (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and torch.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# GIF creation
# ---------------------------------------------------------------------------

def _extract_epoch_index(filename: str) -> int:
    """Extract epoch number from a filename like ``..._index42_...``."""
    match = re.search(r'index(\d+)_', filename)
    return int(match.group(1)) if match else 0


def create_gif(image_folder: str, output_filename: str, fps: int = 10) -> None:
    """Create a GIF (or MP4) from PNG frames in *image_folder*.

    Frames are sorted by epoch index extracted from their filenames.
    Requires ``imageio`` to be installed (``pip install imageio``).

    Parameters
    ----------
    image_folder:
        Directory containing ``*.png`` frames.
    output_filename:
        Output path.  ``.gif`` extension triggers GIF output; anything else
        is written as a video via imageio's ffmpeg plugin.
    fps:
        Frames per second.
    """
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise ImportError("create_gif requires imageio: pip install imageio") from exc

    images = sorted(glob.glob(os.path.join(image_folder, "*.png")), key=_extract_epoch_index)
    if not images:
        print(f"No PNG images found in {image_folder}")
        return

    frames = [imageio.imread(img) for img in images]

    if output_filename.endswith(".gif"):
        imageio.mimsave(output_filename, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(output_filename, frames, fps=fps, quality=8)

    print(f"Animation saved: {output_filename}")