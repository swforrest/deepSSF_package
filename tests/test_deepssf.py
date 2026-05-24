"""Tests for deepssf — one test per public function.

Run with:  pytest
"""

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# deepssf.utils
# ---------------------------------------------------------------------------

def test_get_device_returns_valid_string():
    from deepssf.utils import get_device
    device = get_device()
    assert device in ("cuda", "mps", "cpu")


def test_recover_hour_roundtrip():
    from deepssf.utils import recover_hour
    for hour in (0.0, 6.0, 12.5, 23.9):
        sin_h = np.sin(2 * np.pi * hour / 24)
        cos_h = np.cos(2 * np.pi * hour / 24)
        assert abs(recover_hour(sin_h, cos_h) - hour) < 1e-6


def test_recover_yday_roundtrip():
    from deepssf.utils import recover_yday
    for yday in (1.0, 90.0, 180.0, 300.0):
        sin_d = np.sin(2 * np.pi * yday / 365.25)
        cos_d = np.cos(2 * np.pi * yday / 365.25)
        assert abs(recover_yday(sin_d, cos_d) - yday) < 1e-4


def test_subset_raster_with_padding_torch_centre():
    """Window centred inside the raster should contain no padding."""
    from deepssf.utils import subset_raster_with_padding_torch
    import rasterio.transform

    H, W = 200, 200
    raster = torch.ones(H, W)
    transform = rasterio.transform.from_bounds(0, 0, W, H, W, H)
    # geographic centre → pixel centre
    subset, _, _ = subset_raster_with_padding_torch(raster, W / 2, H / 2, 11, transform)
    assert subset.shape == (11, 11)
    assert (subset == 1.0).all(), "No padding expected when window is fully inside"


def test_subset_raster_with_padding_torch_edge():
    """Window that overlaps the raster edge should be padded with -1."""
    from deepssf.utils import subset_raster_with_padding_torch
    import rasterio.transform

    H, W = 50, 50
    raster = torch.zeros(H, W)
    transform = rasterio.transform.from_bounds(0, 0, W, H, W, H)
    # geographic coordinate at the raster corner → pixel (0, 0)
    subset, _, _ = subset_raster_with_padding_torch(raster, 0.5, H - 0.5, 11, transform)
    assert subset.shape == (11, 11)
    assert (subset == -1.0).any(), "Padding expected at the edge"


def test_subset_raster_all_bands_torch():
    from deepssf.utils import subset_raster_all_bands_torch
    import rasterio.transform

    raster = torch.ones(4, 100, 100)
    transform = rasterio.transform.from_bounds(0, 0, 100, 100, 100, 100)
    subset, _, _ = subset_raster_all_bands_torch(raster, 50, 50, 11, transform)
    assert subset.shape == (4, 11, 11)
    assert (subset == 1.0).all()


def test_subset_raster_with_padding_npy():
    from deepssf.utils import subset_raster_with_padding_npy
    import rasterio.transform

    raster = np.ones((100, 100), dtype=np.float32)
    transform = rasterio.transform.from_bounds(0, 0, 100, 100, 100, 100)
    subset, _, _ = subset_raster_with_padding_npy(raster, 50, 50, 11, transform)
    assert subset.shape == (11, 11)
    assert (subset == 1.0).all()


def test_subset_layer_vectorized_2d():
    from deepssf.utils import subset_layer_vectorized
    arr = np.ones((100, 100), dtype=np.float32)
    patch, col_start, row_start = subset_layer_vectorized(arr, 50, 50, 11)
    assert patch.shape == (11, 11)
    assert patch.dtype == torch.float32
    assert (patch == 1.0).all()


def test_subset_layer_vectorized_3d():
    from deepssf.utils import subset_layer_vectorized
    arr = np.ones((4, 100, 100), dtype=np.float32)
    patch, _, _ = subset_layer_vectorized(arr, 50, 50, 11)
    assert patch.shape == (4, 11, 11)


def test_subset_layer_vectorized_edge_padding():
    from deepssf.utils import subset_layer_vectorized
    arr = np.zeros((50, 50), dtype=np.float32)
    patch, _, _ = subset_layer_vectorized(arr, 0, 0, 11)
    assert (patch == -1.0).any()


def test_clear_memory_does_not_raise():
    from deepssf.utils import clear_memory
    clear_memory()  # should silently no-op on CPU


# ---------------------------------------------------------------------------
# deepssf.data (pure-Python helpers only — no rasterio I/O)
# ---------------------------------------------------------------------------

def test_extract_year_month_regex_found():
    from deepssf.data import extract_year_month_regex
    assert extract_year_month_regex("S2_2021_07_mosaic.tif") == "2021_07"


def test_extract_year_month_regex_not_found():
    from deepssf.data import extract_year_month_regex
    assert extract_year_month_regex("no_date_here.tif") is None


def test_day_to_month_index():
    from deepssf.data import day_to_month_index
    # Day 1 of the year should map to January (month 1)
    assert day_to_month_index(1) == 1
    # Day ~180 should map to June/July
    assert 6 <= day_to_month_index(180) <= 7


# ---------------------------------------------------------------------------
# deepssf.model
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_params():
    """Tiny ModelParams that fits in a few MB of RAM.

    The test uses 2 raw spatial channels and 4 scalar-to-grid channels, so
    ``input_channels`` must be 2 + 4 = 6 (scalars are broadcast and
    concatenated before the first conv layer).
    """
    from deepssf.model import ModelParams
    return ModelParams({
        "batch_size": 2,
        "image_dim": 11,
        "pixel_size": 25,
        "dim_in_nonspatial_to_grid": 4,
        "dense_dim_in_nonspatial": 4,
        "dense_dim_hidden": 8,
        "dense_dim_in_all": 8,   # updated per test_convjointmodel_forward_shape
        "input_channels": 6,     # 2 spatial + 4 scalar-grid channels
        "output_channels": 2,
        "kernel_size": 3,
        "stride": 1,
        "kernel_size_mp": 2,
        "stride_mp": 2,
        "padding": 1,
        "num_movement_params": 12,
        "dropout": 0.0,
        "device": "cpu",
    })


def test_model_params_construction(small_params):
    assert small_params.image_dim == 11
    assert small_params.device == "cpu"


def test_conv2d_block_spatial_output_shape(small_params):
    from deepssf.model import Conv2d_block_spatial
    block = Conv2d_block_spatial(small_params)
    # input_channels=6 from the fixture
    x = torch.zeros(2, small_params.input_channels, 11, 11)
    out = block(x)
    assert out.shape == (2, 11, 11)


def test_scalar_to_grid_block_output_shape(small_params):
    from deepssf.model import Scalar_to_Grid_Block
    block = Scalar_to_Grid_Block(small_params)
    x = torch.zeros(2, 4)
    out = block(x)
    assert out.shape == (2, 4, 11, 11)


def test_convjointmodel_forward_shape(small_params):
    """Full forward pass — output must be [B, H, W, 2]."""
    from deepssf.model import ConvJointModel

    # Adjust dense_dim_in_all to match the actual flattened size produced by
    # Conv2d_block_toFC with these tiny hyperparams (image_dim=11, 3 × MP2).
    import math
    dim = 11
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)  # conv (stride=1, pad=1 keeps dim)
        dim = math.floor((dim - 2) / 2 + 1)           # maxpool kernel=2, stride=2
    flat = small_params.output_channels * dim * dim

    from deepssf.model import ModelParams
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})

    model = ConvJointModel(params)
    B, H, W = 2, 11, 11
    spatial = torch.randn(B, 2, H, W)
    scalars = torch.randn(B, 4)
    bearing = torch.zeros(B, 1)

    out = model((spatial, scalars, bearing))
    assert out.shape == (B, H, W, 2), f"Expected ({B},{H},{W},2), got {out.shape}"


def test_habitat_output_log_normalised(small_params):
    """Habitat sub-network output should sum to 1 in probability space."""
    from deepssf.model import Conv2d_block_spatial
    block = Conv2d_block_spatial(small_params)
    block.eval()
    x = torch.zeros(1, small_params.input_channels, 11, 11)
    log_p = block(x)
    total = torch.exp(log_p).sum()
    assert abs(total.item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# deepssf.train
# ---------------------------------------------------------------------------

def test_negative_log_like_loss_mean():
    from deepssf.train import negativeLogLikeLoss
    loss_fn = negativeLogLikeLoss(reduction="mean")
    B, H, W = 2, 5, 5
    predict = torch.zeros(B, H, W, 2)
    target  = torch.zeros(B, H, W)
    target[:, 2, 2] = 1.0  # observed pixel
    loss = loss_fn(predict, target)
    assert loss.shape == ()      # scalar
    assert torch.isfinite(loss)


def test_negative_log_like_loss_reductions():
    from deepssf.train import negativeLogLikeLoss
    B, H, W = 2, 5, 5
    predict = torch.zeros(B, H, W, 2)
    target  = torch.zeros(B, H, W)
    target[:, 1, 1] = 1.0

    mean_val = negativeLogLikeLoss("mean")(predict, target)
    sum_val  = negativeLogLikeLoss("sum")(predict, target)
    none_val = negativeLogLikeLoss("none")(predict, target)

    assert none_val.shape == (B, H, W)
    assert abs(sum_val.item() / (B * H * W) - mean_val.item()) < 1e-5


def test_negative_log_like_loss_invalid_reduction():
    from deepssf.train import negativeLogLikeLoss
    with pytest.raises(ValueError):
        negativeLogLikeLoss("invalid")


def test_early_stopping_counter_increments(tmp_path):
    from deepssf.train import EarlyStopping
    model = torch.nn.Linear(2, 1)
    es = EarlyStopping(patience=3, path=str(tmp_path / "ckpt.pt"))

    es(1.0, model)  # new best
    assert es.counter == 0
    es(1.5, model)  # worse
    assert es.counter == 1
    es(1.5, model)  # worse
    assert es.counter == 2
    assert not es.early_stop
    es(1.5, model)  # patience exhausted
    assert es.early_stop


def test_early_stopping_resets_on_improvement(tmp_path):
    from deepssf.train import EarlyStopping
    model = torch.nn.Linear(2, 1)
    es = EarlyStopping(patience=3, path=str(tmp_path / "ckpt.pt"))

    es(1.0, model)
    es(1.5, model)
    es(1.5, model)
    assert es.counter == 2
    es(0.5, model)   # new best — counter resets
    assert es.counter == 0
    assert not es.early_stop