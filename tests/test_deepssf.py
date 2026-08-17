"""Tests for deepssf — one test per public function.

Run with:  pytest
"""

import base64
import io
import math
from pathlib import Path

import numpy as np
import pytest
import torch

# Path to the bundled test dataset
_DATA_DIR = Path(__file__).parent.parent / "src" / "deepssf" / "datasets" / "data"
_CSV_PATH  = _DATA_DIR / "buffalo_djelk_id2005.csv"
_LAYER_PATHS = {
    "ndvi":  str(_DATA_DIR / "ndvi_2005.tif"),
    "slope": str(_DATA_DIR / "slope_2005.tif"),
}


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
    import rasterio.transform

    from deepssf.utils import subset_raster_with_padding_torch

    H, W = 200, 200
    raster = torch.ones(H, W)
    transform = rasterio.transform.from_bounds(0, 0, W, H, W, H)
    # geographic centre → pixel centre
    subset, _, _ = subset_raster_with_padding_torch(raster, W / 2, H / 2, 11, transform)
    assert subset.shape == (11, 11)
    assert (subset == 1.0).all(), "No padding expected when window is fully inside"


def test_subset_raster_with_padding_torch_edge():
    """Window that overlaps the raster edge should be padded with -1."""
    import rasterio.transform

    from deepssf.utils import subset_raster_with_padding_torch

    H, W = 50, 50
    raster = torch.zeros(H, W)
    transform = rasterio.transform.from_bounds(0, 0, W, H, W, H)
    # geographic coordinate at the raster corner → pixel (0, 0)
    subset, _, _ = subset_raster_with_padding_torch(raster, 0.5, H - 0.5, 11, transform)
    assert subset.shape == (11, 11)
    assert (subset == -1.0).any(), "Padding expected at the edge"


def test_subset_raster_all_bands_torch():
    import rasterio.transform

    from deepssf.utils import subset_raster_all_bands_torch

    raster = torch.ones(4, 100, 100)
    transform = rasterio.transform.from_bounds(0, 0, 100, 100, 100, 100)
    subset, _, _ = subset_raster_all_bands_torch(raster, 50, 50, 11, transform)
    assert subset.shape == (4, 11, 11)
    assert (subset == 1.0).all()


def test_subset_raster_with_padding_npy():
    import rasterio.transform

    from deepssf.utils import subset_raster_with_padding_npy

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


def _write_tif(path, data):
    """Write a [bands, H, W] float array to a GeoTIFF at *path*."""
    import rasterio
    import rasterio.transform

    bands, height, width = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=bands,
        dtype="float32",
        transform=rasterio.transform.from_bounds(0, 0, width, height, width, height),
    ) as dst:
        dst.write(data.astype("float32"))


@pytest.mark.parametrize(
    "values",
    [
        [-10000.0, 0.0, 10000.0],   # S2 indices x10 000: used to give [0, 2]
        [-100.0, -55.0, -10.0],     # all negative: used to flip sign and axis
        [0.0, 50.0, 100.0],         # lo == 0: the only case the old form got right
        [3.0, 3.0, 3.0],            # constant: used to divide by zero
    ],
)
def test_environmental_layers_scaled_to_unit_range(tmp_path, values):
    """Every .tif layer must land in [0, 1] whatever its original range.

    Regression test: the scaling divided by the maximum rather than the range,
    which only lands in [0, 1] when the minimum happens to be zero.
    """
    from deepssf.data import load_environmental_layers

    path = tmp_path / "layer.tif"
    _write_tif(path, np.array(values, dtype="float32").reshape(1, 1, 3))

    layers, _transform = load_environmental_layers({"layer": str(path)})
    scaled = layers["layer"]

    assert scaled.min() >= 0.0
    assert scaled.max() <= 1.0
    if len(set(values)) > 1:
        # Non-constant layers use the full range and stay monotonic in the input.
        assert scaled.min() == pytest.approx(0.0)
        assert scaled.max() == pytest.approx(1.0)
        assert np.all(np.diff(scaled.ravel()) > 0)
    else:
        assert np.all(scaled == 0.0)


def test_environmental_layers_multiband_scaled_jointly(tmp_path):
    """Bands of one TIFF share a scaling, so relative magnitudes survive."""
    from deepssf.data import load_environmental_layers

    path = tmp_path / "multi.tif"
    data = np.array([[[0.0, 5.0]], [[10.0, 20.0]]], dtype="float32")  # 2 bands
    _write_tif(path, data)

    scaled = load_environmental_layers({"multi": str(path)})[0]["multi"]
    assert scaled.shape == (2, 1, 2)
    # Scaled by the global range (0..20), not per band.
    assert scaled.ravel() == pytest.approx([0.0, 0.25, 0.5, 1.0])


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
    # Adjust dense_dim_in_all to match the actual flattened size produced by
    # Conv2d_block_toFC with these tiny hyperparams (image_dim=11, 3 × MP2).
    import math

    from deepssf.model import ConvJointModel
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
# Configurable conv depth
#
# n_conv_layers_hab / n_conv_layers_move let the user set the depth of the two
# sub-networks independently.  Both are optional, and omitting them must keep
# the architecture the deepSSF paper used (4 habitat convs, 3 movement blocks),
# because saved checkpoints only load into a model with matching layer counts.
# ---------------------------------------------------------------------------

def test_conv_layer_defaults_match_original_architecture(small_params):
    from torch import nn

    from deepssf.model import Conv2d_block_spatial, Conv2d_block_toFC

    assert small_params.n_conv_layers_hab == 4
    assert small_params.n_conv_layers_move == 3

    hab = Conv2d_block_spatial(small_params).conv2d
    mov = Conv2d_block_toFC(small_params).conv2d
    assert sum(isinstance(m, nn.Conv2d) for m in hab) == 4
    assert sum(isinstance(m, nn.ReLU) for m in hab) == 3      # none after the last conv
    assert sum(isinstance(m, nn.Conv2d) for m in mov) == 3
    assert sum(isinstance(m, nn.MaxPool2d) for m in mov) == 3  # one per conv


@pytest.mark.parametrize("n_hab", [1, 2, 6])
def test_habitat_depth_is_configurable(small_params, n_hab):
    """Any depth >= 1 still gives a normalised [B, H, W] surface."""
    from torch import nn

    from deepssf.model import Conv2d_block_spatial, ModelParams
    params = ModelParams({**small_params.__dict__, "n_conv_layers_hab": n_hab})
    block = Conv2d_block_spatial(params)

    convs = [m for m in block.conv2d if isinstance(m, nn.Conv2d)]
    assert len(convs) == n_hab
    # Channels must chain: input → output_channels → ... → 1
    assert convs[0].in_channels == params.input_channels
    assert convs[-1].out_channels == 1

    out = block(torch.zeros(1, params.input_channels, 11, 11))
    assert out.shape == (1, 11, 11)                       # no pooling, dims preserved
    assert abs(torch.exp(out).sum().item() - 1.0) < 1e-5


@pytest.mark.parametrize("n_move", [1, 2, 3])
def test_movement_depth_is_configurable(small_params, n_move):
    """Flattened size must match flattened_conv_dim for any depth."""
    from deepssf.model import Conv2d_block_toFC, ModelParams, flattened_conv_dim
    params = ModelParams({**small_params.__dict__, "n_conv_layers_move": n_move})
    block = Conv2d_block_toFC(params)

    out = block(torch.zeros(2, params.input_channels, 11, 11))
    expected = flattened_conv_dim(
        image_dim=11,
        n_conv_layers_move=n_move,
        output_channels=params.output_channels,
        kernel_size=params.kernel_size,
        stride=params.stride,
        padding=params.padding,
        kernel_size_mp=params.kernel_size_mp,
        stride_mp=params.stride_mp,
    )
    assert out.shape == (2, expected)


def test_conv_depths_are_independent(small_params):
    """Setting one depth must not change the other sub-network."""
    from torch import nn

    from deepssf.model import ConvJointModel, ModelParams, flattened_conv_dim
    n_hab, n_move = 2, 1
    params = ModelParams({
        **small_params.__dict__,
        "n_conv_layers_hab": n_hab,
        "n_conv_layers_move": n_move,
        "dense_dim_in_all": flattened_conv_dim(
            image_dim=11,
            n_conv_layers_move=n_move,
            output_channels=small_params.output_channels,
        ),
    })
    model = ConvJointModel(params)

    n_conv = lambda block: sum(  # noqa: E731
        isinstance(m, nn.Conv2d) for m in block.conv2d
    )
    assert n_conv(model.conv_habitat) == n_hab
    assert n_conv(model.conv_movement) == n_move

    out = model((torch.randn(2, 2, 11, 11), torch.randn(2, 4), torch.zeros(2, 1)))
    assert out.shape == (2, 11, 11, 2)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("key", ["n_conv_layers_hab", "n_conv_layers_move"])
def test_conv_depth_below_one_rejected(small_params, key):
    from deepssf.model import Conv2d_block_spatial, Conv2d_block_toFC, ModelParams
    params = ModelParams({**small_params.__dict__, key: 0})
    block = (
        Conv2d_block_spatial if key == "n_conv_layers_hab" else Conv2d_block_toFC
    )
    with pytest.raises(ValueError, match=key):
        block(params)


def test_habitat_edge_buffer_tracks_habitat_depth(small_params):
    """The receptive-field buffer must follow the configured depth."""
    from deepssf.model import Conv2d_block_spatial, ModelParams
    from deepssf.predict import habitat_edge_buffer

    class _Stub:  # habitat_edge_buffer only reads model.conv_habitat
        def __init__(self, block):
            self.conv_habitat = block

    for n_hab in (1, 3, 5):
        params = ModelParams({**small_params.__dict__, "n_conv_layers_hab": n_hab})
        buffer = habitat_edge_buffer(_Stub(Conv2d_block_spatial(params)))
        assert buffer == n_hab * (params.kernel_size // 2)


def test_flattened_conv_dim_matches_paper_architecture():
    """The published 101-pixel window / 4 channels / 3 layers case."""
    from deepssf.model import flattened_conv_dim
    # 101 → 50 → 25 → 12 after three conv+maxpool blocks; 4 * 12 * 12 = 576
    assert flattened_conv_dim(101, 3, 4) == 576


def test_flattened_conv_dim_rejects_window_pooled_away():
    from deepssf.model import flattened_conv_dim
    with pytest.raises(ValueError, match="too small"):
        flattened_conv_dim(image_dim=11, n_conv_layers_move=6, output_channels=4)


# ---------------------------------------------------------------------------
# deepssf.train
# ---------------------------------------------------------------------------

def test_negative_log_like_loss_mean():
    from deepssf.train import negativeLogLikeLoss
    loss_fn = negativeLogLikeLoss(reduction="mean")
    B, H, W = 2, 5, 5
    predict = torch.zeros(B, H, W, 2)
    px2 = torch.tensor([2, 2])  # observed col for each batch item
    py2 = torch.tensor([2, 2])  # observed row for each batch item
    total, hab, mov = loss_fn(predict, (px2, py2))
    assert total.shape == ()      # scalar
    assert torch.isfinite(total)
    assert torch.isfinite(hab)
    assert torch.isfinite(mov)


def test_negative_log_like_loss_reductions():
    from deepssf.train import negativeLogLikeLoss
    B, H, W = 2, 5, 5
    predict = torch.zeros(B, H, W, 2)
    px2 = torch.tensor([1, 1])
    py2 = torch.tensor([1, 1])
    target = (px2, py2)

    mean_total, _, _ = negativeLogLikeLoss("mean")(predict, target)
    sum_total, _, _  = negativeLogLikeLoss("sum")(predict, target)
    none_total, _, _ = negativeLogLikeLoss("none")(predict, target)

    assert none_total.shape == (B,)
    assert abs(sum_total.item() / B - mean_total.item()) < 1e-5


def test_negative_log_like_loss_median():
    from deepssf.train import negativeLogLikeLoss
    B, H, W = 2, 5, 5
    predict = torch.zeros(B, H, W, 2)
    px2 = torch.tensor([2, 2])
    py2 = torch.tensor([2, 2])
    med_total, _, _ = negativeLogLikeLoss("median")(predict, (px2, py2))
    assert med_total.shape == ()


def test_negative_log_like_loss_freeze_movement():
    from deepssf.train import negativeLogLikeLoss
    B, H, W = 2, 5, 5
    predict = torch.zeros(B, H, W, 2)
    px2 = torch.tensor([2, 2])
    py2 = torch.tensor([2, 2])
    total_frozen, _, _ = negativeLogLikeLoss(
        "mean", freeze_movement=True
    )(predict, (px2, py2))
    total_joint, _, _ = negativeLogLikeLoss(
        "mean", freeze_movement=False
    )(predict, (px2, py2))
    # With all-zero logits both should be finite; frozen uses only habitat channel
    assert torch.isfinite(total_frozen)
    assert torch.isfinite(total_joint)


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


# ---------------------------------------------------------------------------
# deepssf.simulate
# ---------------------------------------------------------------------------

def test_make_simulation_inputs_shape():
    from deepssf.simulate import make_simulation_inputs
    x2, hours, ydays = make_simulation_inputs(
        n_steps=10, starting_yday=90, starting_hour=6
    )
    assert x2.shape == (10, 5)  # sin_hour, cos_hour, sin_yday, cos_yday, dt
    assert hours.shape == (10,)
    assert ydays.shape == (10,)


def test_make_simulation_inputs_cyclic_encoding():
    import math

    from deepssf.simulate import make_simulation_inputs
    x2, _, _ = make_simulation_inputs(n_steps=1, starting_yday=1, starting_hour=0)
    # hour=0 → sin=0, cos=1
    assert abs(x2[0, 0]) < 1e-10
    assert abs(x2[0, 1] - 1.0) < 1e-10
    # yday=1 → sin=sin(2π/365.25), cos=cos(2π/365.25)
    assert abs(x2[0, 2] - math.sin(2 * math.pi / 365.25)) < 1e-10


def test_make_simulation_inputs_hour_wraps():
    from deepssf.simulate import make_simulation_inputs
    _, hours, _ = make_simulation_inputs(n_steps=25, starting_yday=1, starting_hour=0)
    assert hours[24] == 0.0  # wraps at 24


def test_simulate_next_step_returns_coords_and_tensors(small_params):
    """simulate_next_step returns new coordinates."""
    import math

    import rasterio.transform

    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.simulate import simulate_next_step

    # Build a working model (same dim calculation as test_convjointmodel_forward_shape)
    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})
    model = ConvJointModel(params)
    model.eval()

    W = 11
    transform = rasterio.transform.from_bounds(0, 0, W * 25, W * 25, W, W)
    # Two spatial raster channels (image_dim=11, but landscape larger than crop)
    rasters = [torch.ones(W * 4, W * 4) for _ in range(2)]
    scalars = torch.zeros(1, 4)
    bearing = torch.zeros(1, 1)

    new_x, new_y, px, py = simulate_next_step(
        model, rasters, scalars, bearing, window_size=W,
        x_loc=W * 25 / 2, y_loc=W * 25 / 2, transform=transform,
    )
    assert isinstance(new_x, float)
    assert isinstance(new_y, float)
    assert 0 <= px < W
    assert 0 <= py < W


def test_simulate_trajectory_dataframe_shape(small_params):
    """simulate_trajectory returns a DataFrame with one row per step."""
    import math

    import rasterio.transform

    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.simulate import make_simulation_inputs, simulate_trajectory

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim

    # Probe make_simulation_inputs to find out how many scalar columns it emits.
    # This keeps the test in sync with the function automatically if columns change.
    n_scalars = make_simulation_inputs(n_steps=1, starting_yday=1)[0].shape[1]
    n_spatial = 2  # number of raster channels used in this test
    params = ModelParams({
        **small_params.__dict__,
        "dim_in_nonspatial_to_grid": n_scalars,
        "input_channels": n_spatial + n_scalars,
        "dense_dim_in_all": flat,
    })
    model = ConvJointModel(params)

    W = 11
    transform = rasterio.transform.from_bounds(
        0, 0, W * 25 * 10, W * 25 * 10, W * 10, W * 10
    )
    rasters = [torch.ones(W * 10, W * 10) for _ in range(n_spatial)]

    df = simulate_trajectory(
        model,
        get_landscape=lambda _month: rasters,
        transform=transform,
        start_x=W * 25 * 5,
        start_y=W * 25 * 5,
        n_steps=3,
        starting_yday=1,
        window_size=W,
    )
    assert len(df) == 3
    for col in ("x", "y", "hour", "yday", "month_index"):
        assert col in df.columns


# ---------------------------------------------------------------------------
# deepssf.validate
# ---------------------------------------------------------------------------

def _make_movement_df(n: int, x_centre: float, y_centre: float):
    """Synthetic movement DataFrame with required columns."""
    import pandas as pd
    rng = np.random.default_rng(0)
    xs = x_centre + rng.uniform(-50, 50, n)
    ys = y_centre + rng.uniform(-50, 50, n)
    df = pd.DataFrame(
        {
            "x1_": xs,
            "y1_": ys,
            "x2_": np.roll(xs, -1),  # next step = shifted current
            "y2_": np.roll(ys, -1),
            "hour_t2_sin": np.sin(2 * np.pi * np.arange(n) / 24),
            "hour_t2_cos": np.cos(2 * np.pi * np.arange(n) / 24),
            "yday_t2_sin": np.sin(2 * np.pi * np.arange(n) / 365.25),
            "yday_t2_cos": np.cos(2 * np.pi * np.arange(n) / 365.25),
            "yday_t2": (np.arange(n) % 365) + 1,
            "bearing_tm1": np.zeros(n),
        }
    )
    return df


def test_validate_next_step_probs_returns_columns(small_params):
    """validate_next_step_probs appends three probability columns."""
    import math

    import rasterio.transform

    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.validate import validate_next_step_probs

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})
    model = ConvJointModel(params)

    W = 11
    landscape_size = W * 20
    transform = rasterio.transform.from_bounds(
        0, 0, landscape_size * 25, landscape_size * 25, landscape_size, landscape_size
    )
    rasters = [torch.ones(landscape_size, landscape_size) for _ in range(2)]
    centre = landscape_size * 25 / 2

    df = _make_movement_df(5, x_centre=centre, y_centre=centre)
    result = validate_next_step_probs(
        model,
        df,
        get_landscape=lambda _m: rasters,
        transform=transform,
        window_size=W,
    )
    assert len(result) == len(df)
    for col in ("habitat_prob", "move_prob", "next_step_prob"):
        assert col in result.columns


def test_validate_next_step_probs_row0_is_zero(small_params):
    """Row 0 must always be 0.0 (no previous bearing)."""
    import math

    import rasterio.transform

    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.validate import validate_next_step_probs

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})
    model = ConvJointModel(params)

    W = 11
    landscape_size = W * 20
    transform = rasterio.transform.from_bounds(
        0, 0, landscape_size * 25, landscape_size * 25, landscape_size, landscape_size
    )
    rasters = [torch.ones(landscape_size, landscape_size) for _ in range(2)]
    centre = landscape_size * 25 / 2

    df = _make_movement_df(4, x_centre=centre, y_centre=centre)
    result = validate_next_step_probs(
        model, df, get_landscape=lambda _m: rasters,
        transform=transform, window_size=W,
    )
    assert result["habitat_prob"].iloc[0] == 0.0
    assert result["next_step_prob"].iloc[0] == 0.0


def test_day_to_s2_month_wraps_to_1_12():
    """_day_to_s2_month always returns values in 1–12, even for multi-year yday."""
    from deepssf.validate import _day_to_s2_month
    for yday in (1, 90, 180, 300, 365, 400, 730):
        m = _day_to_s2_month(yday)
        assert 1 <= m <= 12, f"yday={yday} → month={m} out of range"


# ---------------------------------------------------------------------------
# deepssf.data — integration tests against the bundled test dataset
# ---------------------------------------------------------------------------

def test_prepare_movement_df_columns():
    """prepare_movement_df produces the required step-format columns."""
    import pandas as pd

    from deepssf.data import prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)

    required = {"x1_", "y1_", "x2_", "y2_", "t1_", "dx", "dy", "bearing",
                "dt_hour", "hour_t1", "yday_t1",
                "hour_t1_sin1", "hour_t1_cos1", "yday_t1_sin1", "yday_t1_cos1"}
    assert required.issubset(df.columns)


def test_prepare_movement_df_row_count():
    """One row is dropped per individual (last fix has no next location)."""
    import pandas as pd

    from deepssf.data import prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)

    # One row dropped per unique id
    n_ids = raw["id"].nunique()
    assert len(df) == len(raw) - n_ids


def test_prepare_movement_df_bearing_finite():
    """Bearing values must all be finite (no NaN from missing coords)."""
    import pandas as pd

    from deepssf.data import prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)
    assert df["bearing"].notna().all()
    assert np.isfinite(df["bearing"].values).all()


def test_prepare_movement_df_cyclic_range():
    """Cyclic encodings must stay in [-1, 1]."""
    import pandas as pd

    from deepssf.data import prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)
    for col in ("hour_t1_sin1", "hour_t1_cos1", "yday_t1_sin1", "yday_t1_cos1"):
        assert df[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


@pytest.mark.skipif(not _CSV_PATH.exists(), reason="test dataset not found")
def test_movement_dataset_getitem_shapes():
    """MovementDataset __getitem__ returns correctly shaped tensors."""
    import pandas as pd

    from deepssf.data import MovementDataset, prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)
    window = 25  # small window for speed

    # Use only 20 rows so __init__ is fast in tests
    dataset = MovementDataset(
        df.iloc[:20].reset_index(drop=True),
        _LAYER_PATHS,
        window_size=window,
        scalar_cols=["hour_t1_sin1", "hour_t1_cos1",
                     "yday_t1_sin1", "yday_t1_cos1", "dt_hour"],
    )

    spatial, scalars, bearing, (px2, py2), transform = dataset[0]

    assert spatial.ndim == 3                    # [C, H, W]
    assert spatial.shape[-1] == window
    assert spatial.shape[-2] == window
    assert scalars.shape == (5,)
    assert bearing.shape == (1,)


def test_prepare_movement_df_has_bearing_tm1():
    """prepare_movement_df includes bearing_tm1 (previous step's bearing)."""
    import pandas as pd

    from deepssf.data import prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)

    assert "bearing_tm1" in df.columns
    assert "yday_t1" in df.columns
    assert df["bearing_tm1"].iloc[0] == 0.0
    assert np.isfinite(df["bearing_tm1"].values).all()


def test_prepare_movement_df_has_dx_dy():
    """prepare_movement_df includes dx and dy displacement columns."""
    import pandas as pd

    from deepssf.data import prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)

    assert "dx" in df.columns
    assert "dy" in df.columns

    dx = df["dx"].to_numpy(dtype=float)
    dy = df["dy"].to_numpy(dtype=float)
    x1 = df["x1_"].to_numpy(dtype=float)
    x2 = df["x2_"].to_numpy(dtype=float)
    y1 = df["y1_"].to_numpy(dtype=float)
    y2 = df["y2_"].to_numpy(dtype=float)

    assert np.isfinite(dx).all()
    assert np.isfinite(dy).all()

    np.testing.assert_allclose(dx, x2 - x1)
    np.testing.assert_allclose(dy, y2 - y1)


def test_filter_steps_by_window_removes_large_steps():
    """filter_steps_by_window drops steps that exceed the spatial window."""
    import pandas as pd

    from deepssf.data import filter_steps_by_window, prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    df = prepare_movement_df(raw)

    window_size = 25
    pixel_size = 25.0
    filtered = filter_steps_by_window(
        df, window_size=window_size, pixel_size=pixel_size
    )

    half_extent = (window_size - 1) * pixel_size / 2
    assert (filtered["dx"].abs() < half_extent).all()
    assert (filtered["dy"].abs() < half_extent).all()
    assert len(filtered) <= len(df)


def test_make_optimisers_returns_tuples(small_params):
    """make_optimisers returns two tuples of (movement, habitat) optimisers."""
    import math

    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.train import make_optimisers

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})
    model = ConvJointModel(params)

    optimisers, schedulers = make_optimisers(model)
    assert len(optimisers) == 2
    assert len(schedulers) == 2
    opt_mov, opt_hab = optimisers
    assert isinstance(opt_mov, torch.optim.Adam)
    assert isinstance(opt_hab, torch.optim.Adam)


def test_fit_returns_loss_history(small_params):
    """fit returns a history dict with the expected keys and epoch count."""
    import math

    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.train import fit, make_optimisers, negativeLogLikeLoss
    from deepssf.utils import get_device

    # The final assertion reads the *untrained* habitat surface, which depends
    # on the random initialisation — seed it so the test does not depend on how
    # much of the global RNG stream earlier tests happened to consume.
    torch.manual_seed(0)

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})
    device = get_device()
    model = ConvJointModel(params).to(device)

    n_spatial = params.input_channels - params.dim_in_nonspatial_to_grid
    H = params.image_dim
    S = params.dim_in_nonspatial_to_grid

    class _DS:
        def __len__(self):
            return 4

    class _DL:
        dataset = _DS()

        def __len__(self):
            return 2

        def __iter__(self):
            for _ in range(2):
                yield (
                    torch.randn(2, n_spatial, H, H),
                    torch.randn(2, S),
                    torch.zeros(2, 1),
                    (
                        torch.full((2,), H // 2, dtype=torch.long),
                        torch.full((2,), H // 2, dtype=torch.long),
                    ),
                    ("t", "t"),
                )

    loss_fn = negativeLogLikeLoss(reduction="mean")
    optimisers, schedulers = make_optimisers(model)
    history = fit(
        model,
        image_trim_pixels=3,       # number of conv layers (used only for snapshots)
        window_size=H,         # spatial crop size (used only for snapshots)
        dl_train=_DL(),
        dl_val=_DL(),
        loss_fn=loss_fn,
        optimisers=optimisers,
        schedulers=schedulers,
        n_epochs=2,
        snapshot_dir=None,
    )
    assert set(history.keys()) == {
        "train_losses", "val_losses",
        "val_habitat_losses", "val_movement_losses", "stage",
    }
    # Index 0 is the pre-training baseline, so 2 epochs give 3 rows.
    assert len(history["train_losses"]) == 3
    assert len(history["val_losses"]) == 3
    assert math.isnan(history["train_losses"][0])
    assert history["stage"] == [-1, 0, 0]
    # An untrained habitat surface is close to uniform over the H x H window.
    assert history["val_habitat_losses"][0] == pytest.approx(math.log(H * H), abs=0.1)


def _joint_model_and_loader(small_params):
    """Build a tiny ConvJointModel plus a fake (train == val) DataLoader.

    Mirrors the setup in :func:`test_fit_returns_loss_history`; the flattened
    dimension has to be derived from image_dim because Conv2d_block_toFC
    max-pools three times.
    """
    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.utils import get_device

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})
    model = ConvJointModel(params).to(get_device())

    n_spatial = params.input_channels - params.dim_in_nonspatial_to_grid
    H = params.image_dim
    S = params.dim_in_nonspatial_to_grid

    class _DS:
        def __len__(self):
            return 4

    class _DL:
        dataset = _DS()

        def __len__(self):
            return 2

        def __iter__(self):
            for _ in range(2):
                yield (
                    torch.randn(2, n_spatial, H, H),
                    torch.randn(2, S),
                    torch.zeros(2, 1),
                    (
                        torch.full((2,), H // 2, dtype=torch.long),
                        torch.full((2,), H // 2, dtype=torch.long),
                    ),
                    ("t", "t"),
                )

    return model, params, _DL()


def test_make_optimisers_owns_every_parameter(small_params):
    """Every model parameter belongs to exactly one optimiser.

    Regression test: conv_movement used to be in neither optimiser, so it was
    never zeroed or stepped.  Its gradients accumulated across the whole run
    until they overflowed, at which point _grads_are_finite (which scans every
    parameter) failed and silently skipped every update, habitat included.
    """
    from deepssf.train import make_optimisers

    model, _params, _dl = _joint_model_and_loader(small_params)
    optimisers, _schedulers = make_optimisers(model)

    owners: dict[int, int] = {}
    for i, opt in enumerate(optimisers):
        for group in opt.param_groups:
            for param in group["params"]:
                assert id(param) not in owners, "parameter owned by two optimisers"
                owners[id(param)] = i

    unowned = [
        name for name, param in model.named_parameters() if id(param) not in owners
    ]
    assert unowned == [], f"parameters owned by no optimiser: {unowned}"


def test_train_loop_leaves_no_stale_gradients(small_params):
    """After an epoch, no parameter is left holding an un-zeroed gradient."""
    from deepssf.train import make_optimisers, negativeLogLikeLoss, train_loop

    model, _params, dl = _joint_model_and_loader(small_params)
    optimisers, _ = make_optimisers(model)
    train_loop(dl, model, negativeLogLikeLoss(reduction="mean"), optimisers)

    for name, param in model.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), name


def test_stages_freeze_the_inactive_subnetwork(small_params):
    """A sub-network excluded from a stage's 'train' set must not change."""
    from deepssf.train import fit, make_optimisers, negativeLogLikeLoss

    model, params, dl = _joint_model_and_loader(small_params)
    optimisers, schedulers = make_optimisers(model, lr_habitat=1e-1, lr_movement=1e-1)

    frozen_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.startswith("conv_habitat")
    }
    movement_before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.startswith(("conv_movement", "fcn_movement_all"))
    }

    history = fit(
        model,
        image_trim_pixels=3,
        window_size=params.image_dim,
        dl_train=dl,
        dl_val=dl,
        loss_fn=negativeLogLikeLoss(reduction="mean"),
        optimisers=optimisers,
        schedulers=schedulers,
        stages=[{"epochs": 2, "train": ("movement",)}],
        snapshot_dir=None,
    )

    for name, before in frozen_before.items():
        assert torch.equal(before, dict(model.named_parameters())[name]), (
            f"{name} changed during a movement-only stage"
        )
    # ...and the active branch did move, so the test is not vacuous.
    assert any(
        not torch.equal(before, dict(model.named_parameters())[name])
        for name, before in movement_before.items()
    )
    assert history["stage"] == [-1, 0, 0]


def test_early_stopping_monitor_both_waits_for_both_heads(tmp_path):
    """A plateaued head cannot stop a run while the other is still improving."""
    from deepssf.train import EarlyStopping

    model = torch.nn.Linear(1, 1)
    es = EarlyStopping(
        patience=2, monitor="both", path=str(tmp_path / "ckpt.pt"),
        trace_func=lambda _: None,
    )

    # Movement plateaued (drifting slightly worse), habitat improving every
    # epoch: movement's counter maxes out but must not end the run.  Note that
    # with delta=0 an exactly-unchanged loss counts as an improvement, which is
    # why a plateau is modelled as a small upward drift.
    for i in range(6):
        es(
            val_loss=10.0 - i, model=model,
            val_habitat=10.0 - i, val_movement=5.0 + 0.01 * i,
        )
        assert not es.early_stop, f"stopped at epoch {i} while habitat improving"
    assert es._head_counter["movement"] >= es.patience
    assert es._head_counter["habitat"] == 0

    # Habitat now plateaus too — both counters reach patience and the run ends.
    for i in range(2):
        es(
            val_loss=5.5, model=model,
            val_habitat=5.5 + 0.01 * i, val_movement=5.1 + 0.01 * i,
        )
    assert es.early_stop

    # reset() clears counters so a new stage starts with a full patience budget,
    # while leaving the best score and saved checkpoint intact.
    best_before = es.best_score
    es.reset()
    assert not es.early_stop
    assert es.best_score == best_before
    es(val_loss=5.5, model=model, val_habitat=5.6, val_movement=5.2)
    assert not es.early_stop


def test_early_stopping_monitor_both_requires_components(tmp_path):
    from deepssf.train import EarlyStopping

    es = EarlyStopping(monitor="both", path=str(tmp_path / "ckpt.pt"))
    with pytest.raises(ValueError, match="requires"):
        es(val_loss=1.0, model=torch.nn.Linear(1, 1))

    with pytest.raises(ValueError, match="monitor must be"):
        EarlyStopping(monitor="habitat")


def test_early_stopping_monitor_total_unchanged(tmp_path):
    """The default monitor still counts patience against the combined loss."""
    from deepssf.train import EarlyStopping

    model = torch.nn.Linear(1, 1)
    es = EarlyStopping(patience=2, path=str(tmp_path / "ckpt.pt"),
                       trace_func=lambda _: None)
    es(val_loss=1.0, model=model)
    assert not es.early_stop
    es(val_loss=2.0, model=model)
    assert not es.early_stop
    es(val_loss=2.0, model=model)
    assert es.early_stop


@pytest.mark.parametrize(
    "stages, match",
    [
        ([], "non-empty"),
        ([{"epochs": 0, "train": ("habitat",)}], "epochs must be"),
        ([{"epochs": 1, "train": ()}], "at least one component"),
        ([{"epochs": 1, "train": ("hab",)}], "unknown component"),
        ([{"epochs": 1, "train": ("habitat",), "lr": 1.0}], "unknown key"),
        ([{"train": ("habitat",)}], "missing key"),
    ],
)
def test_stages_validation(stages, match):
    from deepssf.train import _normalise_stages

    with pytest.raises(ValueError, match=match):
        _normalise_stages(stages, n_epochs=1)


@pytest.mark.skipif(not _CSV_PATH.exists(), reason="test dataset not found")
def test_make_dataloaders_df_parameter():
    """make_dataloaders accepts df= as an alternative to csv_path."""
    import pandas as pd

    from deepssf.data import make_dataloaders, prepare_movement_df

    raw = pd.read_csv(_CSV_PATH)
    step_df = prepare_movement_df(raw)

    dl_train, *_ = make_dataloaders(
        layer_paths=_LAYER_PATHS,
        window_size=25,
        batch_size=4,
        df=step_df,
        scalar_cols=["hour_t1_sin1", "hour_t1_cos1",
                     "yday_t1_sin1", "yday_t1_cos1"],
    )
    assert len(dl_train) > 0


@pytest.mark.skipif(not _CSV_PATH.exists(), reason="test dataset not found")
def test_make_dataloaders_prepare_flag():
    """make_dataloaders(prepare=True) works end-to-end on raw CSV."""
    from deepssf.data import make_dataloaders

    dl_train, dl_val, dl_test = make_dataloaders(
        str(_CSV_PATH),
        _LAYER_PATHS,
        window_size=25,
        batch_size=4,
        train_split=0.7,
        val_split=0.15,
        prepare=True,
        scalar_cols=["hour_t1_sin1", "hour_t1_cos1",
                     "yday_t1_sin1", "yday_t1_cos1", "dt_hour"],
    )
    assert len(dl_train) > 0
    spatial, scalars, bearing, labels, _ = next(iter(dl_train))
    px2, py2 = labels
    assert spatial.ndim == 4       # [B, C, H, W]
    assert px2.shape[0] == spatial.shape[0]  # batch sizes match

# ---------------------------------------------------------------------------
# Numerical stability of the movement kernel
#
# Regression tests for the NaN surfaces reported in 0.2.3: in float32 the von
# Mises normaliser I0(kappa) overflows at kappa ~= 89 (a raw FCN output of only
# log(89) ~= 4.5).  When both mixture components overflowed the whole surface
# became NaN; when only one did, the forward pass stayed finite but the
# gradients did not, silently poisoning the weights.
# ---------------------------------------------------------------------------

def _grid_block(image_dim=21, pixel_size=25, device="cpu"):
    from deepssf.model import ModelParams, Params_to_Grid_Block_ChV
    return Params_to_Grid_Block_ChV(ModelParams({
        "batch_size": 1, "image_dim": image_dim, "pixel_size": pixel_size,
        "dim_in_nonspatial_to_grid": 4, "dense_dim_in_nonspatial": 4,
        "dense_dim_hidden": 8, "dense_dim_in_all": 8, "input_channels": 6,
        "output_channels": 2, "kernel_size": 3, "stride": 1,
        "kernel_size_mp": 2, "stride_mp": 2, "padding": 1,
        "num_movement_params": 12, "dropout": 0.0, "device": device,
    }))


@pytest.mark.parametrize(
    "label,raw",
    [
        ("both kappas overflow I0",    {7: 4.5, 10: 4.5}),
        ("both kappas far past I0",    {7: 8.0, 10: 8.0}),
        ("single kappa overflow",      {7: 6.0}),
        ("saturating mixture weight",  {2: 5.0}),
        ("gamma shape overflows exp",  {0: 90.0}),
        ("gamma scale underflows exp", {1: -90.0}),
        ("all parameters large",       dict.fromkeys(range(12), 50.0)),
        ("all parameters small",       dict.fromkeys(range(12), -50.0)),
    ],
)
def test_movement_grid_finite_for_extreme_params(label, raw):
    """Forward *and* backward stay finite for any movement parameter values."""
    block = _grid_block()
    x = torch.zeros(1, 12, requires_grad=True)
    with torch.no_grad():
        for idx, value in raw.items():
            x[0, idx] = value

    out = block(x, torch.zeros(1, 1))
    assert torch.isfinite(out).all(), f"non-finite surface: {label}"

    out[0, 0, 0].backward()
    assert torch.isfinite(x.grad).all(), f"non-finite gradients: {label}"


def test_movement_grid_is_log_normalised():
    """The returned surface is a log-probability: exp() sums to 1 per sample."""
    block = _grid_block()
    torch.manual_seed(0)
    x = torch.randn(4, 12)
    out = block(x, torch.zeros(4, 1))
    assert torch.allclose(out.exp().sum(dim=(1, 2)),
                          torch.ones(4), atol=1e-4)


def test_mixture_weight_gradients_do_not_vanish():
    """Weight parameters stay trainable at magnitudes that used to saturate.

    softmax(exp(raw)) saturated to exactly (1, 0) by raw ~= 4.75, zeroing the
    gradient; log_softmax over the raw logits keeps it finite and non-zero.
    """
    block = _grid_block()
    x = torch.zeros(1, 12, requires_grad=True)
    with torch.no_grad():
        x[0, 2] = 5.0     # gamma mixture weight, component 1
        x[0, 8] = 5.0     # von Mises mixture weight, component 1

    block(x, torch.zeros(1, 1))[0, 0, 0].backward()
    assert x.grad[0, 2] != 0.0
    assert x.grad[0, 8] != 0.0


def test_movement_grid_buffers_move_with_model():
    """distance/bearing layers are buffers, so .to() relocates them once."""
    block = _grid_block()
    names = {name for name, _ in block.named_buffers()}
    assert {"distance_layer", "bearing_layer"} <= names


def test_grads_are_finite_helper():
    from deepssf.train import _grads_are_finite

    model = torch.nn.Linear(2, 2)
    model(torch.ones(1, 2)).sum().backward()
    assert _grads_are_finite(model)

    with torch.no_grad():
        model.weight.grad[0, 0] = float("nan")
    assert not _grads_are_finite(model)


# ---------------------------------------------------------------------------
# Checkpoint format guard
# ---------------------------------------------------------------------------

def _tiny_model():
    from deepssf.model import ConvJointModel, ModelParams
    return ConvJointModel(ModelParams({
        "batch_size": 1, "image_dim": 21, "pixel_size": 25,
        "dim_in_nonspatial_to_grid": 4, "dense_dim_in_nonspatial": 4,
        "dense_dim_hidden": 8, "dense_dim_in_all": 8, "input_channels": 6,
        "output_channels": 2, "kernel_size": 3, "stride": 1,
        "kernel_size_mp": 2, "stride_mp": 2, "padding": 1,
        "num_movement_params": 12, "dropout": 0.0, "device": "cpu",
    }))


def test_early_stopping_writes_versioned_checkpoint(tmp_path):
    from deepssf.train import CHECKPOINT_FORMAT, EarlyStopping

    path = tmp_path / "ckpt.pt"
    model = _tiny_model()
    EarlyStopping(path=str(path))(1.0, model)

    saved = torch.load(path, weights_only=True)
    assert saved["deepssf_checkpoint_format"] == CHECKPOINT_FORMAT
    assert "state_dict" in saved
    assert saved["val_loss"] == 1.0


def test_load_checkpoint_round_trip(tmp_path):
    from deepssf.train import EarlyStopping, load_checkpoint

    path = tmp_path / "ckpt.pt"
    saved_model = _tiny_model()
    EarlyStopping(path=str(path))(0.5, saved_model)

    fresh = _tiny_model()
    load_checkpoint(str(path), fresh)
    for (_, a), (_, b) in zip(
        saved_model.state_dict().items(), fresh.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b)


def test_load_checkpoint_rejects_legacy_by_default(tmp_path):
    """A bare state_dict (≤ 0.2.3) must fail loudly, not load silently."""
    from deepssf.train import load_checkpoint

    path = tmp_path / "legacy.pt"
    torch.save(_tiny_model().state_dict(), path)   # pre-0.3.0 layout

    with pytest.raises(RuntimeError, match="pre-0.3.0"):
        load_checkpoint(str(path), _tiny_model())


def test_load_checkpoint_allow_legacy_opt_in(tmp_path):
    from deepssf.train import load_checkpoint

    path = tmp_path / "legacy.pt"
    torch.save(_tiny_model().state_dict(), path)

    out = load_checkpoint(str(path), _tiny_model(), allow_legacy=True)
    assert out["deepssf_checkpoint_format"] == 1


def test_load_checkpoint_rejects_future_format(tmp_path):
    from deepssf.train import CHECKPOINT_FORMAT, load_checkpoint

    path = tmp_path / "future.pt"
    torch.save({"deepssf_checkpoint_format": CHECKPOINT_FORMAT + 1,
                "state_dict": _tiny_model().state_dict()}, path)

    with pytest.raises(RuntimeError, match="Upgrade deepssf"):
        load_checkpoint(str(path))


# ---------------------------------------------------------------------------
# deepssf.train — recovering the architecture from a checkpoint
# ---------------------------------------------------------------------------

def _legacy_format2_checkpoint(path, model):
    """A format-2 checkpoint as written before model_params was recorded."""
    from deepssf.train import CHECKPOINT_FORMAT
    torch.save({"deepssf_checkpoint_format": CHECKPOINT_FORMAT,
                "val_loss": 1.0,
                "state_dict": model.state_dict()}, path)


def test_checkpoint_records_model_params(tmp_path):
    from deepssf.train import EarlyStopping, load_checkpoint

    path = tmp_path / "ckpt.pt"
    model = _tiny_model()
    EarlyStopping(path=str(path))(1.0, model)

    assert load_checkpoint(str(path))["model_params"] == model.params.to_dict()


def test_checkpoint_omits_model_params_for_a_plain_module(tmp_path):
    """EarlyStopping takes any nn.Module, not only ConvJointModel."""
    from deepssf.train import EarlyStopping, load_checkpoint

    path = tmp_path / "ckpt.pt"
    EarlyStopping(path=str(path))(1.0, torch.nn.Linear(2, 2))

    assert "model_params" not in load_checkpoint(str(path))


def test_params_from_checkpoint_round_trip(tmp_path):
    """The rebuilt params reproduce the model the checkpoint came from."""
    from deepssf.model import ConvJointModel
    from deepssf.train import EarlyStopping, load_checkpoint, params_from_checkpoint

    path = tmp_path / "ckpt.pt"
    trained = _tiny_model()
    EarlyStopping(path=str(path))(1.0, trained)

    params = params_from_checkpoint(str(path), device="cpu")
    assert params.to_dict() == trained.params.to_dict()

    # The point of the exercise: the weights load into it strictly.
    rebuilt = ConvJointModel(params)
    load_checkpoint(str(path), rebuilt)
    for (_, a), (_, b) in zip(
        trained.state_dict().items(), rebuilt.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b)


def test_params_from_checkpoint_reads_shapes_when_params_absent(tmp_path):
    """Pre-0.3.1 checkpoints: the architecture comes from the layer shapes."""
    from deepssf.model import ConvJointModel
    from deepssf.train import load_checkpoint, params_from_checkpoint

    path = tmp_path / "legacy.pt"
    trained = _tiny_model()
    _legacy_format2_checkpoint(path, trained)

    params = params_from_checkpoint(
        str(path), image_dim=21, pixel_size=25, n_scalar_covariates=4, device="cpu"
    )
    for field in ("input_channels", "output_channels", "kernel_size",
                  "n_conv_layers_hab", "n_conv_layers_move",
                  "dense_dim_in_all", "dense_dim_hidden", "num_movement_params"):
        assert getattr(params, field) == getattr(trained.params, field), field

    load_checkpoint(str(path), ConvJointModel(params))


def test_params_from_checkpoint_requires_window_when_not_recorded(tmp_path):
    from deepssf.train import params_from_checkpoint

    path = tmp_path / "legacy.pt"
    _legacy_format2_checkpoint(path, _tiny_model())

    with pytest.raises(ValueError, match="image_dim is not recorded"):
        params_from_checkpoint(str(path), pixel_size=25)


def test_params_from_checkpoint_rejects_wrong_window(tmp_path):
    """A window that flattens to a different size cannot be the trained one."""
    from deepssf.train import params_from_checkpoint

    path = tmp_path / "legacy.pt"
    _legacy_format2_checkpoint(path, _tiny_model())   # trained at image_dim=21

    with pytest.raises(ValueError, match="flattens to"):
        params_from_checkpoint(str(path), image_dim=101, pixel_size=25)


def test_params_from_checkpoint_accepts_window_within_pooling_slack(tmp_path):
    """Max-pooling floor-divides, so nearby windows flatten identically."""
    from deepssf.train import params_from_checkpoint

    path = tmp_path / "legacy.pt"
    _legacy_format2_checkpoint(path, _tiny_model())   # 21 → 10 → 5 → 2

    assert params_from_checkpoint(str(path), image_dim=20, pixel_size=25).image_dim == 20


def test_params_from_checkpoint_rejects_params_contradicting_weights(tmp_path):
    """A stored params dict that disagrees with the weights is not trusted."""
    from deepssf.train import EarlyStopping, params_from_checkpoint

    path = tmp_path / "ckpt.pt"
    EarlyStopping(path=str(path))(1.0, _tiny_model())

    tampered = torch.load(path, weights_only=True)
    tampered["model_params"]["n_conv_layers_hab"] += 1
    torch.save(tampered, path)

    with pytest.raises(RuntimeError, match="contradict its own weights"):
        params_from_checkpoint(str(path))


def test_params_from_checkpoint_overrides_win(tmp_path):
    from deepssf.train import EarlyStopping, params_from_checkpoint

    path = tmp_path / "ckpt.pt"
    EarlyStopping(path=str(path))(1.0, _tiny_model())

    params = params_from_checkpoint(str(path), pixel_size=10, dropout=0.5)
    assert params.pixel_size == 10
    assert params.dropout == 0.5

    with pytest.raises(ValueError, match="unknown ModelParams field"):
        params_from_checkpoint(str(path), not_a_field=1)


def test_params_from_checkpoint_device_follows_this_machine(tmp_path):
    """Where a model was trained says nothing about where it is being run."""
    from deepssf.train import EarlyStopping, params_from_checkpoint
    from deepssf.utils import get_device

    path = tmp_path / "ckpt.pt"
    model = _tiny_model()          # the fixture builds it on 'cpu'
    EarlyStopping(path=str(path))(1.0, model)

    assert params_from_checkpoint(str(path)).device == get_device()
    assert params_from_checkpoint(str(path), device="cpu").device == "cpu"


def test_params_from_checkpoint_rejects_non_convjoint_checkpoint(tmp_path):
    from deepssf.train import EarlyStopping, params_from_checkpoint

    path = tmp_path / "ckpt.pt"
    EarlyStopping(path=str(path))(1.0, torch.nn.Linear(2, 2))

    with pytest.raises(RuntimeError, match="ConvJointModel"):
        params_from_checkpoint(str(path), image_dim=21, pixel_size=25)


def test_model_params_to_dict_round_trips(small_params):
    from deepssf.model import ModelParams

    d = small_params.to_dict()
    assert set(d) == set(ModelParams.FIELDS)
    assert ModelParams(d).to_dict() == d
    # Values are plain Python, so torch.load(weights_only=True) can read them
    assert all(isinstance(v, (bool, int, float, str)) for v in d.values())


# ---------------------------------------------------------------------------
# deepssf.simulate — multiple trajectories and saving
# ---------------------------------------------------------------------------

def _sim_model_and_landscape(small_params, window=11):
    """Working model + landscape/transform sized for the simulation tests."""
    import rasterio.transform

    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.simulate import make_simulation_inputs

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim

    # Match the scalar channel count to whatever make_simulation_inputs emits
    n_scalars = make_simulation_inputs(n_steps=1, starting_yday=1)[0].shape[1]
    n_spatial = 2
    params = ModelParams({
        **small_params.__dict__,
        "dim_in_nonspatial_to_grid": n_scalars,
        "input_channels": n_spatial + n_scalars,
        "dense_dim_in_all": flat,
    })
    model = ConvJointModel(params)
    model.eval()

    size = window * 4
    # North-up transform (negative y resolution), 25 m cells
    transform = rasterio.transform.from_origin(0, size * 25, 25, 25)
    rasters = [torch.rand(size, size) for _ in range(n_spatial)]
    return model, rasters, transform, size


def test_simulate_trajectories_batched_shape(small_params):
    from deepssf.simulate import simulate_trajectories_batched

    W = 11
    model, rasters, transform, size = _sim_model_and_landscape(small_params, W)
    centre = size * 25 / 2

    df = simulate_trajectories_batched(
        model,
        get_landscape=lambda _m: rasters,
        transform=transform,
        start_x=centre,
        start_y=centre,
        n_steps=5,
        n_trajectories=3,
        window_size=W,
        month_index_fn=lambda _y: 0,
    )
    assert len(df) == 15
    assert sorted(df["trajectory_id"].unique()) == [0, 1, 2]
    assert list(df.columns) == [
        "trajectory_id", "step", "x", "y", "hour", "yday", "month_index",
    ]
    # Every sampled location must sit inside the raster extent
    assert df["x"].between(0, size * 25).all()
    assert df["y"].between(0, size * 25).all()


def test_simulate_trajectories_batched_per_trajectory_starts(small_params):
    """Start locations may differ between trajectories."""
    from deepssf.simulate import simulate_trajectories_batched

    W = 11
    model, rasters, transform, size = _sim_model_and_landscape(small_params, W)
    starts_x = [size * 25 * 0.3, size * 25 * 0.7]
    starts_y = [size * 25 * 0.3, size * 25 * 0.7]

    df = simulate_trajectories_batched(
        model,
        get_landscape=lambda _m: rasters,
        transform=transform,
        start_x=starts_x,
        start_y=starts_y,
        n_steps=2,
        window_size=W,
        month_index_fn=lambda _y: 0,
    )
    assert df["trajectory_id"].nunique() == 2
    first_steps = df[df["step"] == 0].sort_values("trajectory_id")
    # Each trajectory stays within one window of its own start
    for (_, row), sx in zip(first_steps.iterrows(), starts_x, strict=True):
        assert abs(row["x"] - sx) <= (W // 2 + 1) * 25


def test_simulate_trajectories_methods_agree_on_shape(small_params):
    from deepssf.simulate import simulate_trajectories

    W = 11
    model, rasters, transform, size = _sim_model_and_landscape(small_params, W)
    centre = size * 25 / 2

    frames = {
        method: simulate_trajectories(
            model,
            get_landscape=lambda _m: rasters,
            transform=transform,
            start_x=centre,
            start_y=centre,
            n_steps=4,
            n_trajectories=2,
            method=method,
            window_size=W,
            month_index_fn=lambda _y: 0,
        )
        for method in ("batched", "sequential", "parallel")
    }
    for method, df in frames.items():
        assert len(df) == 8, method
        assert set(df["trajectory_id"]) == {0, 1}, method
        assert {"x", "y", "hour", "yday", "step"} <= set(df.columns), method


def test_simulate_trajectories_batched_rejects_varying_start_times(small_params):
    from deepssf.simulate import simulate_trajectories

    W = 11
    model, rasters, transform, size = _sim_model_and_landscape(small_params, W)
    with pytest.raises(ValueError, match="shared clock"):
        simulate_trajectories(
            model,
            get_landscape=lambda _m: rasters,
            transform=transform,
            start_x=size * 25 / 2,
            start_y=size * 25 / 2,
            n_steps=2,
            n_trajectories=2,
            starting_yday=[10, 200],
            window_size=W,
            month_index_fn=lambda _y: 0,
        )


def test_simulate_trajectories_rejects_unknown_method(small_params):
    from deepssf.simulate import simulate_trajectories

    W = 11
    model, rasters, transform, size = _sim_model_and_landscape(small_params, W)
    with pytest.raises(ValueError, match="Unknown method"):
        simulate_trajectories(
            model,
            get_landscape=lambda _m: rasters,
            transform=transform,
            start_x=size * 25 / 2,
            start_y=size * 25 / 2,
            n_steps=2,
            method="mpi",
            window_size=W,
        )


def test_crop_windows_matches_single_window_helper():
    """The batched crop must reproduce subset_raster_with_padding_torch."""
    import rasterio.transform

    from deepssf.simulate import _crop_windows, _pad_landscape, _pixel_from_coords
    from deepssf.utils import subset_raster_with_padding_torch

    W = 7
    transform = rasterio.transform.from_origin(0, 500, 25, 25)
    rasters = [torch.rand(20, 20), torch.rand(20, 20)]
    xs = np.array([50.0, 300.0, 12.0])   # includes a location near the edge
    ys = np.array([450.0, 200.0, 490.0])

    padded = _pad_landscape(rasters, W, "cpu")
    cols, rows = _pixel_from_coords(transform, xs, ys)
    batch = _crop_windows(
        padded, torch.as_tensor(cols), torch.as_tensor(rows), W
    )

    for j, (x, y) in enumerate(zip(xs, ys, strict=True)):
        for c, raster in enumerate(rasters):
            expected, _, _ = subset_raster_with_padding_torch(
                raster, x=x, y=y, window_size=W, transform=transform
            )
            assert torch.allclose(batch[j, c], expected)


def test_save_trajectories_does_not_overwrite(tmp_path):
    import pandas as pd

    from deepssf.simulate import save_trajectories

    df = pd.DataFrame({
        "trajectory_id": [0, 0, 1, 1],
        "step": [0, 1, 0, 1],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
    })

    first = save_trajectories(df, tmp_path, prefix="test", date="2026-01-01")
    second = save_trajectories(df, tmp_path, prefix="test", date="2026-01-01")

    assert first[0].name == "test_2traj_2steps_2026-01-01.csv"
    assert second[0].name == "test_2traj_2steps_2026-01-01_2.csv"
    assert first[0].exists() and second[0].exists()
    assert len(pd.read_csv(first[0])) == 4


def test_save_trajectories_split_writes_one_file_per_trajectory(tmp_path):
    import pandas as pd

    from deepssf.simulate import save_trajectories

    df = pd.DataFrame({
        "trajectory_id": [0, 0, 1, 1],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [1.0, 2.0, 3.0, 4.0],
    })
    written = save_trajectories(
        df, tmp_path, prefix="test", split=True, date="2026-01-01"
    )
    assert len(written) == 2
    assert all(p.exists() for p in written)
    assert len(pd.read_csv(written[0])) == 2


# ---------------------------------------------------------------------------
# deepssf.plot
# ---------------------------------------------------------------------------

def test_plot_trajectories_folium_builds_map():
    import pandas as pd

    folium = pytest.importorskip("folium")
    from deepssf.plot import plot_trajectories_folium

    # A few points in EPSG:3112 (Australian Lambert), near the study area
    df = pd.DataFrame({
        "trajectory_id": [0, 0, 1, 1],
        "x": [40000.0, 40100.0, 40200.0, 40300.0],
        "y": [-1400000.0, -1400100.0, -1400200.0, -1400300.0],
    })
    fmap = plot_trajectories_folium(df, src_crs="EPSG:3112")
    assert isinstance(fmap, folium.Map)
    html = fmap.get_root().render()
    assert "polyline" in html.lower()


def test_plot_trajectories_folium_requires_input():
    pytest.importorskip("folium")
    from deepssf.plot import plot_trajectories_folium

    with pytest.raises(ValueError, match="Nothing to plot"):
        plot_trajectories_folium()


@pytest.mark.parametrize("cell", [10.0, 25.0, 100.0])
def test_simulate_next_step_jitter_scales_with_raster_resolution(small_params, cell):
    """The sub-pixel jitter must keep the location inside the sampled cell.

    Regression test: the jitter was hard-coded to a 25 m cell, so on a 10 m
    raster it scattered locations up to 2.5 cells away from the pixel actually
    sampled, and on a 100 m raster it never left the first quarter of the cell.
    """
    import rasterio.transform

    from deepssf.simulate import make_simulation_inputs, simulate_next_step

    W = 11
    model, rasters, _transform, size = _sim_model_and_landscape(small_params, W)
    transform = rasterio.transform.from_origin(0, size * cell, cell, cell)
    n_scalars = make_simulation_inputs(n_steps=1, starting_yday=1)[0].shape[1]

    x_loc = y_loc = size * cell / 2
    for _ in range(25):
        new_x, new_y, px, py = simulate_next_step(
            model, rasters, torch.zeros(1, n_scalars), torch.zeros(1, 1),
            window_size=W, x_loc=x_loc, y_loc=y_loc, transform=transform,
        )
        # Upper-left corner of the cell that was sampled
        centre_col, centre_row = (~transform) * (x_loc, y_loc)
        half = W // 2
        col = int(np.floor(centre_col)) - half + px
        row = int(np.floor(centre_row)) - half + py
        corner_x, corner_y = transform * (col, row)

        assert 0.0 <= new_x - corner_x <= cell
        # North-up raster: y decreases into the cell from its upper-left corner
        assert -cell <= new_y - corner_y <= 0.0


# ---------------------------------------------------------------------------
# deepssf.predict
# ---------------------------------------------------------------------------

def _habitat_model(small_params, n_spatial=2):
    """Model whose habitat CNN takes n_spatial rasters + the simulation scalars."""
    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.simulate import make_simulation_inputs

    n_scalars = make_simulation_inputs(n_steps=1, starting_yday=1)[0].shape[1]
    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    params = ModelParams({
        **small_params.__dict__,
        "dim_in_nonspatial_to_grid": n_scalars,
        "input_channels": n_spatial + n_scalars,
        "dense_dim_in_all": small_params.output_channels * dim * dim,
    })
    model = ConvJointModel(params)
    model.eval()
    return model, n_scalars


def test_habitat_edge_buffer_matches_receptive_field(small_params):
    from deepssf.predict import habitat_edge_buffer
    model, _ = _habitat_model(small_params)
    # Four 3x3 convolutions, each pulling in one column of padding
    assert habitat_edge_buffer(model) == 4


def test_predict_habitat_landscape_shape_and_mask(small_params):
    from deepssf.predict import habitat_edge_buffer, predict_habitat_landscape

    model, n_scalars = _habitat_model(small_params)
    rasters = [torch.rand(60, 80) for _ in range(2)]
    surface = predict_habitat_landscape(model, rasters, np.zeros(n_scalars))

    buf = habitat_edge_buffer(model)
    assert surface.shape == (60, 80)
    assert np.isnan(surface[:buf, :]).all()
    assert np.isnan(surface[:, -buf:]).all()
    assert np.isfinite(surface[buf:-buf, buf:-buf]).all()
    # Normalised: the valid cells form a probability distribution
    assert np.exp(surface[~np.isnan(surface)]).sum() == pytest.approx(1.0, rel=1e-4)


def test_predict_habitat_landscape_chunking_is_exact(small_params):
    """Chunked and single-pass predictions must agree cell for cell."""
    from deepssf.predict import predict_habitat_landscape

    model, n_scalars = _habitat_model(small_params)
    rasters = [torch.rand(120, 40) for _ in range(2)]
    scalars = np.linspace(-1, 1, n_scalars)

    whole = predict_habitat_landscape(model, rasters, scalars, chunk_rows=None)
    chunked = predict_habitat_landscape(model, rasters, scalars, chunk_rows=16)

    valid = ~np.isnan(whole)
    assert np.allclose(whole[valid], chunked[valid], atol=1e-5)


def test_predict_habitat_landscape_matches_windowed_forward(small_params):
    """A pixel's landscape value must equal what a window centred there gives.

    This is the property that makes the whole approach valid: the habitat CNN
    is fully convolutional, so running it over the landscape is the same as
    sliding the training window across it.
    """
    from deepssf.predict import predict_habitat_landscape

    model, n_scalars = _habitat_model(small_params)
    H = W = 61
    rasters = [torch.rand(H, W) for _ in range(2)]
    scalars = np.zeros(n_scalars)

    landscape = predict_habitat_landscape(
        model, rasters, scalars, normalise=False, chunk_rows=None
    )

    # A 41 px window centred on the middle of the raster
    win = 41
    half = win // 2
    cy = cx = H // 2
    crop = torch.stack(
        [r[cy - half:cy + half + 1, cx - half:cx + half + 1] for r in rasters]
    ).unsqueeze(0)
    scalar_maps = torch.zeros(1, n_scalars, win, win)
    with torch.no_grad():
        windowed = model.conv_habitat.conv2d(
            torch.cat([crop, scalar_maps], dim=1)
        ).squeeze()

    # Compare the centre of the window, far from either set of padding artifacts
    assert windowed[half, half].item() == pytest.approx(landscape[cy, cx], abs=1e-4)


# ---------------------------------------------------------------------------
# deepssf.train — checkpointing during single-head stages
# ---------------------------------------------------------------------------

# Real numbers from a feral-pig run's habitat-only stage (epochs 11-20).  The
# combined loss falls every epoch while the habitat loss bottoms out at epoch
# 12 and then gets steadily worse: habitat is buying a lower joint loss by
# lowering logZ, not by fitting the observed locations better.
_HABITAT_STAGE = [
    # (val_total, val_habitat, val_movement)
    (4.115486, 8.516167, 4.126772),
    (4.115594, 8.481732, 4.126772),   # <- best habitat
    (4.113183, 8.508135, 4.126772),
    (4.112547, 8.521410, 4.126772),
    (4.110984, 8.550330, 4.126772),
    (4.108857, 8.612041, 4.126772),
    (4.107189, 8.547318, 4.126772),
    (4.106579, 8.581531, 4.126772),
    (4.105697, 8.570892, 4.126772),
    (4.104966, 8.578071, 4.126772),
]


def test_habitat_stage_total_and_habitat_losses_disagree():
    """The premise of the fix: these two criteria pick different epochs.

    total = habitat + movement + logZ.  Movement is pinned during the stage, so
    any fall in total that is not a fall in habitat came from logZ.
    """
    totals = [row[0] for row in _HABITAT_STAGE]
    habitats = [row[1] for row in _HABITAT_STAGE]
    movements = [row[2] for row in _HABITAT_STAGE]

    assert len(set(movements)) == 1, "movement should be frozen in this stage"
    assert totals.index(min(totals)) == 9
    assert habitats.index(min(habitats)) == 1

    # logZ absorbed the difference: it fell by more than habitat rose
    log_z = [t - h - m for t, h, m in _HABITAT_STAGE]
    assert log_z[9] < log_z[1]
    assert habitats[9] > habitats[1]


def _stamped_model(value: float):
    """A tiny model whose single parameter records which epoch saved it."""
    import torch.nn as nn

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(value)
    return model


@pytest.mark.parametrize(
    ("checkpoint_on", "expected_epoch"),
    [
        ("total", 9),    # last epoch: combined loss falls monotonically
        ("active", 1),   # epoch with the lowest habitat loss
    ],
)
def test_checkpoint_on_selects_the_right_epoch(tmp_path, checkpoint_on, expected_epoch):
    from deepssf.train import EarlyStopping, load_checkpoint

    path = tmp_path / "best.pt"
    stopper = EarlyStopping(
        patience=100, path=str(path), monitor="both", checkpoint_on=checkpoint_on
    )
    for epoch, (total, hab, mov) in enumerate(_HABITAT_STAGE):
        stopper(
            total, _stamped_model(epoch),
            val_habitat=hab, val_movement=mov,
            active=("habitat",),
        )

    saved = load_checkpoint(str(path))
    assert saved["state_dict"]["weight"].item() == pytest.approx(expected_epoch)


def test_checkpoint_on_active_uses_total_when_all_heads_train(tmp_path):
    """With both heads training, the joint likelihood is the right criterion."""
    from deepssf.train import EarlyStopping, load_checkpoint

    path = tmp_path / "best.pt"
    stopper = EarlyStopping(
        patience=100, path=str(path), monitor="both", checkpoint_on="active"
    )
    # Total improves; habitat gets worse.  The checkpoint should follow total.
    for epoch, (total, hab) in enumerate([(5.0, 8.0), (4.0, 8.5), (3.0, 9.0)]):
        stopper(
            total, _stamped_model(epoch),
            val_habitat=hab, val_movement=1.0,
            active=("habitat", "movement"),
        )

    assert load_checkpoint(str(path))["state_dict"]["weight"].item() == 2
    assert load_checkpoint(str(path))["checkpoint_criterion"] == "total"


def test_checkpoint_criterion_change_resets_best_score(tmp_path):
    """A habitat-only score must not be compared against a combined one."""
    from deepssf.train import EarlyStopping, load_checkpoint

    path = tmp_path / "best.pt"
    stopper = EarlyStopping(
        patience=100, path=str(path), monitor="both", checkpoint_on="active"
    )
    # Habitat-only stage: criterion is habitat (8.0), a much larger number
    stopper(4.0, _stamped_model(0), val_habitat=8.0, val_movement=4.0,
            active=("habitat",))
    assert load_checkpoint(str(path))["checkpoint_criterion"] == "habitat"

    # Joint stage: criterion becomes total (4.5).  Without a reset, -4.5 would
    # beat the stale best of -8.0 for the wrong reason, and every later epoch
    # would be judged against a criterion it is not on.
    stopper(4.5, _stamped_model(1), val_habitat=8.2, val_movement=4.4,
            active=("habitat", "movement"))
    assert load_checkpoint(str(path))["state_dict"]["weight"].item() == 1

    # A worse total in the same stage must NOT save
    stopper(4.9, _stamped_model(2), val_habitat=8.1, val_movement=4.3,
            active=("habitat", "movement"))
    assert load_checkpoint(str(path))["state_dict"]["weight"].item() == 1


def test_head_paths_track_each_head_across_stages(tmp_path):
    from deepssf.train import EarlyStopping, load_checkpoint

    main = tmp_path / "best.pt"
    hab_path = tmp_path / "best_habitat.pt"
    mov_path = tmp_path / "best_movement.pt"
    stopper = EarlyStopping(
        patience=100, path=str(main), monitor="both",
        head_paths={"habitat": str(hab_path), "movement": str(mov_path)},
    )

    # habitat best at epoch 1, movement best at epoch 2
    rows = [(5.0, 8.5, 4.5), (4.8, 8.1, 4.6), (4.6, 8.3, 4.2)]
    for epoch, (total, hab, mov) in enumerate(rows):
        stopper(total, _stamped_model(epoch), val_habitat=hab, val_movement=mov,
                active=("habitat", "movement"))

    assert load_checkpoint(str(hab_path))["state_dict"]["weight"].item() == 1
    assert load_checkpoint(str(mov_path))["state_dict"]["weight"].item() == 2
    # The main checkpoint still follows the combined loss
    assert load_checkpoint(str(main))["state_dict"]["weight"].item() == 2
    assert load_checkpoint(str(hab_path))["checkpoint_criterion"] == "habitat"


def test_head_paths_survive_a_stage_reset(tmp_path):
    """reset() clears patience, but must not re-save a worse per-head best."""
    from deepssf.train import EarlyStopping, load_checkpoint

    hab_path = tmp_path / "best_habitat.pt"
    stopper = EarlyStopping(
        patience=100, path=str(tmp_path / "best.pt"), monitor="both",
        head_paths={"habitat": str(hab_path)},
    )
    stopper(5.0, _stamped_model(0), val_habitat=8.0, val_movement=4.0,
            active=("habitat",))
    stopper.reset()   # stage boundary
    stopper(4.0, _stamped_model(1), val_habitat=8.5, val_movement=3.5,
            active=("habitat", "movement"))

    # Epoch 1's habitat is worse, so the habitat file must still hold epoch 0
    assert load_checkpoint(str(hab_path))["state_dict"]["weight"].item() == 0


def test_early_stopping_rejects_bad_checkpoint_on():
    from deepssf.train import EarlyStopping

    with pytest.raises(ValueError, match="checkpoint_on"):
        EarlyStopping(checkpoint_on="habitat")
    with pytest.raises(ValueError, match="unknown head"):
        EarlyStopping(head_paths={"bearing": "x.pt"})


def test_load_head_weights_loads_only_that_head(tmp_path, small_params):
    """Loading the habitat head must leave the movement head untouched."""
    import copy

    from deepssf.train import EarlyStopping, load_head_weights

    model_a, _ = _habitat_model(small_params)
    model_b, _ = _habitat_model(small_params)   # different random init

    path = tmp_path / "a.pt"
    EarlyStopping(path=str(path))._save(
        str(path), 1.0, model_a, criterion=1.0, criterion_name="habitat"
    )

    before = copy.deepcopy(model_b.state_dict())
    n_loaded = load_head_weights(str(path), model_b, "habitat")
    after = model_b.state_dict()

    assert n_loaded == len(list(model_b.conv_habitat.state_dict()))
    for key in after:
        if key.startswith("conv_habitat."):
            assert torch.equal(after[key], model_a.state_dict()[key]), key
        else:
            assert torch.equal(after[key], before[key]), key


def test_load_head_weights_rejects_unknown_head(tmp_path, small_params):
    from deepssf.train import EarlyStopping, load_head_weights

    model, _ = _habitat_model(small_params)
    path = tmp_path / "a.pt"
    EarlyStopping(path=str(path))._save(
        str(path), 1.0, model, criterion=1.0, criterion_name="total"
    )
    with pytest.raises(ValueError, match="unknown head"):
        load_head_weights(str(path), model, "bearing")


def test_conv_habitat_is_translation_equivariant(small_params):
    """The habitat CNN must give identical values for identical covariates.

    This is the property that makes predict_habitat_landscape valid at all: the
    habitat block is convolutions only, with the scalar covariates broadcast to
    *constant* layers, so it has no positional input and cannot treat "near the
    animal" differently from "at the window edge".  Everything the joint loss
    does to the habitat surface therefore happens in covariate space, never in
    window coordinates.

    Two overlapping crops rather than torch.roll: roll's wrap seam *and* each
    crop's own zero-padding artifacts otherwise land inside the compared region.
    """
    from deepssf.predict import habitat_edge_buffer

    model, n_scalars = _habitat_model(small_params)
    buf = habitat_edge_buffer(model)
    n_spatial = 2
    size, shift = 60, 7

    rng = torch.Generator().manual_seed(0)
    full = torch.rand(n_spatial, size, size + shift, generator=rng)

    def surface(patch):
        scalars = torch.zeros(1, n_scalars, patch.shape[-2], patch.shape[-1])
        with torch.no_grad():
            return model.conv_habitat.conv2d(
                torch.cat([patch.unsqueeze(0), scalars], dim=1)
            ).squeeze()

    left = surface(full[:, :, :size])          # columns 0 .. size-1
    right = surface(full[:, :, shift:])        # columns shift .. size+shift-1

    rows = slice(buf, size - buf)
    # The same ground, addressed in each crop's own local coordinates
    from_left = left[rows, shift + buf : size - buf]
    from_right = right[rows, buf : size - shift - buf]

    assert from_left.numel() > 0
    assert torch.equal(from_left, from_right)


# ---------------------------------------------------------------------------
# deepssf.simulate.trajectory_heatmap
# ---------------------------------------------------------------------------

def _heatmap_inputs():
    """A 20x20 raster of 10 m cells with the origin at (0, 200) — north-up."""
    import pandas as pd
    import rasterio.transform

    transform = rasterio.transform.from_origin(0, 200, 10, 10)
    df = pd.DataFrame({
        "trajectory_id": [0, 0, 0, 1, 1, 1],
        "step": [0, 1, 2, 0, 1, 2],
        # First two locations share a cell; the rest are spread out
        "x": [5.0, 8.0, 45.0, 5.0, 105.0, 155.0],
        "y": [195.0, 192.0, 155.0, 195.0, 105.0, 55.0],
    })
    return df, transform, (20, 20)


def test_trajectory_heatmap_counts_and_placement():
    from deepssf.simulate import trajectory_heatmap

    df, transform, shape = _heatmap_inputs()
    counts, hm_transform = trajectory_heatmap(df, transform, shape)

    assert counts.shape == shape
    assert counts.sum() == len(df)
    assert counts[0, 0] == 3        # three locations in the top-left cell
    assert counts[4, 4] == 1        # (45, 155) → row 4, col 4
    assert hm_transform == transform  # agg=1 leaves the grid untouched


def test_trajectory_heatmap_burn_in_drops_early_steps():
    from deepssf.simulate import trajectory_heatmap

    df, transform, shape = _heatmap_inputs()
    counts, _ = trajectory_heatmap(df, transform, shape, burn_in=2)

    assert counts.sum() == 2        # only step 2 of each trajectory survives
    assert counts[0, 0] == 0        # the release cell is gone


def test_trajectory_heatmap_burn_in_without_step_column():
    """Without a step column, burn-in falls back to row order within each id."""
    from deepssf.simulate import trajectory_heatmap

    df, transform, shape = _heatmap_inputs()
    counts, _ = trajectory_heatmap(
        df.drop(columns="step"), transform, shape, burn_in=2
    )
    assert counts.sum() == 2


def test_trajectory_heatmap_agg_coarsens_grid():
    from deepssf.simulate import trajectory_heatmap

    df, transform, shape = _heatmap_inputs()
    counts, hm_transform = trajectory_heatmap(df, transform, shape, agg=4)

    assert counts.shape == (5, 5)
    assert counts.sum() == len(df)
    assert counts[0, 0] == 3
    assert hm_transform.a == 40 and hm_transform.e == -40
    # Origin is unchanged, so the coarse grid still aligns with the rasters
    assert (hm_transform.c, hm_transform.f) == (transform.c, transform.f)


def test_trajectory_heatmap_drops_out_of_extent_locations():
    import pandas as pd

    from deepssf.simulate import trajectory_heatmap

    df, transform, shape = _heatmap_inputs()
    outside = pd.DataFrame({
        "trajectory_id": [0, 0], "step": [3, 4],
        "x": [-50.0, 500.0], "y": [195.0, 195.0],
    })
    counts, _ = trajectory_heatmap(
        pd.concat([df, outside], ignore_index=True), transform, shape
    )
    assert counts.sum() == len(df)


def test_trajectory_heatmap_accepts_list_of_frames():
    from deepssf.simulate import trajectory_heatmap

    df, transform, shape = _heatmap_inputs()
    frames = [sub for _, sub in df.groupby("trajectory_id")]
    counts, _ = trajectory_heatmap(frames, transform, shape)
    assert counts.sum() == len(df)


def test_trajectory_heatmap_rejects_total_burn_in():
    import pytest

    from deepssf.simulate import trajectory_heatmap

    df, transform, shape = _heatmap_inputs()
    with pytest.raises(ValueError, match="burn_in"):
        trajectory_heatmap(df, transform, shape, burn_in=99)


# ---------------------------------------------------------------------------
# deepssf.data.save_raster
# ---------------------------------------------------------------------------

def test_save_raster_roundtrip(tmp_path):
    import rasterio
    import rasterio.transform

    from deepssf.data import save_raster

    counts = np.arange(12, dtype=np.int64).reshape(3, 4)
    transform = rasterio.transform.from_origin(0, 30, 10, 10)
    path = save_raster(
        counts, tmp_path / "sub" / "counts.tif", transform, "EPSG:3112",
        nodata=0, band_descriptions="simulated locations per cell",
    )

    assert path.exists()
    with rasterio.open(path) as src:
        assert src.count == 1
        assert src.crs.to_string() == "EPSG:3112"
        assert src.transform == transform
        assert src.nodata == 0
        assert src.descriptions[0] == "simulated locations per cell"
        # int64 is narrowed to int32, which GeoTIFF supports
        assert src.dtypes[0] == "int32"
        assert np.array_equal(src.read(1), counts)


def test_save_raster_multiband(tmp_path):
    import rasterio
    import rasterio.transform

    from deepssf.data import save_raster

    arr = np.random.default_rng(0).random((2, 3, 4)).astype("float32")
    path = save_raster(
        arr, tmp_path / "multi.tif",
        rasterio.transform.from_origin(0, 30, 10, 10), "EPSG:4326",
        band_descriptions=["first", "second"],
    )
    with rasterio.open(path) as src:
        assert src.count == 2
        assert src.descriptions == ("first", "second")
        assert np.allclose(src.read(), arr)


def test_save_raster_rejects_4d(tmp_path):
    import pytest
    import rasterio.transform

    from deepssf.data import save_raster

    with pytest.raises(ValueError, match="2-D"):
        save_raster(
            np.zeros((1, 1, 2, 2)), tmp_path / "bad.tif",
            rasterio.transform.from_origin(0, 30, 10, 10), "EPSG:4326",
        )


# ---------------------------------------------------------------------------
# deepssf.plot.add_heatmap_overlay
# ---------------------------------------------------------------------------

def _overlay_rgba(overlay) -> np.ndarray:
    """Decode the PNG folium embedded in an ImageOverlay back to an RGBA array."""
    Image = pytest.importorskip("PIL.Image")
    png = base64.b64decode(overlay.url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def test_add_heatmap_overlay_renders_into_layer_control():
    """The overlay is added after the layer control, but still gets a checkbox."""
    import rasterio.transform

    folium = pytest.importorskip("folium")

    from deepssf.plot import add_heatmap_overlay

    counts = np.zeros((10, 10), dtype=np.int32)
    counts[2:4, 2:4] = 5
    # A small extent in southern Australia, in the CRS the examples use
    transform = rasterio.transform.from_origin(1_100_000, -1_400_000, 100, 100)

    fmap = folium.Map(location=[-25.0, 133.0], zoom_start=10)
    folium.FeatureGroup(name="observed").add_to(fmap)
    folium.LayerControl().add_to(fmap)  # added *before* the overlay

    overlay = add_heatmap_overlay(fmap, counts, transform, name="sim heatmap")
    assert isinstance(overlay, folium.raster_layers.ImageOverlay)

    html = fmap.get_root().render()
    assert '"sim heatmap"' in html.split("_layers = {")[1].split("};")[0]


def test_add_heatmap_overlay_colours_only_occupied_cells():
    """Empty cells are fully transparent and the rest carry the given opacity."""
    import rasterio.transform

    pytest.importorskip("folium")
    import folium

    from deepssf.plot import add_heatmap_overlay

    counts = np.zeros((10, 10), dtype=np.int32)
    counts[5, 5] = 3
    transform = rasterio.transform.from_origin(1_100_000, -1_400_000, 100, 100)

    fmap = folium.Map(location=[-25.0, 133.0], zoom_start=10)
    overlay = add_heatmap_overlay(fmap, counts, transform, opacity=0.6)
    alpha = _overlay_rgba(overlay)[..., 3]

    # folium rescales a float image per channel, which would flatten alpha to
    # 255; a uint8 image is embedded as given.
    assert set(np.unique(alpha)) == {0, round(0.6 * 255)}
    assert (alpha == 0).sum() > alpha.size / 2  # nearly every cell is empty


def test_add_heatmap_overlay_aligns_with_the_source_grid():
    """The coloured pixel lands where the counted cell actually is."""
    import rasterio.transform

    pytest.importorskip("folium")
    import folium
    from rasterio.warp import transform as warp_transform
    from rasterio.warp import transform_bounds

    from deepssf.plot import add_heatmap_overlay

    counts = np.zeros((20, 20), dtype=np.int32)
    row, col = 3, 7
    counts[row, col] = 5
    transform = rasterio.transform.from_origin(1_100_000, -1_400_000, 100, 100)

    fmap = folium.Map(location=[-25.0, 133.0], zoom_start=10)
    overlay = add_heatmap_overlay(fmap, counts, transform)
    alpha = _overlay_rgba(overlay)[..., 3]

    # Where the image sits, in the Web Mercator the overlay is stretched in
    (south, west), (north, east) = overlay.bounds
    w, s, e, n = transform_bounds("EPSG:4326", "EPSG:3857", west, south, east, north)
    # Where the counted cell's centre sits, in the same coordinates
    x, y = transform * (col + 0.5, row + 0.5)
    (cx,), (cy,) = warp_transform("EPSG:3112", "EPSG:3857", [x], [y])

    img_row = int((n - cy) / (n - s) * alpha.shape[0])
    img_col = int((cx - w) / (e - w) * alpha.shape[1])
    assert alpha[img_row, img_col] > 0
    assert (alpha > 0).sum() <= 4  # one source cell, give or take resampling
