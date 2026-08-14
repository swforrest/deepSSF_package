"""Tests for deepssf — one test per public function.

Run with:  pytest
"""

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
