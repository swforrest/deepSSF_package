"""Tests for deepssf — one test per public function.

Run with:  pytest
"""

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
    total_frozen, _, _ = negativeLogLikeLoss("mean", freeze_movement=True)(predict, (px2, py2))
    total_joint, _, _  = negativeLogLikeLoss("mean", freeze_movement=False)(predict, (px2, py2))
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
    x2, hours, ydays = make_simulation_inputs(n_steps=10, starting_yday=90, starting_hour=6)
    assert x2.shape == (10, 4)
    assert hours.shape == (10,)
    assert ydays.shape == (10,)


def test_make_simulation_inputs_cyclic_encoding():
    from deepssf.simulate import make_simulation_inputs
    import math
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
    """simulate_next_step returns new coordinates and three log-prob tensors."""
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

    new_x, new_y, hab, move, step, px, py = simulate_next_step(
        model, rasters, scalars, bearing, window_size=W,
        x_loc=W * 25 / 2, y_loc=W * 25 / 2, transform=transform,
    )
    assert isinstance(new_x, float)
    assert isinstance(new_y, float)
    assert hab.shape == (W, W)
    assert move.shape == (W, W)
    assert step.shape == (W, W)
    assert 0 <= px < W
    assert 0 <= py < W


def test_simulate_trajectory_dataframe_shape(small_params):
    """simulate_trajectory returns a DataFrame with one row per step."""
    import math
    import rasterio.transform
    from deepssf.model import ConvJointModel, ModelParams
    from deepssf.simulate import simulate_trajectory

    dim = small_params.image_dim
    for _ in range(3):
        dim = math.floor((dim + 2 * 1 - 3) / 1 + 1)
        dim = math.floor((dim - 2) / 2 + 1)
    flat = small_params.output_channels * dim * dim
    params = ModelParams({**small_params.__dict__, "dense_dim_in_all": flat})
    model = ConvJointModel(params)

    W = 11
    transform = rasterio.transform.from_bounds(0, 0, W * 25 * 10, W * 25 * 10, W * 10, W * 10)
    rasters = [torch.ones(W * 10, W * 10) for _ in range(2)]

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

def _make_movement_df(n: int, x_centre: float, y_centre: float) -> "pd.DataFrame":
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

    required = {"x1_", "y1_", "x2_", "y2_", "t1_", "bearing", "dt_hour",
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