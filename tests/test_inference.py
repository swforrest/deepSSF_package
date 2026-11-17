"""Tests for deepssf.inference.

Every inference function must work for any deepSSF model, so the tests run
against several toy ``ConvJointModel`` shapes (window size, habitat depth,
channel width, number of raster bands and of scalars all vary) and compare
against brute-force ``model.forward`` / ``predict_habitat_landscape`` results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Toy models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToyShape:
    """Architecture knobs that inference code must not hard-code."""

    window: int
    n_conv_layers_hab: int
    n_conv_layers_move: int
    channels: int
    n_raster: int
    n_scalars: int

    @property
    def id(self) -> str:
        return (
            f"w{self.window}-hab{self.n_conv_layers_hab}-move{self.n_conv_layers_move}"
            f"-ch{self.channels}-r{self.n_raster}-s{self.n_scalars}"
        )


TOY_SHAPES = [
    ToyShape(
        window=21,
        n_conv_layers_hab=1,
        n_conv_layers_move=2,
        channels=4,
        n_raster=1,
        n_scalars=3,
    ),
    ToyShape(
        window=33,
        n_conv_layers_hab=3,
        n_conv_layers_move=3,
        channels=16,
        n_raster=5,
        n_scalars=7,
    ),
    # Many raster bands relative to scalars, like a real landcover model.  Also
    # the shape that exposed an MPS bug (a conv with a weight sliced to 12 of 15
    # input channels returned wrong values) that the smaller shapes miss.
    ToyShape(
        window=25,
        n_conv_layers_hab=2,
        n_conv_layers_move=2,
        channels=8,
        n_raster=12,
        n_scalars=3,
    ),
]


@dataclass
class ToyCase:
    """A toy model plus inputs that match it."""

    shape: ToyShape
    model: nn.Module
    landscape: torch.Tensor  # [n_raster, H, W], larger than the window


def build_toy_model(shape: ToyShape, seed: int = 0) -> nn.Module:
    """A randomly initialised ConvJointModel with the given shape, in eval mode."""
    from deepssf.model import ConvJointModel, ModelParams, flattened_conv_dim

    torch.manual_seed(seed)
    params = ModelParams(
        {
            "batch_size": 2,
            "image_dim": shape.window,
            "pixel_size": 25,
            "dim_in_nonspatial_to_grid": shape.n_scalars,
            "dense_dim_in_nonspatial": shape.n_scalars,
            "dense_dim_hidden": 8,
            "dense_dim_in_all": flattened_conv_dim(
                image_dim=shape.window,
                n_conv_layers_move=shape.n_conv_layers_move,
                output_channels=shape.channels,
            ),
            "input_channels": shape.n_raster + shape.n_scalars,
            "output_channels": shape.channels,
            "kernel_size": 3,
            "stride": 1,
            "kernel_size_mp": 2,
            "stride_mp": 2,
            "padding": 1,
            "num_movement_params": 12,
            "dropout": 0.0,
            "n_conv_layers_hab": shape.n_conv_layers_hab,
            "n_conv_layers_move": shape.n_conv_layers_move,
            "device": "cpu",
        }
    )
    return ConvJointModel(params).eval()


@pytest.fixture(params=TOY_SHAPES, ids=lambda s: s.id)
def toy(request) -> ToyCase:
    shape: ToyShape = request.param
    model = build_toy_model(shape)
    g = torch.Generator().manual_seed(1)
    # Non-square landscape, comfortably larger than the window, in [0, 1] like
    # load_environmental_layers output.
    landscape = torch.rand(
        shape.n_raster, 2 * shape.window + 7, 3 * shape.window + 2, generator=g
    )
    return ToyCase(shape=shape, model=model, landscape=landscape)


# ---------------------------------------------------------------------------
# Fixture sanity checks — the toy models really vary in the ways the
# generality requirements list, and the structure the inference code reads
# matches what was asked for.
# ---------------------------------------------------------------------------


def test_toy_shapes_cover_required_variation():
    for attr in ("window", "n_conv_layers_hab", "channels", "n_raster", "n_scalars"):
        assert len({getattr(s, attr) for s in TOY_SHAPES}) >= 2, attr


def test_toy_model_structure_is_readable(toy: ToyCase):
    from deepssf.predict import habitat_edge_buffer

    s, m = toy.shape, toy.model
    convs = [c for c in m.conv_habitat.conv2d if isinstance(c, nn.Conv2d)]
    assert len(convs) == s.n_conv_layers_hab
    assert convs[0].in_channels - s.n_scalars == s.n_raster
    assert m.movement_grid_output.distance_layer.shape == (s.window, s.window)
    assert habitat_edge_buffer(m) == s.n_conv_layers_hab


def test_toy_model_forward(toy: ToyCase):
    s, m = toy.shape, toy.model
    B = 3
    spatial = torch.rand(B, s.n_raster, s.window, s.window)
    scalars = torch.randn(B, s.n_scalars)
    bearing = torch.zeros(B, 1)
    with torch.no_grad():
        out = m((spatial, scalars, bearing))
    assert out.shape == (B, s.window, s.window, 2)
    assert torch.isfinite(out).all()
    # Both heads are log-normalised over the window.
    total = torch.logsumexp(out.flatten(1, 2), dim=1)
    torch.testing.assert_close(total, torch.zeros(B, 2), atol=1e-5, rtol=0)


def test_inference_module_imports():
    import deepssf.inference as inf

    assert inf.__doc__ and "shift_c" in inf.__doc__
    assert set(inf.__all__) <= set(__import__("deepssf").__all__)


# ---------------------------------------------------------------------------
# Toy scalar specs — one per toy shape's scalar count, covering every entry type
# ---------------------------------------------------------------------------


def toy_spec(n_scalars: int) -> tuple[list[dict], list[str]]:
    """A spec and its scalar_cols with *n_scalars* columns (3 or 7)."""
    hour = {
        "type": "cyclic",
        "input": "hour",
        "period": 24,
        "columns": ["h_sin", "h_cos"],
    }
    if n_scalars == 3:
        spec = [hour, {"type": "raw", "input": "dt", "column": "dt"}]
        cols = ["h_sin", "h_cos", "dt"]
    elif n_scalars == 7:
        spec = [
            hour,
            {
                "type": "linear",
                "input": "rain",
                "column": "rain_z",
                "transform": "log1p",
                "mean": 0.05,
                "sd": 0.2,
            },
            {
                "type": "linear",
                "input": "temp",
                "column": "temp_z",
                "transform": "none",
                "mean": 27.8,
                "sd": 3.9,
            },
            {
                "type": "circular_degrees",
                "input": "wdir",
                "columns": ["w_sin", "w_cos"],
                "calm": {"input": "wspd", "equals": 0},
            },
            {
                "type": "linear",
                "input": "wspd",
                "column": "wspd_z",
                "mean": 7.2,
                "sd": 5.8,
            },
        ]
        # Deliberately not in spec order: scalar_cols sets the output order
        cols = ["temp_z", "h_sin", "h_cos", "rain_z", "wspd_z", "w_sin", "w_cos"]
    else:
        raise ValueError(n_scalars)
    return spec, cols


def toy_natural(n_scalars: int, n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"hour": rng.uniform(0, 24, n)})
    if n_scalars == 3:
        df["dt"] = rng.uniform(0.4, 0.6, n)
    else:
        df["rain"] = np.where(rng.random(n) < 0.7, 0.0, rng.exponential(2.0, n))
        df["temp"] = rng.normal(28, 4, n)
        df["wspd"] = np.where(rng.random(n) < 0.2, 0.0, rng.uniform(1, 30, n))
        # BoM style: 10-degree steps, north as 360, 0 for calm
        df["wdir"] = np.where(df["wspd"] == 0, 0.0, rng.integers(1, 37, n) * 10.0)
    return df


def _circ_diff(a, b, period):
    return np.abs((np.asarray(a) - np.asarray(b) + period / 2) % period - period / 2)


@pytest.fixture(params=[3, 7], ids=lambda s: f"s{s}")
def n_scalars(request) -> int:
    return request.param


# ---------------------------------------------------------------------------
# ScalarEncoder
# ---------------------------------------------------------------------------


def test_encoder_matches_manual_encoding(n_scalars):
    from deepssf.inference import ScalarEncoder

    spec, cols = toy_spec(n_scalars)
    enc = ScalarEncoder(spec, cols)
    df = toy_natural(n_scalars)
    out = enc.encode(df)
    assert out.dtype == np.float32 and out.shape == (len(df), n_scalars)

    expected = {
        "h_sin": np.sin(2 * np.pi * df["hour"] / 24),
        "h_cos": np.cos(2 * np.pi * df["hour"] / 24),
    }
    if n_scalars == 3:
        expected["dt"] = df["dt"]
    else:
        expected["rain_z"] = (np.log1p(df["rain"]) - 0.05) / 0.2
        expected["temp_z"] = (df["temp"] - 27.8) / 3.9
        expected["wspd_z"] = (df["wspd"] - 7.2) / 5.8
        calm = df["wspd"] == 0
        expected["w_sin"] = np.where(calm, 0.0, np.sin(np.radians(df["wdir"])))
        expected["w_cos"] = np.where(calm, 0.0, np.cos(np.radians(df["wdir"])))
    manual = np.column_stack(
        [np.asarray(expected[c], dtype=float) for c in cols]
    ).astype(np.float32)
    np.testing.assert_array_equal(out, manual)


def test_encoder_round_trip(n_scalars):
    from deepssf.inference import ScalarEncoder

    spec, cols = toy_spec(n_scalars)
    enc = ScalarEncoder(spec, cols)
    df = toy_natural(n_scalars)
    encoded = enc.encode(df)
    back = enc.decode(encoded)

    assert list(back.columns) == [e["input"] for e in spec]
    assert _circ_diff(back["hour"], df["hour"], 24).max() < 1e-4
    if n_scalars == 3:
        np.testing.assert_allclose(back["dt"], df["dt"], atol=1e-6)
    else:
        np.testing.assert_allclose(back["rain"], df["rain"], atol=1e-5)
        np.testing.assert_allclose(back["temp"], df["temp"], atol=1e-4)
        np.testing.assert_allclose(back["wspd"], df["wspd"], atol=1e-4)
        calm = (df["wspd"] == 0).to_numpy()
        assert calm.any() and (df.loc[~calm, "wdir"] == 360).any()  # cases covered
        assert back.loc[calm, "wdir"].isna().all()
        nc = back.loc[~calm, "wdir"].to_numpy()
        assert ((nc >= 0) & (nc < 360)).all()  # 360 comes back as 0
        assert _circ_diff(nc, df.loc[~calm, "wdir"], 360).max() < 1e-3
    # ... and re-encoding the decoded values gives the encoded rows back,
    # including calm rows (NaN direction, wind speed decoded to ~0)
    np.testing.assert_allclose(enc.encode(back), encoded, atol=1e-6)
    # torch input and a single [S] row
    pd.testing.assert_frame_equal(
        enc.decode(torch.from_numpy(encoded[:1])), enc.decode(encoded[0])
    )


def test_encoder_dict_input_broadcasts():
    from deepssf.inference import ScalarEncoder

    enc = ScalarEncoder(*toy_spec(3))
    out = enc.encode({"hour": [0.0, 6.0, 12.0], "dt": 0.5})
    np.testing.assert_allclose(out[:, 2], 0.5)
    np.testing.assert_allclose(out[:, :2], [[0, 1], [1, 0], [0, -1]], atol=1e-6)
    assert enc.encode({"hour": 6.0, "dt": 0.5}).shape == (1, 3)


def test_encoder_spec_validation():
    from deepssf.inference import ScalarEncoder

    spec, cols = toy_spec(7)
    with pytest.raises(ValueError, match=r"missing.*'extra_col'"):
        ScalarEncoder(spec, cols + ["extra_col"])
    with pytest.raises(ValueError, match=r"extra.*'temp_z'"):
        ScalarEncoder(spec, [c for c in cols if c != "temp_z"])
    with pytest.raises(ValueError, match="more than one entry"):
        ScalarEncoder(spec + [{"type": "raw", "input": "x", "column": "temp_z"}], cols)
    with pytest.raises(ValueError, match="repeated in scalar_cols"):
        ScalarEncoder(spec, cols + ["temp_z"])
    with pytest.raises(ValueError, match="type 'polar'"):
        ScalarEncoder([{"type": "polar", "input": "x", "column": "x"}], ["x"])
    with pytest.raises(ValueError, match="transform 'log'"):
        ScalarEncoder(
            [
                {
                    "type": "linear",
                    "input": "x",
                    "column": "x",
                    "transform": "log",
                    "mean": 0,
                    "sd": 1,
                }
            ],
            ["x"],
        )
    with pytest.raises(ValueError, match="sd > 0"):
        ScalarEncoder(
            [{"type": "linear", "input": "x", "column": "x", "mean": 0, "sd": 0}], ["x"]
        )
    with pytest.raises(ValueError, match="missing \\['period'\\]"):
        ScalarEncoder(
            [{"type": "cyclic", "input": "x", "columns": ["a", "b"]}], ["a", "b"]
        )
    with pytest.raises(ValueError, match="2 columns"):
        ScalarEncoder(
            [{"type": "cyclic", "input": "x", "period": 1, "columns": ["a"]}], ["a"]
        )

    enc = ScalarEncoder(spec, cols)
    with pytest.raises(KeyError, match="wspd"):
        enc.encode(toy_natural(7).drop(columns="wspd"))
    bad = toy_natural(7)
    bad.loc[bad.index[bad["wspd"] > 0][0], "wdir"] = (
        np.nan
    )  # not calm, so no rule covers it
    with pytest.raises(ValueError, match="Non-finite.*w_sin"):
        enc.encode(bad)


def test_encoder_ranges(n_scalars):
    from deepssf.inference import ScalarEncoder

    enc = ScalarEncoder(*toy_spec(n_scalars))
    df = toy_natural(n_scalars)
    assert enc.ranges_from(df) is enc
    assert set(enc.ranges) == set(enc.inputs)
    assert enc.check_ranges(df) == []
    outside = df.iloc[:3].copy()
    outside.loc[outside.index[0], "hour"] = 30.0
    msgs = enc.check_ranges(outside)
    assert len(msgs) == 1 and msgs[0].startswith("hour: 1 of 3")
    with pytest.raises(KeyError):
        enc.set_range("not_an_input", 0, 1)


def test_encoder_json_round_trip(tmp_path, n_scalars):
    from deepssf.inference import ScalarEncoder

    enc = ScalarEncoder(*toy_spec(n_scalars)).ranges_from(toy_natural(n_scalars))
    df = toy_natural(n_scalars, seed=5)
    for loaded in (
        ScalarEncoder.from_json(enc.to_json()),
        ScalarEncoder.from_json(
            enc.to_json(tmp_path / "spec.json") and tmp_path / "spec.json"
        ),
        ScalarEncoder.from_json(str(tmp_path / "spec.json")),
    ):
        assert loaded.to_dict() == enc.to_dict()
        np.testing.assert_array_equal(loaded.encode(df), enc.encode(df))


def test_encoder_from_scaling_table(tmp_path):
    from deepssf.inference import ScalarEncoder

    spec, cols = toy_spec(7)
    linear = [e for e in spec if e["type"] == "linear"]
    others = [e for e in spec if e["type"] != "linear"]
    table = pd.DataFrame(
        {
            "column": [e["input"] for e in linear],
            "transform": [e.get("transform", "none") for e in linear],
            "mean": [e["mean"] for e in linear],
            "sd": [e["sd"] for e in linear],
        }
    )
    table.loc[table["transform"] == "none", "transform"] = np.nan  # as read from a CSV
    renamed = [c if not c.endswith("_z") else c[:-2] + "_scaled" for c in cols]
    direct = ScalarEncoder(
        [
            {**e, "column": e["column"][:-2] + "_scaled"}
            if e["type"] == "linear"
            else e
            for e in spec
        ],
        renamed,
    )
    table.to_csv(tmp_path / "scaling.csv", index=False)
    df = toy_natural(7)
    for source in (table, table.set_index("column"), tmp_path / "scaling.csv"):
        enc = ScalarEncoder.from_scaling_table(source, renamed, extra_spec=others)
        np.testing.assert_array_equal(enc.encode(df), direct.encode(df))
    with pytest.raises(ValueError, match="needs columns"):
        ScalarEncoder.from_scaling_table(
            table.drop(columns="sd"), renamed, extra_spec=others
        )


# ---------------------------------------------------------------------------
# crop_windows vs MovementDataset
# ---------------------------------------------------------------------------


def _write_tif(path, data, transform):
    import rasterio

    bands, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        transform=transform,
    ) as dst:
        dst.write(data.astype("float32"))


def test_crop_windows_matches_movement_dataset(tmp_path, toy):
    from rasterio.transform import from_origin

    from deepssf.data import MovementDataset, load_environmental_layers
    from deepssf.inference import crop_windows

    s = toy.shape
    height, width = 2 * s.window + 7, 3 * s.window + 2
    x0, y0, pix = 500_000.0, 8_600_000.0, 20.0
    transform = from_origin(x0, y0, pix, pix)
    rng = np.random.default_rng(3)

    # One single-band layer, plus a multi-band layer for the rest of the
    # channels, so both MovementDataset crop paths are exercised
    layer_paths = {"a": tmp_path / "a.tif"}
    _write_tif(layer_paths["a"], rng.normal(size=(1, height, width)) * 50, transform)
    if s.n_raster > 1:
        layer_paths["b"] = tmp_path / "b.tif"
        _write_tif(
            layer_paths["b"], rng.random((s.n_raster - 1, height, width)), transform
        )
    layer_paths = {k: str(v) for k, v in layer_paths.items()}

    # Interior points, points near / beyond every edge, and exact pixel corners
    n = 40
    xs = np.concatenate(
        [
            rng.uniform(x0, x0 + width * pix, n),
            [
                x0 + 0.1,
                x0 + width * pix - 0.1,
                x0 - 3 * pix,
                x0 + (width + 2) * pix,
                x0 + 7 * pix,
                x0 + 11 * pix,
            ],
        ]
    )
    ys = np.concatenate(
        [
            rng.uniform(y0 - height * pix, y0, n),
            [
                y0 - 0.1,
                y0 - height * pix + 0.1,
                y0 - 5 * pix,
                y0 - 9 * pix,
                y0 - 13 * pix,
                y0,
            ],
        ]
    )
    df = pd.DataFrame(
        {
            "x1_": xs,
            "y1_": ys,
            "x2_": xs + 1.0,
            "y2_": ys - 1.0,
            "t1_": pd.Timestamp("2024-06-01 12:00"),
            "bearing_tm1": 0.0,
            "dt": 0.5,
        }
    )
    ds = MovementDataset(df, layer_paths, window_size=s.window, scalar_cols=["dt"])

    env, env_transform = load_environmental_layers(layer_paths)
    layers = []
    for arr in env.values():
        t = torch.from_numpy(np.asarray(arr).astype(np.float32))
        layers.extend([t] if t.ndim == 2 else list(t))

    windows = crop_windows(layers, xs, ys, env_transform, s.window)
    assert windows.shape == (len(df), s.n_raster, s.window, s.window)
    for i in range(len(df)):
        assert torch.equal(windows[i], ds[i][0]), i
    assert (windows == -1).any()  # some windows really did hang off the edge

    # Stacked input and transform coefficients (as MovementDataset items carry)
    alt = crop_windows(
        torch.stack(layers), xs, ys, ds.raster_transform_coeffs, s.window
    )
    assert torch.equal(alt, windows)
    filled = crop_windows(layers, xs, ys, env_transform, s.window, fill=-7.0)
    assert torch.equal(filled == -7.0, windows == -1.0)


# ---------------------------------------------------------------------------
# HabitatLandscapePredictor vs predict_habitat_landscape
# ---------------------------------------------------------------------------


def _interior(toy: ToyCase) -> np.ndarray:
    from deepssf.predict import habitat_edge_buffer

    b = habitat_edge_buffer(toy.model)
    inside = np.zeros(toy.landscape.shape[1:], dtype=bool)
    inside[b:-b, b:-b] = True
    return inside


def _toy_mask(toy: ToyCase) -> np.ndarray:
    """A blob in one corner (touching the raster edge) plus scattered cells,
    all in the top-left half, so the bounding box is smaller than the raster."""
    h, w = toy.landscape.shape[1:]
    mask = np.zeros((h, w), dtype=bool)
    mask[: h // 3, : w // 4] = True
    rng = np.random.default_rng(4)
    mask[rng.integers(0, h // 2, 15), rng.integers(w // 4, w // 2, 15)] = True
    return mask


def _scenarios(toy: ToyCase, n: int = 4) -> np.ndarray:
    g = torch.Generator().manual_seed(7)
    return torch.randn(n, toy.shape.n_scalars, generator=g).numpy()


def test_predictor_matches_predict_habitat_landscape(toy):
    from deepssf.inference import HabitatLandscapePredictor
    from deepssf.predict import predict_habitat_landscape

    pred = HabitatLandscapePredictor(
        toy.model, toy.landscape, toy.shape.n_scalars, device="cpu"
    )
    assert pred.shape == toy.landscape.shape[1:] and pred.n_raster == toy.shape.n_raster
    scen = _scenarios(toy)
    for normalise in (False, True):
        out = pred.predict(scen, normalise=normalise)
        assert out.shape == (len(scen),) + pred.shape
        for b, s in enumerate(scen):
            ref = predict_habitat_landscape(
                toy.model, toy.landscape, s, normalise=normalise
            )
            np.testing.assert_array_equal(np.isnan(out[b]), np.isnan(ref))
            assert np.isnan(ref).sum() == (~_interior(toy)).sum()
            assert np.nanmax(np.abs(out[b] - ref)) < 1e-5
        if normalise:
            np.testing.assert_allclose(
                np.nansum(np.exp(out), axis=(1, 2)), 1.0, atol=1e-5
            )


@pytest.mark.parametrize("crop", [True, False], ids=["crop", "nocrop"])
def test_predictor_with_mask(toy, crop):
    from deepssf.inference import HabitatLandscapePredictor
    from deepssf.predict import predict_habitat_landscape

    mask = _toy_mask(toy)
    valid = mask & _interior(toy)
    pred = HabitatLandscapePredictor(
        toy.model,
        list(toy.landscape),
        toy.shape.n_scalars,
        mask=mask,
        crop_to_mask=crop,
        device="cpu",
    )
    if crop:
        assert (
            pred.shape[0] < toy.landscape.shape[1]
            or pred.shape[1] < toy.landscape.shape[2]
        )
    scen = _scenarios(toy, 3)
    raw = pred.to_full(pred.predict(scen, normalise=False))
    norm = pred.to_full(pred.predict(scen))
    for b, s in enumerate(scen):
        ref = predict_habitat_landscape(toy.model, toy.landscape, s, normalise=False)
        np.testing.assert_array_equal(~np.isnan(raw[b]), valid)
        assert np.abs(raw[b][valid] - ref[valid]).max() < 1e-5
        # Normalised over the mask's valid cells only
        ref_norm = ref[valid] - np.log(np.exp(ref[valid].astype(np.float64)).sum())
        assert np.abs(norm[b][valid] - ref_norm).max() < 1e-5


def test_predictor_crop_geometry(toy):
    from affine import Affine

    from deepssf.inference import HabitatLandscapePredictor

    mask = _toy_mask(toy)
    pred = HabitatLandscapePredictor(
        toy.model, toy.landscape, toy.shape.n_scalars, mask=mask
    )
    full = pred.to_full(pred.valid.astype(np.float32))
    assert full.shape == toy.landscape.shape[1:]
    assert np.array_equal(full == 1, mask & _interior(toy))
    t = Affine(20.0, 0, 1000.0, 0, -20.0, 5000.0)
    ct = pred.crop_transform(t)
    assert ct * (0, 0) == t * (pred.col_slice.start, pred.row_slice.start)


def test_predictor_batch_equals_loop(toy):
    from deepssf.inference import HabitatLandscapePredictor

    pred = HabitatLandscapePredictor(
        toy.model, toy.landscape, toy.shape.n_scalars, device="cpu"
    )
    scen = _scenarios(toy, 5)
    batched = pred.predict(scen)
    looped = np.stack([pred.predict(s)[0] for s in scen])
    np.testing.assert_allclose(batched, looped, atol=1e-6, equal_nan=True)
    chunked = pred.predict(scen, batch_size=2)
    np.testing.assert_allclose(chunked, batched, atol=1e-6, equal_nan=True)
    # The memory-aware batch size shrinks with the byte budget
    small = HabitatLandscapePredictor(
        toy.model, toy.landscape, toy.shape.n_scalars, max_batch_bytes=1
    )
    assert small.batch_size() == 1
    np.testing.assert_allclose(small.predict(scen), batched, atol=1e-6, equal_nan=True)
    tensor_out = pred.predict(torch.from_numpy(scen), as_numpy=False)
    assert isinstance(tensor_out, torch.Tensor)
    np.testing.assert_allclose(tensor_out.numpy(), batched, atol=1e-6, equal_nan=True)


def test_predictor_input_errors(toy):
    from deepssf.inference import HabitatLandscapePredictor

    s = toy.shape
    with pytest.raises(ValueError, match="raster channels"):
        HabitatLandscapePredictor(
            toy.model,
            toy.landscape[:-1] if s.n_raster > 1 else torch.cat([toy.landscape] * 2),
            s.n_scalars,
        )
    with pytest.raises(ValueError, match="No valid cells"):
        empty = np.zeros(toy.landscape.shape[1:], dtype=bool)
        empty[0, 0] = True  # only inside the edge buffer
        HabitatLandscapePredictor(toy.model, toy.landscape, s.n_scalars, mask=empty)
    pred = HabitatLandscapePredictor(toy.model, toy.landscape, s.n_scalars)
    with pytest.raises(ValueError, match="scalar rows"):
        pred.predict(np.zeros(s.n_scalars + 1))


def test_encoder_to_predictor_end_to_end(toy):
    """Natural rows -> ScalarEncoder -> predictor equals the brute-force path."""
    from deepssf.inference import HabitatLandscapePredictor, ScalarEncoder
    from deepssf.predict import predict_habitat_landscape

    enc = ScalarEncoder(*toy_spec(toy.shape.n_scalars))
    encoded = enc.encode(toy_natural(toy.shape.n_scalars, n=3))
    pred = HabitatLandscapePredictor(toy.model, toy.landscape, toy.shape.n_scalars)
    out = pred.predict(encoded)
    for b, s in enumerate(encoded):
        ref = predict_habitat_landscape(toy.model, toy.landscape, s)
        assert np.nanmax(np.abs(out[b] - ref)) < 1e-5


def _accelerators() -> list[str]:
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.skipif(not _accelerators(), reason="no CUDA or MPS device")
@pytest.mark.parametrize("device", _accelerators() or ["none"])
def test_predictor_matches_on_accelerator(toy, device):
    """Same identity on a GPU: guards against device-specific kernel bugs
    (MPS once returned wrong values for a conv with a sliced weight)."""
    import copy

    from deepssf.inference import HabitatLandscapePredictor
    from deepssf.predict import predict_habitat_landscape

    model = copy.deepcopy(toy.model).to(device)
    scen = _scenarios(toy, 3)
    for mask in (None, _toy_mask(toy)):
        pred = HabitatLandscapePredictor(
            model, toy.landscape, toy.shape.n_scalars, mask=mask
        )
        out = pred.to_full(pred.predict(scen, normalise=False))
        for b, s in enumerate(scen):
            ref = predict_habitat_landscape(model, toy.landscape, s, normalise=False)
            valid = ~np.isnan(out[b])
            assert np.abs(out[b][valid] - ref[valid]).max() < 1e-4
