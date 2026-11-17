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


# ---------------------------------------------------------------------------
# availability_mask
# ---------------------------------------------------------------------------


def _mask_points(height, width, pix=20.0, n=12, seed=0):
    from rasterio.transform import from_origin

    x0, y0 = 1000.0, 9000.0
    transform = from_origin(x0, y0, pix, pix)
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x0 - 3 * pix, x0 + (width + 3) * pix, n)  # some off the raster
    ys = rng.uniform(y0 - (height + 3) * pix, y0 + 3 * pix, n)
    return transform, xs, ys


def _brute_mask(height, width, transform, xs, ys, r, shape):
    px, py = ~transform * (np.asarray(xs), np.asarray(ys))
    px, py = np.floor(px).astype(int), np.floor(py).astype(int)
    rows, cols = np.mgrid[0:height, 0:width]
    out = np.zeros((height, width), dtype=bool)
    for x, y in zip(px, py, strict=True):
        if shape == "square":
            out |= (np.abs(rows - y) <= r) & (np.abs(cols - x) <= r)
        else:
            out |= (rows - y) ** 2 + (cols - x) ** 2 <= r * r
    return out


@pytest.mark.parametrize("shape", ["square", "disc"])
@pytest.mark.parametrize("r", [0, 1, 4, 10])
def test_availability_mask_matches_brute_force(shape, r):
    from deepssf.inference import availability_mask

    height, width = 37, 52
    transform, xs, ys = _mask_points(height, width)
    mask = availability_mask((height, width), transform, xs, ys, r, shape=shape)
    assert mask.dtype == bool and mask.shape == (height, width)
    np.testing.assert_array_equal(
        mask, _brute_mask(height, width, transform, xs, ys, r, shape)
    )


def test_availability_mask_is_union_of_model_windows(toy):
    """Square mask of radius window // 2 == every cell some input window covers."""
    from deepssf.inference import availability_mask, crop_windows

    s = toy.shape
    height, width = toy.landscape.shape[1:]
    transform, xs, ys = _mask_points(height, width, n=6, seed=2)
    mask = availability_mask((height, width), transform, xs, ys, s.window // 2)

    # Crop windows of a raster holding each cell's flat index, then mark them
    index = torch.arange(height * width, dtype=torch.float32).reshape(1, height, width)
    windows = crop_windows(index, xs, ys, transform, s.window)
    covered = np.zeros(height * width, dtype=bool)
    ids = windows[windows >= 0].long().numpy()
    covered[ids] = True
    np.testing.assert_array_equal(mask, covered.reshape(height, width))


def test_availability_mask_errors():
    from deepssf.inference import availability_mask

    transform, xs, ys = _mask_points(10, 10)
    with pytest.raises(ValueError, match="radius_px"):
        availability_mask((10, 10), transform, xs, ys, -1)
    with pytest.raises(ValueError, match="shape"):
        availability_mask((10, 10), transform, xs, ys, 2, shape="hexagon")


# ---------------------------------------------------------------------------
# class_selection
# ---------------------------------------------------------------------------


def _one_hot_classes(height, width, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, n_classes, (height, width))
    return (labels[None] == np.arange(n_classes)[:, None, None]).astype(np.float32)


def test_class_selection_uniform_surface_gives_sr_one():
    from deepssf.inference import class_selection

    classes = _one_hot_classes(20, 30)
    surf = np.full((20, 30), -np.log(600.0), dtype=np.float32)
    df = class_selection(surf, classes, ["a", "b", "c", "d"])
    assert list(df.columns) == [
        "scenario",
        "class",
        "use",
        "availability",
        "selection_ratio",
        "log_sr",
    ]
    assert len(df) == 4 and (df["scenario"] == 0).all()
    np.testing.assert_allclose(df["selection_ratio"], 1.0, atol=1e-12)
    np.testing.assert_allclose(df["log_sr"], 0.0, atol=1e-12)
    np.testing.assert_allclose(df["availability"], classes.mean(axis=(1, 2)))
    # An unnormalised constant surface gives the same answer
    df2 = class_selection(surf + 5.0, classes)
    np.testing.assert_allclose(df2["selection_ratio"], 1.0, atol=1e-12)


def test_class_selection_all_mass_in_one_class_and_mask():
    from deepssf.inference import class_selection

    classes = _one_hot_classes(20, 30, seed=1)
    in_b = classes[1] == 1
    surf = np.where(in_b, 0.0, -np.inf).astype(np.float32)
    surf[0, :] = np.nan  # invalid row
    mask = np.ones((20, 30), dtype=bool)
    mask[:, :5] = False
    df = class_selection(
        np.stack([surf, surf]),
        classes,
        list("abcd"),
        mask=mask,
        scenario_ids=["s1", "s2"],
    )
    assert list(df["scenario"].unique()) == ["s1", "s2"]
    valid = mask.copy()
    valid[0, :] = False
    avail = classes[:, valid].mean(axis=1)
    for _, sub in df.groupby("scenario"):
        np.testing.assert_allclose(sub["use"], [0, 1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(sub["availability"], avail)
        np.testing.assert_allclose(sub["selection_ratio"].iloc[1], 1 / avail[1])
        assert np.isneginf(sub["log_sr"].iloc[0])


def test_class_selection_zero_availability_is_nan_and_torch_matches_numpy():
    from deepssf.inference import class_selection

    classes = _one_hot_classes(15, 15, n_classes=3, seed=2)
    classes = np.concatenate([classes, np.zeros((1, 15, 15), np.float32)])  # empty
    rng = np.random.default_rng(3)
    surf = rng.normal(size=(2, 15, 15)).astype(np.float32)
    df = class_selection(surf, classes)
    empty = df[df["class"] == "class_3"]
    assert (
        empty[["use", "availability", "selection_ratio", "log_sr"]].isna().all().all()
    )
    # use sums to 1 over the classes when they cover every cell
    np.testing.assert_allclose(df.groupby("scenario")["use"].sum(), 1.0)
    dft = class_selection(torch.from_numpy(surf), torch.from_numpy(classes))
    pd.testing.assert_frame_equal(df, dft)
    with pytest.raises(ValueError, match="do not match"):
        class_selection(surf, classes[:, :10])
    with pytest.raises(ValueError, match="class names"):
        class_selection(surf, classes, ["a"])


def test_class_selection_on_toy_predictor(toy):
    """Predictor surfaces + class_selection == brute force from the reference."""
    from deepssf.inference import HabitatLandscapePredictor, class_selection
    from deepssf.predict import predict_habitat_landscape

    height, width = toy.landscape.shape[1:]
    classes = _one_hot_classes(height, width, n_classes=3, seed=5)
    mask = _toy_mask(toy)
    pred = HabitatLandscapePredictor(
        toy.model, toy.landscape, toy.shape.n_scalars, mask=mask
    )
    scen = _scenarios(toy, 3)
    crop = classes[:, pred.row_slice, pred.col_slice]
    df = class_selection(pred.predict(scen), crop)
    for b, s in enumerate(scen):
        ref = predict_habitat_landscape(toy.model, toy.landscape, s).astype(np.float64)
        valid = ~np.isnan(ref) & mask
        p = np.exp(ref[valid] - ref[valid].max())
        p /= p.sum()
        use = classes[:, valid] @ p
        avail = classes[:, valid].mean(axis=1)
        got = df[df["scenario"] == b]
        np.testing.assert_allclose(got["use"], use, atol=1e-5)
        np.testing.assert_allclose(got["log_sr"], np.log(use / avail), atol=1e-4)


# ---------------------------------------------------------------------------
# Partial dependence, ALE, binned averages, H
# ---------------------------------------------------------------------------


def _raw_encoder(names=("x1", "x2", "x3")):
    from deepssf.inference import ScalarEncoder

    return ScalarEncoder(
        [{"type": "raw", "input": n, "column": n} for n in names], list(names)
    )


def _toy_metric(encoded):
    """Known structure: additive (2 x1 + x2²), interaction (x1 x2), linear x3."""
    x1, x2, x3 = encoded[:, 0], encoded[:, 1], encoded[:, 2]
    return pd.DataFrame(
        {"add": 2 * x1 + x2.astype(float) ** 2, "inter": x1 * x2, "lin3": x3}
    )


def _toy_rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.uniform(0, 1, n),
            "x2": rng.uniform(0, 1, n),
            "x3": rng.uniform(0, 1, n),
        },
        index=pd.RangeIndex(100, 100 + n),
    )


def test_partial_dependence_recovers_known_structure():
    from deepssf.inference import partial_dependence

    enc = _raw_encoder()
    rows = _toy_rows()
    grid = np.linspace(0, 1, 5)
    pd_df, ice_df = partial_dependence(_toy_metric, rows, "x1", grid, enc, batch_size=7)
    assert set(pd_df.columns) >= {"x1", "metric", "pd", "sd", "n", "frac_out_of_range"}
    add = pd_df[pd_df["metric"] == "add"]
    np.testing.assert_allclose(
        add["pd"], 2 * grid + (rows["x2"] ** 2).mean(), rtol=1e-6
    )
    # ICE of each row is its own curve: 2 v + x2_i²
    ice = ice_df[(ice_df["metric"] == "add") & (ice_df["ice_id"] == 103)]
    np.testing.assert_allclose(
        ice["ice"], 2 * grid + rows.loc[103, "x2"] ** 2, rtol=1e-6
    )
    assert len(ice_df) == len(grid) * len(rows) * 3
    # A grid column named like an output column is refused, not overwritten
    with pytest.raises(ValueError, match="clash"):
        partial_dependence(_toy_metric, rows, "x1", grid, enc, grid_name="ice")
    # No training ranges recorded -> nothing out of range
    assert (pd_df["frac_out_of_range"] == 0).all()


def test_partial_dependence_set_fn_coupling_and_2d_grid():
    from deepssf.inference import partial_dependence

    enc = _raw_encoder().ranges_from(_toy_rows())
    rows = _toy_rows()

    def couple(r, v):  # x3 follows x1, as dew point/wet bulb follow temperature
        return r.assign(x1=v, x3=2 * v)

    grid = np.array([0.0, 0.25, 0.5])
    pd_df, _ = partial_dependence(_toy_metric, rows, couple, grid, enc, grid_name="x1")
    np.testing.assert_allclose(pd_df.loc[pd_df["metric"] == "lin3", "pd"], 2 * grid)
    # x3 = 2 * 0.5 = 1 lies above x3's training range for every row
    oor = pd_df.loc[pd_df["metric"] == "lin3", "frac_out_of_range"].to_numpy()
    assert oor[0] == 1.0 or oor[-1] == 1.0

    grid2 = pd.DataFrame({"x1": [0.0, 1.0, 0.0, 1.0], "x2": [0.0, 0.0, 1.0, 1.0]})
    pd2, _ = partial_dependence(
        _toy_metric, rows, lambda r, v: r.assign(x1=v["x1"], x2=v["x2"]), grid2, enc
    )
    inter = pd2[pd2["metric"] == "inter"]
    assert list(inter.columns[:2]) == ["x1", "x2"]
    np.testing.assert_allclose(inter["pd"], [0, 0, 0, 1])


def test_ale_recovers_known_structure():
    from deepssf.inference import accumulated_local_effects

    enc = _raw_encoder()
    rows = _toy_rows(n=400, seed=1)
    ale = accumulated_local_effects(_toy_metric, rows, "x2", None, enc, n_bins=8)
    add = ale[ale["metric"] == "add"]
    edges = add["x2"].to_numpy()
    # x2 enters additively as x2², so ALE(z) - ALE(z0) = z² - z0² (up to the
    # encoder's float32)
    np.testing.assert_allclose(
        add["ale"] - add["ale"].iloc[0], edges**2 - edges[0] ** 2, atol=1e-6
    )
    # Centred: count-weighted mean of bin midpoints is 0
    a, n = add["ale"].to_numpy(), add["n"].to_numpy()[1:]
    assert abs((n * (a[:-1] + a[1:]) / 2).sum() / n.sum()) < 1e-12
    assert n.sum() == len(rows)
    # x3 does not enter "add" at all
    ale3 = accumulated_local_effects(_toy_metric, rows, "x3", None, enc)
    np.testing.assert_allclose(ale3.loc[ale3["metric"] == "add", "ale"], 0, atol=1e-12)
    # A derived variable via values + set_fn
    ale_v = accumulated_local_effects(
        _toy_metric,
        rows,
        "half_x1",
        lambda r, v: r.assign(x1=2 * np.asarray(v)),
        enc,
        values=rows["x1"] / 2,
    )
    lin = ale_v[ale_v["metric"] == "add"]
    np.testing.assert_allclose(
        np.diff(lin["ale"]), 4 * np.diff(lin["half_x1"]), atol=1e-6
    )


def test_h_statistic_zero_for_additive_positive_for_interaction():
    from deepssf.inference import h_statistic

    enc = _raw_encoder()
    rows = _toy_rows(n=15, seed=2)
    h = h_statistic(_toy_metric, rows, enc, "x1", "x2").set_index("metric")
    assert h.loc["add", "h"] < 1e-6
    assert h.loc["inter", "h"] > 0.2


def test_binned_average_mirrors_bin_weather():
    from deepssf.inference import binned_average

    enc = _raw_encoder()
    rng = np.random.default_rng(3)
    n = 2000
    rows = pd.DataFrame(
        {
            "x1": np.where(
                rng.random(n) < 0.5, 0.0, rng.exponential(1.0, n)
            ),  # 50% zeros
            "x2": rng.integers(0, 4, n).astype(float),  # 4 values: categorical
            "x3": rng.uniform(0, 1, n),
        }
    )
    out = binned_average(_toy_metric, rows, "x1", enc, n_bins=5)
    lin3 = out[out["metric"] == "lin3"]
    assert lin3["bin"].iloc[0] == "0" and lin3["n"].iloc[0] == (rows["x1"] == 0).sum()
    assert lin3["bin"].iloc[1].startswith("[") and lin3["bin"].iloc[2].startswith("(")
    assert lin3["n"].sum() == n and len(lin3) == 6
    # Bin means equal a direct groupby on the metric
    zero = rows["x1"] == 0
    np.testing.assert_allclose(lin3["mean"].iloc[0], rows.loc[zero, "x3"].mean())

    cat = binned_average(_toy_metric, rows, "x2", enc)
    assert sorted(cat["bin"].unique()) == ["0", "1", "2", "3"]
    # min_n drops thin bins; precomputed metrics give the same answer
    thin = binned_average(_toy_metric, rows, "x3", enc, n_bins=10, min_n=n)
    assert thin.empty
    assert "mean" in thin.columns  # empty, but still has its columns
    pre = binned_average(None, rows, "x3", enc, metrics=_toy_metric(rows.to_numpy()))
    np.testing.assert_allclose(
        pre["mean"], binned_average(_toy_metric, rows, "x3", enc)["mean"]
    )


def test_partial_dependence_on_toy_model_matches_brute_force(toy):
    """PDP of log SR from the predictor equals a brute-force loop."""
    from deepssf.inference import (
        HabitatLandscapePredictor,
        ScalarEncoder,
        class_selection,
        partial_dependence,
    )
    from deepssf.predict import predict_habitat_landscape

    enc = ScalarEncoder(*toy_spec(toy.shape.n_scalars))
    base = toy_natural(toy.shape.n_scalars, n=3)
    height, width = toy.landscape.shape[1:]
    classes = _one_hot_classes(height, width, n_classes=3, seed=6)
    pred = HabitatLandscapePredictor(toy.model, toy.landscape, toy.shape.n_scalars)

    def metric(encoded):
        df = class_selection(pred.predict(encoded), classes)
        return df.pivot(index="scenario", columns="class", values="log_sr")

    grid = [3.0, 15.0]
    pd_df, ice_df = partial_dependence(metric, base, "hour", grid, enc, batch_size=2)
    for hour in grid:
        rows = base.assign(hour=hour)
        brute = []
        for s in enc.encode(rows):
            surf = predict_habitat_landscape(toy.model, toy.landscape, s)
            brute.append(class_selection(surf, classes)["log_sr"].to_numpy())
        got = pd_df[pd_df["hour"] == hour].set_index("metric")["pd"]
        np.testing.assert_allclose(got.to_numpy(), np.mean(brute, axis=0), atol=1e-4)


# ---------------------------------------------------------------------------
# step_nll and permutation importance
# ---------------------------------------------------------------------------


def _toy_steps(toy: ToyCase, n=24, seed=0):
    """(spatial, scalars, bearing, targets) tensors matching the toy model."""
    s = toy.shape
    g = torch.Generator().manual_seed(seed)
    spatial = torch.rand(n, s.n_raster, s.window, s.window, generator=g)
    scalars = torch.randn(n, s.n_scalars, generator=g)
    bearing = (torch.rand(n, 1, generator=g) - 0.5) * 2 * np.pi
    targets = torch.randint(0, s.window, (n, 2), generator=g)
    return spatial, scalars, bearing, targets


def test_step_nll_matches_training_loss(toy):
    from deepssf.inference import step_nll
    from deepssf.train import negativeLogLikeLoss

    spatial, scalars, bearing, targets = _toy_steps(toy)
    nll = step_nll(toy.model, spatial, scalars, bearing, targets)
    with torch.no_grad():
        out = toy.model((spatial, scalars, bearing))
    total, hab, mov = negativeLogLikeLoss(reduction="none")(
        out, (targets[:, 0], targets[:, 1])
    )
    torch.testing.assert_close(nll["joint"], total)
    torch.testing.assert_close(nll["habitat"], hab)
    torch.testing.assert_close(nll["movement"], mov)
    # (px, py) tuple targets give the same answer
    nll2 = step_nll(
        toy.model, spatial, scalars, bearing, (targets[:, 0], targets[:, 1])
    )
    torch.testing.assert_close(nll2["joint"], nll["joint"])


def test_apply_permutation_keeps_groups_paired():
    from deepssf.inference import _apply_permutation

    n = 10
    ids = torch.arange(n, dtype=torch.float32)
    scalars = torch.stack([ids, ids * 10, -ids], dim=1)  # cols 0 and 1 pair up
    spatial = ids.reshape(n, 1, 1, 1).expand(n, 3, 2, 2).clone()
    spatial[:, 2] *= -1
    idx = torch.arange(n)
    source = torch.from_numpy(np.random.default_rng(0).permutation(n))
    sp, sc = _apply_permutation(spatial, scalars, idx, source, [0, 1], [0, 1])
    torch.testing.assert_close(sc[:, 1], sc[:, 0] * 10)  # still a pair
    torch.testing.assert_close(sc[:, 0], ids[source])  # from the source step
    torch.testing.assert_close(sc[:, 2], -ids)  # other columns untouched
    torch.testing.assert_close(sp[:, 0], sp[:, 1])  # raster channels moved together
    torch.testing.assert_close(sp[:, 0, 0, 0], ids[source])
    torch.testing.assert_close(sp[:, 2, 0, 0], -ids)
    assert torch.equal(scalars[:, 0], ids)  # inputs not modified in place


def test_permutation_importance_ignored_channel_is_zero(toy):
    import copy

    from deepssf.inference import permutation_importance

    s = toy.shape
    model = copy.deepcopy(toy.model)
    with torch.no_grad():
        # Zero the first raster channel and the last scalar in both heads
        for conv in (model.conv_habitat.conv2d[0], model.conv_movement.conv2d[0]):
            conv.weight[:, 0] = 0.0
            conv.weight[:, -1] = 0.0
    raster_names = [f"r{i}" for i in range(s.n_raster)]
    scalar_cols = [f"s{i}" for i in range(s.n_scalars)]
    groups = {
        "ignored_raster": ["r0"],
        "ignored_scalar": [scalar_cols[-1]],
        "used_scalars": scalar_cols[:2],
    }
    data = _toy_steps(toy, n=40)
    res = permutation_importance(
        model,
        data,
        groups,
        scalar_cols,
        raster_names,
        n_repeats=3,
        batch_size=16,
        seed=1,
        device="cpu",
    )
    assert set(res["head"]) == {"habitat", "movement", "joint"}
    ignored = res[res["group"].str.startswith("ignored")]
    np.testing.assert_allclose(ignored["increase"], 0.0, atol=1e-9)
    used = res[(res["group"] == "used_scalars") & (res["head"] == "joint")]
    assert abs(used["increase"].iloc[0]) > 1e-6
    # Deterministic under a seed; a different seed gives different permutations
    again = permutation_importance(
        model,
        data,
        groups,
        scalar_cols,
        raster_names,
        n_repeats=3,
        batch_size=16,
        seed=1,
        device="cpu",
    )
    pd.testing.assert_frame_equal(again, res)  # same seed, same batches: identical
    other = permutation_importance(
        model,
        data,
        {"used_scalars": scalar_cols[:2]},
        scalar_cols,
        raster_names,
        n_repeats=3,
        seed=2,
        device="cpu",
    )
    assert not np.allclose(
        other["permuted_nll"],
        used_val := res.loc[res["group"] == "used_scalars", "permuted_nll"].to_numpy(),
    )
    assert used_val.shape == (3,)
    with pytest.raises(ValueError, match="unknown"):
        permutation_importance(
            model, data, {"bad": ["nope"]}, scalar_cols, raster_names
        )


def test_permutation_importance_on_movement_dataset(tmp_path):
    """MovementDataset input: baseline equals the mean step_nll of its items."""
    from rasterio.transform import from_origin

    from deepssf.data import MovementDataset
    from deepssf.inference import permutation_importance, step_nll

    shape = TOY_SHAPES[0]
    model = build_toy_model(shape)
    transform = from_origin(0.0, 2000.0, 25.0, 25.0)
    rng = np.random.default_rng(0)
    path = tmp_path / "r.tif"
    _write_tif(path, rng.random((1, 60, 60)), transform)
    n = 12
    x1 = rng.uniform(400, 1100, n)
    y1 = rng.uniform(900, 1600, n)
    df = pd.DataFrame(
        {
            "x1_": x1,
            "y1_": y1,
            "x2_": x1 + rng.uniform(-100, 100, n),
            "y2_": y1 + rng.uniform(-100, 100, n),
            "t1_": pd.Timestamp("2024-01-01"),
            "bearing_tm1": rng.uniform(-3, 3, n),
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
        }
    )
    ds = MovementDataset(
        df, {"r": str(path)}, window_size=shape.window, scalar_cols=["a", "b", "c"]
    )
    res = permutation_importance(
        model,
        ds,
        {"time": ["a", "b"]},
        ["a", "b", "c"],
        ["r"],
        n_repeats=2,
        batch_size=5,
        device="cpu",
    )
    items = [ds[i] for i in range(n)]
    nll = step_nll(
        model,
        torch.stack([it[0] for it in items]),
        torch.stack([it[1] for it in items]),
        torch.stack([it[2] for it in items]),
        torch.tensor([[it[3][0], it[3][1]] for it in items]),
    )
    base = res.set_index("head")["baseline_nll"]
    for head in ("habitat", "movement", "joint"):
        assert base[head] == pytest.approx(float(nll[head].double().mean()), rel=1e-6)


# ---------------------------------------------------------------------------
# Movement kernel
# ---------------------------------------------------------------------------


def test_movement_parameters_reproduce_forward(toy):
    from deepssf.inference import movement_parameters

    spatial, scalars, bearing, _ = _toy_steps(toy, n=6)
    m = toy.model
    dec = movement_parameters(m, spatial, scalars)
    assert dec["raw"].shape == (6, 12)
    for key in (
        "gamma_shape",
        "gamma_scale",
        "gamma_weight",
        "vm_mu_offset",
        "vm_kappa",
        "vm_weight",
    ):
        assert dec[key].shape == (6, 2), key
    torch.testing.assert_close(dec["gamma_weight"].sum(1), torch.ones(6))
    with torch.no_grad():
        forward_move = m((spatial, scalars, bearing))[..., 1]
        rebuilt = m.movement_grid_output(dec["raw"], bearing)
    torch.testing.assert_close(rebuilt, forward_move)

    # The decoded values rebuild the same grid through the block's own densities,
    # so shapes, scales (incl. the x500 factor), weights and offsets are right
    block = m.movement_grid_output
    dist, brg = block.distance_layer, block.bearing_layer
    gam = torch.logsumexp(
        torch.stack(
            [
                torch.log(dec["gamma_weight"][:, j, None, None])
                + block._gamma_log(
                    dist,
                    dec["gamma_shape"][:, j, None, None],
                    dec["gamma_scale"][:, j, None, None],
                )
                for j in range(2)
            ]
        ),
        dim=0,
    )
    vm = torch.logsumexp(
        torch.stack(
            [
                torch.log(dec["vm_weight"][:, j, None, None])
                + block._vonmises_log(
                    brg,
                    dec["vm_kappa"][:, j, None, None],
                    (dec["vm_mu_offset"][:, j] + bearing[:, 0])[:, None, None],
                )
                for j in range(2)
            ]
        ),
        dim=0,
    )
    grid = gam + vm
    grid = grid - torch.logsumexp(grid, dim=(1, 2), keepdim=True)
    torch.testing.assert_close(grid, forward_move, atol=1e-4, rtol=1e-4)


def test_movement_parameters_rejects_unknown_layout():
    from deepssf.inference import movement_parameters
    from deepssf.model import ConvJointModel, ModelParams

    shape = TOY_SHAPES[0]
    params = build_toy_model(shape).params
    model = ConvJointModel(ModelParams({**params.__dict__, "num_movement_params": 13}))
    spatial = torch.rand(2, shape.n_raster, shape.window, shape.window)
    with pytest.raises(ValueError, match="layout has 12"):
        movement_parameters(model, spatial, torch.zeros(2, shape.n_scalars))


def _grid_block(window, pix=20.0, jacobian=True):
    """A movement grid block for any window (ModelParams needs a full set)."""
    from deepssf.model import (
        ModelParams,
        Params_to_Grid_Block,
        Params_to_Grid_Block_ChV,
    )

    cls = Params_to_Grid_Block_ChV if jacobian else Params_to_Grid_Block
    base = build_toy_model(TOY_SHAPES[0]).params.__dict__
    return cls(ModelParams({**base, "image_dim": window, "pixel_size": pix}))


def _grids(window, pix=20.0):
    block = _grid_block(window, pix)
    return block.distance_layer, block.bearing_layer


@pytest.mark.parametrize("window", [21, 33])
def test_kernel_summary_on_synthetic_kernels(window):
    from deepssf.inference import kernel_summary

    pix = 20.0
    dist, brg = _grids(window, pix)
    c = window // 2
    neg = torch.full((window, window), -torch.inf)

    centre = neg.clone()
    centre[c, c] = 0.0
    east3 = neg.clone()
    east3[c, c + 3] = 0.0  # 3 cells east
    uniform = torch.zeros(window, window)
    df = kernel_summary(
        torch.stack([centre, east3, uniform]),
        dist,
        brg,
        bearing_tm1=[0.0, np.pi / 2, 0.0],
    )
    a, b, u = df.iloc[0], df.iloc[1], df.iloc[2]
    assert a["p_stay"] == pytest.approx(1.0)
    assert a["expected_distance"] == pytest.approx(0.3826 * pix, rel=1e-5)
    assert a["outer_ring_mass"] == 0
    assert b["expected_distance"] == pytest.approx(3 * pix)
    assert b["sd_distance"] == pytest.approx(0, abs=1e-6)
    assert b["q0.5"] == pytest.approx(3 * pix) and b["p_stay"] == 0
    assert b["turn_resultant"] == pytest.approx(1.0)
    assert b["turn_mean_angle"] == pytest.approx(-np.pi / 2)  # east after heading north
    assert u["expected_distance"] == pytest.approx(
        float(dist.double().mean()), rel=1e-9
    )
    assert u["outer_ring_mass"] == pytest.approx((4 * window - 4) / window**2)
    assert u["turn_resultant"] < 1e-6  # float32 grids
    assert u["p_stay"] == pytest.approx(1 / window**2)

    classes = torch.zeros(2, window, window)
    classes[0, :, : c + 1] = 1  # west half incl. centre column
    classes[1, :, c + 1 :] = 1
    dfc = kernel_summary(
        east3, dist, brg, class_windows=classes, class_names=["w", "e"]
    )
    assert dfc["p_w"].iloc[0] == 0 and dfc["p_e"].iloc[0] == pytest.approx(1.0)


def test_mixture_moments_known_values():
    from deepssf.inference import mixture_moments

    big = 30.0
    dec = {
        "gamma_shape": torch.tensor([[2.0, 1.0]]),
        "gamma_scale": torch.tensor([[50.0, 400.0]]),
        "gamma_weight": torch.softmax(torch.tensor([[big, -big]]), dim=1),
        "vm_mu_offset": torch.tensor([[0.3, 0.0]]),
        "vm_kappa": torch.tensor([[2.0, 0.1]]),
        "vm_weight": torch.softmax(torch.tensor([[big, -big]]), dim=1),
    }
    mm = mixture_moments(dec).iloc[0]
    assert mm["gamma_mean"] == pytest.approx(100.0)
    assert mm["gamma_sd"] == pytest.approx(np.sqrt(2) * 50.0)
    assert mm["gamma2_mean"] == pytest.approx(400.0)
    i0, i1 = torch.special.i0(torch.tensor(2.0)), torch.special.i1(torch.tensor(2.0))
    assert mm["vm_resultant"] == pytest.approx(float(i1 / i0), rel=1e-6)
    assert mm["vm_mean_turn"] == pytest.approx(0.3)


@pytest.mark.parametrize("jacobian", [True, False], ids=["ChV", "no_jacobian"])
def test_kernel_summary_matches_mixture_for_short_steps(jacobian):
    """For a kernel well inside the window, grid and analytic means agree."""
    from deepssf.inference import kernel_summary, mixture_moments

    window, pix = 101, 20.0
    block = _grid_block(window, pix, jacobian)
    raw = torch.zeros(1, 12)
    raw[0, 0] = np.log(4.0)  # shape 4
    raw[0, 1] = np.log(30.0)  # scale 30 m -> mean 120 m, far inside 1 km
    raw[0, 2], raw[0, 5] = 20.0, -20.0  # all weight on component 1
    raw[0, 7] = np.log(1.5)
    raw[0, 8], raw[0, 11] = 20.0, -20.0
    grid = block(raw, torch.zeros(1, 1))
    dec = {
        "gamma_shape": torch.tensor([[4.0, 1.0]]),
        "gamma_scale": torch.tensor([[30.0, 500.0]]),
        "gamma_weight": torch.softmax(torch.tensor([[20.0, -20.0]]), dim=1),
        "vm_mu_offset": torch.zeros(1, 2),
        "vm_kappa": torch.tensor([[1.5, 1.0]]),
        "vm_weight": torch.softmax(torch.tensor([[20.0, -20.0]]), dim=1),
        "polar_jacobian": jacobian,
    }
    ks = kernel_summary(grid, block.distance_layer, block.bearing_layer).iloc[0]
    mm = mixture_moments(dec).iloc[0]
    assert ks["expected_distance"] == pytest.approx(mm["gamma_mean"], rel=0.03)
    assert ks["turn_resultant"] == pytest.approx(mm["vm_resultant"], rel=0.05)


@pytest.mark.skipif(not _accelerators(), reason="no CUDA or MPS device")
@pytest.mark.parametrize("device", _accelerators() or ["none"])
def test_step_and_kernel_functions_on_accelerator(toy, device):
    """No float64 on MPS: these must move results to the CPU before widening."""
    import copy

    from deepssf.inference import (
        class_selection,
        kernel_summary,
        mixture_moments,
        movement_parameters,
        permutation_importance,
        step_nll,
    )

    model = copy.deepcopy(toy.model).to(device)
    spatial, scalars, bearing, targets = (t.to(device) for t in _toy_steps(toy, n=12))
    cpu_nll = step_nll(
        toy.model, *(t.cpu() for t in (spatial, scalars, bearing, targets))
    )
    dev_nll = step_nll(model, spatial, scalars, bearing, targets)
    torch.testing.assert_close(
        dev_nll["joint"].cpu(), cpu_nll["joint"], atol=1e-4, rtol=1e-4
    )

    s = toy.shape
    names = [f"s{i}" for i in range(s.n_scalars)]
    res = permutation_importance(
        model,
        (spatial, scalars, bearing, targets),
        {"g": names[:1]},
        names,
        [f"r{i}" for i in range(s.n_raster)],
        n_repeats=2,
        device=device,
    )
    assert np.isfinite(res["increase"]).all()

    dec = movement_parameters(model, spatial, scalars)
    mm = mixture_moments(dec)
    mm_cpu = mixture_moments(
        movement_parameters(toy.model, spatial.cpu(), scalars.cpu())
    )
    np.testing.assert_allclose(mm["gamma_mean"], mm_cpu["gamma_mean"], rtol=1e-4)
    block = model.movement_grid_output
    with torch.no_grad():
        move = model((spatial, scalars, bearing))[..., 1]
    ks = kernel_summary(move, block.distance_layer, block.bearing_layer, bearing[:, 0])
    assert np.isfinite(ks.to_numpy()).all()
    df = class_selection(move, torch.ones(2, s.window, s.window, device=device) / 2)
    assert np.isfinite(df["use"]).all()


# ---------------------------------------------------------------------------
# Window edge masking (habitat CNN zero-padding artefacts)
# ---------------------------------------------------------------------------


def test_mask_window_edges():
    from deepssf.inference import mask_window_edges

    g = torch.Generator().manual_seed(0)
    s = torch.randn(3, 15, 15, generator=g)
    m = mask_window_edges(s, 2)
    assert m.shape == s.shape
    border = torch.ones(15, 15, dtype=torch.bool)
    border[2:-2, 2:-2] = False
    assert torch.isneginf(m[:, border]).all()
    torch.testing.assert_close(torch.logsumexp(m, dim=(1, 2)), torch.zeros(3))
    # Interior keeps its relative values
    d = m[:, 2:-2, 2:-2] - s[:, 2:-2, 2:-2]
    torch.testing.assert_close(d - d[:, :1, :1], torch.zeros_like(d))
    torch.testing.assert_close(mask_window_edges(s, 0, renormalise=False), s)
    with pytest.raises(ValueError, match="leaves nothing"):
        mask_window_edges(s, 8)


def test_next_step_surfaces_excludes_edge_buffer(toy):
    from deepssf.inference import next_step_surfaces
    from deepssf.predict import habitat_edge_buffer

    spatial, scalars, bearing, _ = _toy_steps(toy, n=5)
    surf = next_step_surfaces(toy.model, spatial, scalars, bearing)
    b = habitat_edge_buffer(toy.model)
    assert b == toy.shape.n_conv_layers_hab
    with torch.no_grad():
        out = toy.model((spatial, scalars, bearing))
    torch.testing.assert_close(surf["movement"], out[..., 1])
    inner = (slice(None), slice(b, -b), slice(b, -b))
    for name, raw in (("habitat", out[..., 0]), ("joint", out[..., 0] + out[..., 1])):
        s = surf[name]
        assert torch.isneginf(s[:, :b]).all() and torch.isneginf(s[:, :, -b:]).all()
        expected = raw[inner] - torch.logsumexp(raw[inner], dim=(1, 2), keepdim=True)
        torch.testing.assert_close(s[inner], expected, atol=1e-5, rtol=1e-5)
    full = next_step_surfaces(toy.model, spatial, scalars, bearing, edge_buffer=0)
    assert torch.isfinite(full["joint"]).all()


def test_kernel_summary_ring_follows_masked_interior():
    from deepssf.inference import kernel_summary, mask_window_edges

    window, w = 21, 3
    dist, brg = _grids(window)
    k = mask_window_edges(torch.zeros(1, window, window), w)
    ks = kernel_summary(k, dist, brg, edge=w).iloc[0]
    side = window - 2 * w
    assert ks["outer_ring_mass"] == pytest.approx((4 * side - 4) / side**2)


@pytest.mark.skipif(not _accelerators(), reason="no CUDA or MPS device")
@pytest.mark.parametrize("device", _accelerators() or ["none"])
def test_next_step_surfaces_on_accelerator(toy, device):
    import copy

    from deepssf.inference import next_step_surfaces

    model = copy.deepcopy(toy.model).to(device)
    data = _toy_steps(toy, n=4)
    cpu = next_step_surfaces(toy.model, *data[:3])
    dev = next_step_surfaces(model, *(t.to(device) for t in data[:3]))
    for name in ("habitat", "movement", "joint"):
        a, b = dev[name].cpu(), cpu[name]
        assert torch.equal(torch.isfinite(a), torch.isfinite(b))
        fin = torch.isfinite(b)
        torch.testing.assert_close(a[fin], b[fin], atol=1e-4, rtol=1e-4)
