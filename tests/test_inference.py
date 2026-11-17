"""Tests for deepssf.inference.

Every inference function must work for any deepSSF model, so the tests run
against several toy ``ConvJointModel`` shapes (window size, habitat depth,
channel width, number of raster bands and of scalars all vary) and compare
against brute-force ``model.forward`` / ``predict_habitat_landscape`` results.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        window=21, n_conv_layers_hab=1, n_conv_layers_move=2, channels=4,
        n_raster=1, n_scalars=3,
    ),
    ToyShape(
        window=33, n_conv_layers_hab=3, n_conv_layers_move=3, channels=16,
        n_raster=5, n_scalars=7,
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
