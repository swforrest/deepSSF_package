"""Model inference: how a fitted deepSSF responds to its inputs.

Tools for interrogating a fitted :class:`~deepssf.model.ConvJointModel` —
partial dependence of habitat selection on the scalar covariates, feature
importance, and the movement kernel's parameters — written for *any* deepSSF
model, not one particular input set.  Nothing here assumes a fixed number or
naming of raster bands or scalar covariates, a fixed window size, or a fixed
network depth: all of it is read from the model's module structure or passed
in by the caller.

How the model works (the facts this module relies on)
-----------------------------------------------------
These are properties of :mod:`deepssf.model` that the functions here exploit.
If the architecture changes, re-check them.

**Scalar inputs enter only the first conv layer of each head.**
``ConvJointModel.forward`` tiles the ``[B, S]`` scalar row into ``S``
constant-valued maps (``Scalar_to_Grid_Block``), concatenates them *after* the
raster channels, and passes the same ``[B, C_raster + S, D, D]`` tensor to
both ``conv_habitat`` and ``conv_movement``.  Only the first ``nn.Conv2d`` of
each head reads those channels; every later layer reads feature maps.  The
movement fully connected block (``fcn_movement_all``) receives only the
flattened conv output — there is no direct scalar input.  Hence:

* the number of raster channels is
  ``conv_habitat.conv2d[0].in_channels - n_scalars``, and the scalar channels
  are the slice after the rasters;
* **habitat shortcut** — for any pixel whose kernel footprint lies inside the
  zero-padded border, the scalars' contribution to the first habitat layer's
  pre-activation is a constant per output channel,
  ``shift_c(s) = sum_j s_j * sum_{kh,kw} W[c, n_raster + j, kh, kw]``.
  The whole habitat surface therefore depends on the scalars only through this
  length-``out_channels`` vector.  The raster part of layer 1 can be computed
  once per landscape and, for each scenario, the shift added before running
  the remaining layers.  This is exact away from the padded edge, which
  :func:`deepssf.predict.habitat_edge_buffer` already masks.

**Output.** ``forward((spatial [B, C, D, D], scalars [B, S], bearing [B, 1]))``
returns ``[B, D, D, 2]``: ``[..., 0]`` is the habitat log-probability and
``[..., 1]`` the movement log-probability, each normalised (log-sum-exp) over
the window.  The joint next-step surface is ``hab + move``, renormalised by the
caller (as the loss, the simulator and validation all do).  The spatial size
must equal the training window ``params.image_dim``, because
``Scalar_to_Grid_Block`` expands to that fixed size.

**Movement kernel.**  ``fcn_movement_all`` emits ``num_movement_params`` raw
values that ``movement_grid_output`` (``Params_to_Grid_Block_ChV``) turns into a
log-normalised ``[B, D, D]`` grid.  For the current 12-parameter layout:

====== =======================================
Index  Parameter
====== =======================================
0      gamma 1 shape
1      gamma 1 scale (CRS units, e.g. m)
2      gamma 1 weight logit
3      gamma 2 shape
4      gamma 2 scale (×500)
5      gamma 2 weight logit
6      von Mises 1 mean offset (rad)
7      von Mises 1 concentration κ
8      von Mises 1 weight logit
9      von Mises 2 mean offset (rad)
10     von Mises 2 concentration κ
11     von Mises 2 weight logit
====== =======================================

* Positive parameters use ``_positive(x) = exp(clamp(x, -raw_clamp,
  raw_clamp))``; mixture weights are ``log_softmax`` over the pairs (2, 5) and
  (8, 11).
* The von Mises means are offsets from ``bearing_tm1`` — the *only* use of the
  previous bearing; it is not a network input.
* The grid buffers ``distance_layer`` and ``bearing_layer`` are in CRS units
  and radians (0 = east, counter-clockwise).  The centre cell's distance is set
  to ``0.3826 * pixel_size`` (the mean radius within one pixel).
* The gamma log-density includes the polar → Cartesian ``1/r`` Jacobian.
* The kernel is truncated to the window (half-width along the axes, ×√2 at the
  corners).
* No model method returns the raw parameters, but the pieces can be called
  separately: ``scalar_grid_output``, ``conv_movement``, ``fcn_movement_all``,
  ``movement_grid_output(params, bearing)`` and its ``_positive``,
  ``_gamma_log`` and ``_vonmises_log`` methods.  Decoding here reuses those
  methods and buffers rather than copying them.

Generality
----------
Window size comes from the model (``movement_grid_output.distance_layer`` or
``params.image_dim``); pixel size from the raster transform or the grid
buffers; layer counts, channel counts, kernel size and padding from the module
structure.  Analysis choices (variables, grids, sample sizes, importance
groups, devices) are arguments.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from deepssf.predict import habitat_edge_buffer

__all__ = ["HabitatLandscapePredictor", "ScalarEncoder", "crop_windows"]


# ---------------------------------------------------------------------------
# Scalar encoding: natural units <-> model inputs
# ---------------------------------------------------------------------------

_SPEC_FORMAT = "deepssf.ScalarEncoder/1"
_TRANSFORMS = ("none", "log1p")
# A calm-rule value decoded from float32 model inputs is only equal to its
# target to ~1e-6 (e.g. a wind speed of 0 comes back as ~1e-7), so the rule
# matches within this tolerance by default.
_CALM_ATOL = 1e-5
_REQUIRED_KEYS = {
    "cyclic": ("input", "columns", "period"),
    "linear": ("input", "column", "mean", "sd"),
    "circular_degrees": ("input", "columns"),
    "raw": ("input", "column"),
}


def _entry_columns(entry: Mapping[str, Any]) -> list[str]:
    """Model columns an entry produces."""
    if entry["type"] in ("cyclic", "circular_degrees"):
        return list(entry["columns"])
    return [entry["column"]]


def _validate_entry(entry: Mapping[str, Any], i: int) -> dict[str, Any]:
    """Check one spec entry and return a normalised plain-dict copy."""
    entry = dict(entry)
    kind = entry.get("type")
    if kind not in _REQUIRED_KEYS:
        raise ValueError(
            f"Spec entry {i} has type {kind!r}; "
            f"expected one of {sorted(_REQUIRED_KEYS)}"
        )
    missing = [k for k in _REQUIRED_KEYS[kind] if k not in entry]
    if missing:
        raise ValueError(
            f"Spec entry {i} ({kind}, {entry.get('input')!r}) is missing {missing}"
        )

    entry["input"] = str(entry["input"])
    if kind in ("cyclic", "circular_degrees"):
        cols = list(entry["columns"])
        if len(cols) != 2:
            raise ValueError(
                f"Spec entry {i} ({kind}, {entry['input']!r}) needs 2 columns "
                f"[sin, cos], got {cols}"
            )
        entry["columns"] = [str(c) for c in cols]
    else:
        entry["column"] = str(entry["column"])

    if kind == "cyclic":
        entry["period"] = float(entry["period"])
        if not np.isfinite(entry["period"]) or entry["period"] <= 0:
            raise ValueError(f"Spec entry {i} ({entry['input']!r}): period must be > 0")
    elif kind == "linear":
        transform = entry.get("transform", "none")
        if transform is None or (isinstance(transform, float) and np.isnan(transform)):
            transform = "none"
        if transform not in _TRANSFORMS:
            raise ValueError(
                f"Spec entry {i} ({entry['input']!r}): transform {transform!r} not in "
                f"{_TRANSFORMS}"
            )
        entry["transform"] = transform
        entry["mean"] = float(entry["mean"])
        entry["sd"] = float(entry["sd"])
        if (
            not np.isfinite(entry["mean"])
            or not np.isfinite(entry["sd"])
            or entry["sd"] <= 0
        ):
            raise ValueError(
                f"Spec entry {i} ({entry['input']!r}): mean must be finite and sd > 0, "
                f"got mean={entry['mean']}, sd={entry['sd']}"
            )
    elif kind == "circular_degrees":
        calm = entry.get("calm")
        if calm is not None:
            if "input" not in calm or "equals" not in calm:
                raise ValueError(
                    f"Spec entry {i} ({entry['input']!r}): calm must be "
                    "{'input': <name>, 'equals': <value>[, 'atol': <tolerance>]}"
                )
            calm = {
                "input": str(calm["input"]),
                "equals": float(calm["equals"]),
                "atol": float(calm.get("atol", _CALM_ATOL)),
            }
        entry["calm"] = calm
    return entry


class ScalarEncoder:
    """Convert scalar covariates between natural units and model inputs.

    A deepSSF model takes its scalar covariates as an encoded row in a fixed
    column order (``scalar_cols``): cyclic variables as sin/cos pairs,
    continuous variables standardised, and so on.  The encoder is driven by a
    *spec* — a list of entries that each map one natural-unit input to one or
    two model columns — so it holds no variable names of its own.

    Parameters
    ----------
    spec:
        Sequence of dicts, one per natural input.  Supported ``type`` values:

        * ``"cyclic"`` — ``{"input", "period", "columns": [sin_col, cos_col]}``.
          Encodes ``sin(2π x / period)``, ``cos(2π x / period)``; e.g. hour of
          day with period 24, day of year with period 365.25.
        * ``"linear"`` — ``{"input", "column", "transform", "mean", "sd"}``.
          Encodes ``(f(x) - mean) / sd`` with ``f`` the identity
          (``"none"``) or ``log1p``.
        * ``"circular_degrees"`` — ``{"input", "columns": [sin_col, cos_col],
          "calm": {"input", "equals", "atol"} | None}``.  Encodes
          ``sin``/``cos`` of the angle in degrees.  When a calm rule is given,
          rows where ``calm.input`` is within ``atol`` (default 1e-5, enough to
          absorb float32 round trips) of ``calm.equals`` — e.g. wind speed 0 —
          are encoded as (0, 0) and the angle is ignored.
        * ``"raw"`` — ``{"input", "column"}``.  Passed through unchanged.
    scalar_cols:
        The model's scalar columns, in the order it was trained with.  The spec
        must produce each of them exactly once.
    ranges:
        Optional ``{input: (min, max)}`` training ranges, used by
        :meth:`check_ranges`.  See also :meth:`ranges_from`.

    Raises
    ------
    ValueError
        If an entry is malformed, or the spec's columns are not exactly
        ``scalar_cols`` (the message lists the missing, extra and duplicated
        columns).
    """

    def __init__(
        self,
        spec: Sequence[Mapping[str, Any]],
        scalar_cols: Sequence[str],
        ranges: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self.spec = [_validate_entry(e, i) for i, e in enumerate(spec)]
        self.scalar_cols = [str(c) for c in scalar_cols]

        produced = [c for e in self.spec for c in _entry_columns(e)]
        missing = [c for c in self.scalar_cols if c not in produced]
        extra = [c for c in produced if c not in self.scalar_cols]
        dupes = sorted({c for c in produced if produced.count(c) > 1})
        dupe_cols = sorted(
            {c for c in self.scalar_cols if self.scalar_cols.count(c) > 1}
        )
        if missing or extra or dupes or dupe_cols:
            parts = []
            if missing:
                parts.append(
                    f"missing (in scalar_cols, not produced by the spec): {missing}"
                )
            if extra:
                parts.append(
                    f"extra (produced by the spec, not in scalar_cols): {extra}"
                )
            if dupes:
                parts.append(f"produced by more than one entry: {dupes}")
            if dupe_cols:
                parts.append(f"repeated in scalar_cols: {dupe_cols}")
            raise ValueError("Spec does not match scalar_cols — " + "; ".join(parts))

        inputs = [e["input"] for e in self.spec]
        dupe_inputs = sorted({x for x in inputs if inputs.count(x) > 1})
        if dupe_inputs:
            raise ValueError(f"Inputs used by more than one spec entry: {dupe_inputs}")

        self.ranges: dict[str, tuple[float, float]] = {}
        for name, bounds in (ranges or {}).items():
            self.set_range(name, *bounds)

    # -- descriptive -------------------------------------------------------

    @property
    def inputs(self) -> list[str]:
        """Natural-unit inputs :meth:`encode` needs (including calm-rule inputs)."""
        names = [e["input"] for e in self.spec]
        for e in self.spec:
            calm = e.get("calm")
            if calm and calm["input"] not in names:
                names.append(calm["input"])
        return names

    def __repr__(self) -> str:
        kinds = ", ".join(f"{e['input']}:{e['type']}" for e in self.spec)
        return f"ScalarEncoder({len(self.scalar_cols)} columns; {kinds})"

    # -- ranges ------------------------------------------------------------

    def set_range(self, name: str, lo: float, hi: float) -> None:
        """Set the training range of one natural input."""
        if name not in self.inputs:
            raise KeyError(f"{name!r} is not an input of this encoder ({self.inputs})")
        lo, hi = float(lo), float(hi)
        if lo > hi:
            raise ValueError(f"Range for {name!r} has min {lo} > max {hi}")
        self.ranges[name] = (lo, hi)

    def ranges_from(
        self, data: pd.DataFrame, inputs: Sequence[str] | None = None
    ) -> ScalarEncoder:
        """Set training ranges from the observed min/max of *data* (NaN ignored).

        *inputs* defaults to every encoder input that is a column of *data*.
        Returns ``self`` so it can be chained.
        """
        names = (
            [n for n in self.inputs if n in data.columns] if inputs is None else inputs
        )
        for name in names:
            values = pd.to_numeric(data[name], errors="coerce")
            self.set_range(name, values.min(), values.max())
        return self

    def check_ranges(self, natural: pd.DataFrame | Mapping[str, Any]) -> list[str]:
        """Messages for natural-unit values outside the training ranges.

        Only inputs with a range (see :meth:`ranges_from`) are checked.  An
        empty list means every value is within range.
        """
        df = _as_frame(natural)
        messages = []
        for name, (lo, hi) in self.ranges.items():
            if name not in df.columns:
                continue
            values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
            out = (values < lo) | (values > hi)
            if out.any():
                bad = values[out]
                messages.append(
                    f"{name}: {out.sum()} of {len(values)} value(s) outside the "
                    f"training range [{lo:g}, {hi:g}] "
                    f"(min {np.nanmin(bad):g}, max {np.nanmax(bad):g})"
                )
        return messages

    # -- encode / decode ---------------------------------------------------

    def encode(self, natural: pd.DataFrame | Mapping[str, Any]) -> np.ndarray:
        """Encode natural-unit rows as model inputs.

        Parameters
        ----------
        natural:
            DataFrame with one column per input (see :attr:`inputs`), or a dict
            of scalars/arrays (scalars are broadcast).

        Returns
        -------
        np.ndarray, float32, shape [N, len(scalar_cols)]
            In ``scalar_cols`` order.  Values are computed in float64 with the
            same expressions as :func:`deepssf.prepare_movement_df`, then cast.

        Raises
        ------
        KeyError
            If an input is missing.
        ValueError
            If any encoded value is not finite (e.g. a NaN input that no calm
            rule covers).
        """
        df = _as_frame(natural)
        missing = [n for n in self.inputs if n not in df.columns]
        if missing:
            raise KeyError(f"Natural-unit data is missing input(s) {missing}")

        cols: dict[str, np.ndarray] = {}
        for e in self.spec:
            x = df[e["input"]].to_numpy(dtype=np.float64)
            kind = e["type"]
            if kind == "cyclic":
                angle = 2 * np.pi * x / e["period"]
                cols[e["columns"][0]] = np.sin(angle)
                cols[e["columns"][1]] = np.cos(angle)
            elif kind == "linear":
                if e["transform"] == "log1p":
                    x = np.log1p(x)
                cols[e["column"]] = (x - e["mean"]) / e["sd"]
            elif kind == "circular_degrees":
                rad = np.radians(x)
                s, c = np.sin(rad), np.cos(rad)
                if e["calm"] is not None:
                    rule = e["calm"]
                    calm_x = df[rule["input"]].to_numpy(dtype=np.float64)
                    calm = np.abs(calm_x - rule["equals"]) <= rule["atol"]
                    s, c = np.where(calm, 0.0, s), np.where(calm, 0.0, c)
                cols[e["columns"][0]], cols[e["columns"][1]] = s, c
            else:  # raw
                cols[e["column"]] = x

        out = np.column_stack([cols[c] for c in self.scalar_cols])
        bad = ~np.isfinite(out)
        if bad.any():
            names = [self.scalar_cols[j] for j in np.where(bad.any(axis=0))[0]]
            raise ValueError(
                f"Non-finite encoded values in {names} ({bad.any(axis=1).sum()} row(s))"
            )
        return out.astype(np.float32)

    def decode(self, encoded: np.ndarray | torch.Tensor) -> pd.DataFrame:
        """Recover natural units from encoded rows ([N, S] or [S]).

        Cyclic inputs come back in ``[0, period)`` (via ``atan2``); angles in
        ``[0, 360)`` with NaN for calm (0, 0) rows; linear inputs unscaled
        with the inverse transform; raw inputs unchanged.  One column per spec
        entry, in spec order.
        """
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().numpy()
        arr = np.atleast_2d(np.asarray(encoded, dtype=np.float64))
        if arr.shape[1] != len(self.scalar_cols):
            raise ValueError(
                f"Encoded rows have {arr.shape[1]} columns; "
                f"expected {len(self.scalar_cols)}"
            )
        idx = {c: j for j, c in enumerate(self.scalar_cols)}

        out: dict[str, np.ndarray] = {}
        for e in self.spec:
            kind = e["type"]
            if kind in ("cyclic", "circular_degrees"):
                s, c = arr[:, idx[e["columns"][0]]], arr[:, idx[e["columns"][1]]]
                theta = np.arctan2(s, c)
                if kind == "cyclic":
                    out[e["input"]] = (e["period"] * theta) / (2 * np.pi) % e["period"]
                else:
                    # Round before wrapping so north decodes as 0, not 359.9999…
                    deg = np.round(np.degrees(theta), 6) % 360
                    out[e["input"]] = np.where(np.hypot(s, c) < 1e-6, np.nan, deg)
            elif kind == "linear":
                x = arr[:, idx[e["column"]]] * e["sd"] + e["mean"]
                out[e["input"]] = np.expm1(x) if e["transform"] == "log1p" else x
            else:
                out[e["input"]] = arr[:, idx[e["column"]]]
        return pd.DataFrame(out)

    # -- construction / persistence ------------------------------------------

    @classmethod
    def from_scaling_table(
        cls,
        table: pd.DataFrame | str | PathLike,
        scalar_cols: Sequence[str],
        extra_spec: Sequence[Mapping[str, Any]] = (),
        suffix: str = "_scaled",
        ranges: Mapping[str, Sequence[float]] | None = None,
    ) -> ScalarEncoder:
        """Build an encoder whose linear entries come from a scaling table.

        Parameters
        ----------
        table:
            DataFrame (or CSV path) with columns ``column``, ``transform``,
            ``mean`` and ``sd`` (``column`` may be the index).  Each row
            becomes a linear entry mapping input ``column`` to model column
            ``column + suffix``.
        scalar_cols:
            As for the constructor.
        extra_spec:
            The non-linear entries (cyclic, circular, raw, …).
        suffix:
            Appended to each table ``column`` to give the model column.
        ranges:
            As for the constructor.
        """
        if not isinstance(table, pd.DataFrame):
            table = pd.read_csv(table)
        if "column" not in table.columns:
            table = table.reset_index()
        required = {"column", "transform", "mean", "sd"}
        if not required <= set(table.columns):
            raise ValueError(
                f"Scaling table needs columns {sorted(required)}; "
                f"got {list(table.columns)}"
            )
        linear = [
            {
                "type": "linear",
                "input": row["column"],
                "column": f"{row['column']}{suffix}",
                "transform": row["transform"],
                "mean": row["mean"],
                "sd": row["sd"],
            }
            for _, row in table.iterrows()
        ]
        return cls([*extra_spec, *linear], scalar_cols, ranges=ranges)

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form, as written by :meth:`to_json`."""
        return {
            "format": _SPEC_FORMAT,
            "scalar_cols": list(self.scalar_cols),
            "spec": [dict(e) for e in self.spec],
            "ranges": {k: list(v) for k, v in self.ranges.items()},
        }

    def to_json(self, path: str | PathLike | None = None) -> str:
        """Serialise to JSON; also written to *path* when given."""
        text = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            Path(path).write_text(text + "\n")
        return text

    @classmethod
    def from_json(cls, source: str | PathLike) -> ScalarEncoder:
        """Load from a JSON string or a path to a JSON file."""
        if isinstance(source, PathLike) or not str(source).lstrip().startswith("{"):
            source = Path(source).read_text()
        data = json.loads(source)
        if data.get("format") != _SPEC_FORMAT:
            warnings.warn(
                f"Unrecognised encoder format {data.get('format')!r}; "
                f"expected {_SPEC_FORMAT!r}",
                stacklevel=2,
            )
        return cls(data["spec"], data["scalar_cols"], ranges=data.get("ranges"))


def _as_frame(natural: pd.DataFrame | Mapping[str, Any]) -> pd.DataFrame:
    """DataFrame from a DataFrame or a dict of scalars/arrays (broadcast)."""
    if isinstance(natural, pd.DataFrame):
        return natural
    if isinstance(natural, pd.Series):
        return natural.to_frame().T
    keys = list(natural)
    arrays = np.broadcast_arrays(*[np.atleast_1d(np.asarray(natural[k])) for k in keys])
    return pd.DataFrame({k: a.ravel() for k, a in zip(keys, arrays, strict=True)})


# ---------------------------------------------------------------------------
# Raster windows
# ---------------------------------------------------------------------------


def _stack_rasters(
    rasters: Sequence[torch.Tensor] | torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """[C, H, W] float32 tensor from a stacked array/tensor or a list of 2-D layers."""
    if isinstance(rasters, (torch.Tensor, np.ndarray)):
        stack = torch.as_tensor(rasters)
        if stack.ndim == 2:
            stack = stack.unsqueeze(0)
        if stack.ndim != 3:
            raise ValueError(
                f"Expected a [C, H, W] raster stack, got shape {tuple(stack.shape)}"
            )
    else:
        layers = [torch.as_tensor(r) for r in rasters]
        if any(r.ndim != 2 for r in layers):
            raise ValueError("A list of rasters must hold 2-D [H, W] layers")
        stack = torch.stack(layers, dim=0)
    return stack.to(torch.float32)


def _as_affine(transform: Any):
    from affine import Affine

    if isinstance(transform, Affine):
        return transform
    coeffs = transform.tolist() if hasattr(transform, "tolist") else list(transform)
    return Affine(*coeffs[:6])


def crop_windows(
    rasters: Sequence[torch.Tensor] | torch.Tensor | np.ndarray,
    xs: Any,
    ys: Any,
    transform: Any,
    window_size: int,
    fill: float = -1.0,
) -> torch.Tensor:
    """Square windows of a raster stack centred on many locations at once.

    Reproduces the crop :class:`deepssf.MovementDataset` gives the model in
    training: the centre pixel is ``floor(~transform * (x, y))``, the window
    spans ``window_size // 2`` pixels either side of it, and cells beyond the
    raster are *fill* (-1, as in training).

    Parameters
    ----------
    rasters:
        A [C, H, W] tensor/array, or a sequence of 2-D [H, W] layers (as
        returned by ``get_landscape``), in the model's channel order.
    xs, ys:
        Window centres in the raster's CRS (array-likes of length N).
    transform:
        The raster's affine transform (an ``Affine`` or its 6 coefficients).
    window_size:
        Window edge length in pixels (the model's ``image_dim``).
    fill:
        Value for cells outside the raster.

    Returns
    -------
    torch.Tensor, float32, shape [N, C, window_size, window_size]
        On the device of *rasters*.  Memory is ``N * C * window_size**2 * 4``
        bytes, so crop large samples in chunks.
    """
    stack = _stack_rasters(rasters)
    _, height, width = stack.shape
    inv = ~_as_affine(transform)
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    if xs.shape != ys.shape:
        raise ValueError(f"xs and ys differ in length ({xs.size} vs {ys.size})")
    px, py = inv * (xs, ys)  # same arithmetic as MovementDataset's scalar path
    px = torch.from_numpy(np.floor(px).astype(np.int64)).to(stack.device)
    py = torch.from_numpy(np.floor(py).astype(np.int64)).to(stack.device)

    offsets = torch.arange(window_size, device=stack.device) - window_size // 2
    rows = py[:, None] + offsets  # [N, ws]
    cols = px[:, None] + offsets
    row_ok = (rows >= 0) & (rows < height)
    col_ok = (cols >= 0) & (cols < width)

    windows = stack[
        :, rows.clamp(0, height - 1)[:, :, None], cols.clamp(0, width - 1)[:, None, :]
    ]
    windows = windows.permute(1, 0, 2, 3)  # [N, C, ws, ws]
    inside = (row_ok[:, :, None] & col_ok[:, None, :])[:, None]
    return torch.where(
        inside, windows, torch.tensor(fill, dtype=stack.dtype, device=stack.device)
    )


# ---------------------------------------------------------------------------
# Fast landscape habitat surfaces for many scalar scenarios
# ---------------------------------------------------------------------------


class HabitatLandscapePredictor:
    """Landscape habitat surfaces for many scalar scenarios, sharing the raster work.

    The scalar covariates enter the habitat CNN only through its first conv
    layer, as constant channels, so for every pixel away from the padded edge
    their contribution to that layer is a per-output-channel constant
    ``shift(s) = W_scalar.sum(kernel) @ s`` (see the module docstring).  This
    class convolves the rasters with the first layer once, then for each
    scenario adds the shift and runs the remaining layers.  The result matches
    :func:`deepssf.predict_habitat_landscape` on every valid cell.

    Parameters
    ----------
    model:
        Fitted ConvJointModel (only ``conv_habitat`` is used; it is put in
        eval mode, as ``predict_habitat_landscape`` does).
    landscape_rasters:
        [C, H, W] tensor/array or a sequence of 2-D layers — the same layers,
        order and scaling the model was trained on.
    n_scalars:
        Number of scalar covariates (``len(scalar_cols)``).  The raster channel
        count is ``conv_habitat.conv2d[0].in_channels - n_scalars`` and must
        match *landscape_rasters*.
    mask:
        Optional boolean [H, W] array; only True cells are valid (e.g. an
        availability mask).  The edge buffer is always excluded.
    crop_to_mask:
        Work on the bounding box of the valid cells plus the edge buffer
        instead of the whole landscape (much faster for a small mask).  Output
        arrays then cover only that box; see :attr:`row_slice`,
        :attr:`col_slice`, :meth:`to_full` and :meth:`crop_transform`.
    device:
        Where to compute.  Defaults to the model's device.
    max_batch_bytes:
        Rough cap on activation memory per batch of scenarios, used to pick
        the batch size in :meth:`predict`.

    Attributes
    ----------
    valid : np.ndarray of bool, [h, w]
        Cells that receive a value (mask and not within the edge buffer).
    edge_buffer : int
        :func:`deepssf.habitat_edge_buffer` of the model.
    n_raster, n_scalars : int
    """

    def __init__(
        self,
        model: nn.Module,
        landscape_rasters: Sequence[torch.Tensor] | torch.Tensor | np.ndarray,
        n_scalars: int,
        mask: np.ndarray | torch.Tensor | None = None,
        crop_to_mask: bool = True,
        device: str | torch.device | None = None,
        max_batch_bytes: int = 2**30,
    ) -> None:
        model.eval()
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )
        self.max_batch_bytes = int(max_batch_bytes)

        seq = model.conv_habitat.conv2d
        first = seq[0]
        if not isinstance(first, nn.Conv2d):
            raise TypeError(
                f"conv_habitat.conv2d[0] must be nn.Conv2d, got {type(first).__name__}"
            )
        convs = [m for m in seq if isinstance(m, nn.Conv2d)]
        for j, conv in enumerate(convs):
            k, p = conv.kernel_size, conv.padding
            if (
                conv.stride != (1, 1)
                or conv.dilation != (1, 1)
                or conv.groups != 1
                or isinstance(p, str)
                or any(
                    kk % 2 == 0 or pp != kk // 2 for kk, pp in zip(k, p, strict=True)
                )
            ):
                raise ValueError(
                    f"Habitat conv {j} (kernel {k}, stride {conv.stride}, padding {p}, "
                    f"dilation {conv.dilation}, groups {conv.groups}) does not "
                    "preserve the spatial size; the landscape shortcut needs odd "
                    "kernels, stride 1 and 'same' padding."
                )
        self._max_channels = max(c.out_channels for c in convs)

        stack = _stack_rasters(landscape_rasters)
        self.n_scalars = int(n_scalars)
        self.n_raster = first.in_channels - self.n_scalars
        if self.n_raster < 1 or stack.shape[0] != self.n_raster:
            raise ValueError(
                f"The habitat CNN takes {first.in_channels} input channels; with "
                f"{self.n_scalars} scalars that leaves {self.n_raster} raster "
                f"channels, but {stack.shape[0]} were supplied."
            )
        height, width = stack.shape[1:]
        self.full_shape = (int(height), int(width))
        self.edge_buffer = habitat_edge_buffer(model)
        b = self.edge_buffer
        if 2 * b >= min(height, width):
            raise ValueError(
                f"Edge buffer of {b} px leaves nothing of a {height}x{width} raster"
            )

        valid = torch.zeros(height, width, dtype=torch.bool)
        valid[b : height - b, b : width - b] = True
        if mask is not None:
            mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
            if tuple(mask.shape) != self.full_shape:
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} != raster shape {self.full_shape}"
                )
            valid &= mask
        if not valid.any():
            raise ValueError(
                "No valid cells: the mask lies entirely within the edge buffer"
            )

        if crop_to_mask and mask is not None:
            rows = torch.where(valid.any(dim=1))[0]
            cols = torch.where(valid.any(dim=0))[0]
            r0, r1 = max(int(rows[0]) - b, 0), min(int(rows[-1]) + 1 + b, height)
            c0, c1 = max(int(cols[0]) - b, 0), min(int(cols[-1]) + 1 + b, width)
        else:
            r0, r1, c0, c1 = 0, height, 0, width
        self.row_slice, self.col_slice = slice(r0, r1), slice(c0, c1)

        self.valid = valid[r0:r1, c0:c1].numpy()
        self._valid = torch.from_numpy(self.valid).to(self.device)
        self.shape = self.valid.shape

        weight = first.weight.detach().to(self.device)
        bias = first.bias.detach().to(self.device) if first.bias is not None else None
        # .contiguous() matters: on MPS (torch 2.12) a conv with a sliced,
        # non-contiguous weight silently returns wrong values.
        raster_weight = weight[:, : self.n_raster].contiguous()
        crop = stack[:, r0:r1, c0:c1].to(self.device).contiguous()
        with torch.no_grad():
            # Raster part of layer 1, computed once: [1, out_channels, h, w]
            self._base = F.conv2d(
                crop.unsqueeze(0), raster_weight, bias, padding=first.padding
            )
        # Per-scenario shift = scalars @ (scalar weights summed over the kernel)ᵀ,
        # with the sum taken over the kernel: [out_channels, S]
        self._scalar_weight = (
            weight[:, self.n_raster :].contiguous().sum(dim=(2, 3)).contiguous()
        )
        self._rest = seq[1:]

    # -- geometry helpers --------------------------------------------------

    def to_full(self, surfaces: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Embed [..., h, w] cropped surfaces in [..., H, W] NaN arrays."""
        lead = tuple(surfaces.shape[:-2])
        if isinstance(surfaces, torch.Tensor):
            full = torch.full(
                lead + self.full_shape,
                float("nan"),
                dtype=surfaces.dtype,
                device=surfaces.device,
            )
        else:
            full = np.full(lead + self.full_shape, np.nan, dtype=surfaces.dtype)
        full[..., self.row_slice, self.col_slice] = surfaces
        return full

    def crop_transform(self, transform: Any):
        """Affine transform of the cropped grid, given the full raster's."""
        from affine import Affine

        return _as_affine(transform) * Affine.translation(
            self.col_slice.start, self.row_slice.start
        )

    def batch_size(self) -> int:
        """Scenarios per batch that keep activations under ``max_batch_bytes``."""
        # ~3 live activations of the widest layer per scenario (input, conv
        # output, ReLU output), float32
        per = 3 * self._max_channels * self.shape[0] * self.shape[1] * 4
        return max(1, self.max_batch_bytes // per)

    # -- prediction ----------------------------------------------------------

    def predict(
        self,
        encoded_scalars: np.ndarray | torch.Tensor | Sequence[float],
        normalise: bool = True,
        as_numpy: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray | torch.Tensor:
        """Habitat surfaces for one or more encoded scalar rows.

        Parameters
        ----------
        encoded_scalars:
            [S] or [B, S] model inputs in ``scalar_cols`` order (e.g. from
            :meth:`ScalarEncoder.encode`).
        normalise:
            Subtract each surface's log-sum-exp over the valid cells, so it is a
            log-probability summing to 1 over them (over the whole interior
            when there is no mask, exactly as ``predict_habitat_landscape``).
            False gives the raw conv output.
        as_numpy:
            Return a NumPy array on the CPU (default) or a tensor on
            :attr:`device`.
        batch_size:
            Scenarios per forward pass; defaults to :meth:`batch_size`.

        Returns
        -------
        float32 array/tensor, shape [B, h, w]
            NaN outside :attr:`valid`.
        """
        if not isinstance(encoded_scalars, torch.Tensor):
            encoded_scalars = np.asarray(encoded_scalars, dtype=np.float32)
        s = torch.as_tensor(encoded_scalars).to(device=self.device, dtype=torch.float32)
        if s.ndim == 1:
            s = s.unsqueeze(0)
        if s.ndim != 2 or s.shape[1] != self.n_scalars:
            raise ValueError(
                f"Expected [B, {self.n_scalars}] scalar rows, got {tuple(s.shape)}"
            )

        step = int(batch_size) if batch_size is not None else self.batch_size()
        invalid = ~self._valid
        outs = []
        with torch.no_grad():
            for i in range(0, s.shape[0], step):
                shift = s[i : i + step] @ self._scalar_weight.T  # [b, out_channels]
                x = self._base + shift[:, :, None, None]
                logits = self._rest(x).squeeze(1)  # [b, h, w]
                if normalise:
                    lse = torch.logsumexp(
                        logits.masked_fill(invalid, float("-inf")), dim=(1, 2)
                    )
                    logits = logits - lse[:, None, None]
                logits = logits.masked_fill(invalid, float("nan"))
                outs.append(logits.cpu() if as_numpy else logits)
        out = torch.cat(outs, dim=0)
        return out.numpy() if as_numpy else out
