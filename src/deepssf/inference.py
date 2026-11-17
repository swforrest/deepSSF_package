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

__all__ = [
    "HabitatLandscapePredictor",
    "ScalarEncoder",
    "accumulated_local_effects",
    "availability_mask",
    "binned_average",
    "class_selection",
    "crop_windows",
    "h_statistic",
    "kernel_summary",
    "mask_window_edges",
    "mixture_moments",
    "movement_parameters",
    "next_step_surfaces",
    "partial_dependence",
    "permutation_importance",
    "step_nll",
]


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

    def in_range(self, natural: pd.DataFrame | Mapping[str, Any]) -> np.ndarray:
        """Boolean per row: every ranged input is within its training range.

        NaN values (e.g. the angle of a calm row) count as in range.
        """
        df = _as_frame(natural)
        ok = np.ones(len(df), dtype=bool)
        for name, (lo, hi) in self.ranges.items():
            if name in df.columns:
                values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
                ok &= ~((values < lo) | (values > hi))
        return ok

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


# ---------------------------------------------------------------------------
# Availability and landcover selection
# ---------------------------------------------------------------------------


def availability_mask(
    raster_shape: Sequence[int],
    transform: Any,
    xs: Any,
    ys: Any,
    radius_px: int,
    shape: str = "square",
) -> np.ndarray:
    """Cells within *radius_px* of any of the given locations.

    The locations are rasterised (pixel = ``floor(~transform * (x, y))``, as
    in :func:`crop_windows`) and dilated with a max-pool.  With
    ``shape="square"`` and ``radius_px = window_size // 2`` the result is
    exactly the union of the model's input windows centred on the locations —
    the cells the habitat network ever saw around them.

    Parameters
    ----------
    raster_shape:
        ``(H, W)`` (or ``(C, H, W)``; the last two entries are used).
    transform:
        The raster's affine transform (``Affine`` or 6 coefficients).
    xs, ys:
        Locations in the raster's CRS.  Locations outside the raster still
        mark the cells within *radius_px* of them that are inside it.
    radius_px:
        Dilation radius in pixels; the caller chooses it, typically the window
        half-width ``window_size // 2``.
    shape:
        ``"square"`` (Chebyshev distance ≤ radius, the window footprint) or
        ``"disc"`` (Euclidean distance ≤ radius, in pixels).

    Returns
    -------
    np.ndarray of bool, shape [H, W]
    """
    height, width = (int(v) for v in tuple(raster_shape)[-2:])
    r = int(radius_px)
    if r < 0:
        raise ValueError(f"radius_px must be >= 0, got {radius_px}")
    if shape not in ("square", "disc"):
        raise ValueError(f"shape must be 'square' or 'disc', got {shape!r}")

    inv = ~_as_affine(transform)
    xs = np.asarray(xs, dtype=np.float64).ravel()
    ys = np.asarray(ys, dtype=np.float64).ravel()
    px, py = inv * (xs, ys)
    # Rasterise onto a grid padded by r, so points just off the raster still
    # reach into it
    cols = np.floor(px).astype(np.int64) + r
    rows = np.floor(py).astype(np.int64) + r
    keep = (rows >= 0) & (rows < height + 2 * r) & (cols >= 0) & (cols < width + 2 * r)
    grid = torch.zeros(1, 1, height + 2 * r, width + 2 * r)
    grid[0, 0, torch.from_numpy(rows[keep]), torch.from_numpy(cols[keep])] = 1.0
    if r == 0:
        return grid[0, 0].bool().numpy()

    def hpool(half: int) -> torch.Tensor:
        if half == 0:
            return grid
        return F.max_pool2d(grid, (1, 2 * half + 1), stride=1, padding=(0, half))

    if shape == "square":
        # A square max-pool is separable: rows, then columns
        dilated = F.max_pool2d(hpool(r), (2 * r + 1, 1), stride=1, padding=(r, 0))
        out = dilated[0, 0, r : r + height, r : r + width]
    else:
        # Disc: union over row offsets dy of a horizontal dilation of
        # half-width floor(sqrt(r² - dy²))
        out = torch.zeros(height, width)
        for dy in range(-r, r + 1):
            half = int(np.floor(np.sqrt(r * r - dy * dy) + 1e-9))
            pooled = hpool(half)[0, 0]
            out = torch.maximum(out, pooled[r + dy : r + dy + height, r : r + width])
    return out.bool().numpy()


def class_selection(
    log_surfaces: np.ndarray | torch.Tensor,
    class_rasters: np.ndarray | torch.Tensor,
    class_names: Sequence[str] | None = None,
    mask: np.ndarray | torch.Tensor | None = None,
    scenario_ids: Sequence[Any] | None = None,
) -> pd.DataFrame:
    """Use, availability and selection ratio of each class under each surface.

    For each surface (scenario) and class ``k``, over the valid cells (not NaN
    in the surface, and inside *mask* if given):

    * ``use`` — the share of the surface's probability in class ``k``
      (the surface is renormalised over the valid cells first);
    * ``availability`` — class ``k``'s share of the valid cells;
    * ``selection_ratio = use / availability`` and ``log_sr`` its log, where
      0 means the class is used in proportion to its availability.

    Classes with no valid cells get NaN.  The class rasters are typically a
    one-hot set; a cell in no class counts towards neither use nor
    availability of any class, but still towards the totals.

    Parameters
    ----------
    log_surfaces:
        [B, H, W] or [H, W] log-probability (or unnormalised log-intensity)
        surfaces, NaN where invalid — e.g. :meth:`HabitatLandscapePredictor.predict`
        output.  Tensors are processed on their own device.
    class_rasters:
        [K, H, W] class membership (0/1, or fractional cover), aligned with
        the surfaces (crop them to the predictor's ``row_slice``/``col_slice``
        if it was cropped).
    class_names:
        K names; defaults to ``class_0``, ``class_1``, ….
    mask:
        Optional boolean [H, W] (or [B, H, W]) restriction of the valid cells.
    scenario_ids:
        B labels for the ``scenario`` column; defaults to 0..B-1.

    Returns
    -------
    pd.DataFrame
        Long format: one row per scenario and class, with columns
        ``scenario, class, use, availability, selection_ratio, log_sr``.
    """
    surf = torch.as_tensor(log_surfaces)
    if surf.ndim == 2:
        surf = surf.unsqueeze(0)
    if surf.ndim != 3:
        raise ValueError(
            f"log_surfaces must be [B, H, W] or [H, W], got {tuple(surf.shape)}"
        )
    device = surf.device
    # float64 where the device has it (CPU); float32 on MPS
    dtype = torch.float32 if device.type == "mps" else torch.float64
    surf = surf.to(dtype)
    n_scen, height, width = surf.shape

    classes = torch.as_tensor(class_rasters).to(device=device, dtype=dtype)
    if classes.ndim == 2:
        classes = classes.unsqueeze(0)
    if tuple(classes.shape[1:]) != (height, width):
        raise ValueError(
            f"class_rasters {tuple(classes.shape)} do not match the surfaces' "
            f"{(height, width)}"
        )
    n_classes = classes.shape[0]
    names = (
        [f"class_{k}" for k in range(n_classes)]
        if class_names is None
        else [str(c) for c in class_names]
    )
    if len(names) != n_classes:
        raise ValueError(f"{len(names)} class names for {n_classes} class rasters")
    ids = list(range(n_scen)) if scenario_ids is None else list(scenario_ids)
    if len(ids) != n_scen:
        raise ValueError(f"{len(ids)} scenario ids for {n_scen} surfaces")

    valid = ~torch.isnan(surf)
    if mask is not None:
        m = torch.as_tensor(
            np.asarray(mask) if not isinstance(mask, torch.Tensor) else mask
        )
        valid &= m.to(device=device, dtype=torch.bool).expand_as(valid)

    log_p = surf.masked_fill(~valid, float("-inf"))
    peak = log_p.amax(dim=(1, 2), keepdim=True)
    p = torch.exp(log_p - peak).reshape(n_scen, -1)  # 0 at invalid cells
    class_cols = classes.reshape(n_classes, -1).T  # [cells, K]
    use = (p @ class_cols) / p.sum(dim=1, keepdim=True)
    valid_f = valid.reshape(n_scen, -1).to(dtype)
    available = valid_f @ class_cols
    availability = available / valid_f.sum(dim=1, keepdim=True)

    use = use.cpu().double().numpy()
    availability = availability.cpu().double().numpy()
    none_available = available.cpu().numpy() <= 0
    use[none_available] = np.nan
    availability[none_available] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = use / availability
        log_sr = np.log(ratio)

    return pd.DataFrame(
        {
            "scenario": np.repeat(np.asarray(ids, dtype=object), n_classes),
            "class": np.tile(names, n_scen),
            "use": use.ravel(),
            "availability": availability.ravel(),
            "selection_ratio": ratio.ravel(),
            "log_sr": log_sr.ravel(),
        }
    ).infer_objects()


# ---------------------------------------------------------------------------
# Partial dependence, ALE, binned averages and interaction strength
# ---------------------------------------------------------------------------

SetFn = Any  # Callable[[pd.DataFrame, Any], pd.DataFrame] | str


def _setter(set_fn: SetFn, what: str = "set_fn"):
    """A set_fn from a callable, or from a column name (plain assignment)."""
    if callable(set_fn):
        return set_fn
    if isinstance(set_fn, str):
        column = set_fn

        def assign(rows: pd.DataFrame, value: Any) -> pd.DataFrame:
            out = rows.copy()
            out[column] = np.broadcast_to(np.asarray(value), len(out))
            return out

        return assign
    raise TypeError(f"{what} must be a callable or a column name, got {type(set_fn)}")


def _as_metric_array(values: Any, n_rows: int, names: list[str] | None):
    """(array [n, M], names) from a metric_fn result."""
    if isinstance(values, pd.DataFrame):
        arr = values.to_numpy(dtype=np.float64)
        out_names = [str(c) for c in values.columns]
    else:
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        out_names = [f"metric_{j}" for j in range(arr.shape[1])]
    if arr.ndim != 2 or arr.shape[0] != n_rows:
        raise ValueError(
            f"metric_fn returned shape {arr.shape} for {n_rows} rows; expected "
            "[rows] or [rows, metrics]"
        )
    if names is not None and names != out_names:
        raise ValueError(f"metric_fn returned metrics {out_names}, earlier {names}")
    return arr, out_names


def _evaluate(
    metric_fn, natural_rows: pd.DataFrame, encoder: ScalarEncoder, batch_size: int
) -> tuple[np.ndarray, list[str]]:
    """Encode natural rows and evaluate metric_fn in batches: ([N, M], names)."""
    encoded = encoder.encode(natural_rows)
    step = max(int(batch_size), 1)
    chunks, names = [], None
    for i in range(0, len(encoded), step):
        part = encoded[i : i + step]
        arr, names = _as_metric_array(metric_fn(part), len(part), names)
        chunks.append(arr)
    return np.concatenate(chunks, axis=0), names


def partial_dependence(
    metric_fn,
    base_rows: pd.DataFrame,
    set_fn: SetFn,
    grid: Any,
    encoder: ScalarEncoder,
    batch_size: int = 256,
    grid_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partial dependence (PDP) and individual conditional expectation (ICE).

    For every grid value, every base row is modified by ``set_fn(rows,
    value)``, encoded, and passed to *metric_fn*.  The ICE curve of a base row
    is its metric across the grid; the PDP is their mean.

    Parameters
    ----------
    metric_fn:
        ``metric_fn(encoded [B, S]) -> DataFrame | array [B] | [B, M]`` — one
        or more metrics per encoded row (e.g. log SR per class from a
        :class:`HabitatLandscapePredictor` and :func:`class_selection`).
        DataFrame columns name the metrics.
    base_rows:
        Natural-unit rows (the ICE base sample), with every input the encoder
        needs.  Their index labels the ICE curves.
    set_fn:
        ``set_fn(rows, value) -> rows`` returning modified copies; this is
        where coupled variables are recomputed (e.g. RH and wet bulb when
        temperature changes at a fixed dew point).  A column name means plain
        assignment of that column.
    grid:
        1-D array-like of values, or a DataFrame whose rows are passed to
        *set_fn* as Series (for multi-variable grids, e.g. 2-D PDPs).
    encoder:
        The model's :class:`ScalarEncoder`.  If it has training ranges, the
        share of modified rows outside them is reported per grid value.
    batch_size:
        Encoded rows per *metric_fn* call.
    grid_name:
        Column name for a 1-D grid; defaults to *set_fn* if it is a string,
        else ``"value"``.

    Returns
    -------
    pd_df : pd.DataFrame
        One row per grid value and metric: the grid column(s), ``metric``,
        ``pd`` (mean), ``sd`` (across base rows), ``n`` and
        ``frac_out_of_range``.
    ice_df : pd.DataFrame
        One row per grid value, base row and metric: the grid column(s),
        ``ice_id``, ``metric``, ``ice`` (the metric value).
    """
    setter = _setter(set_fn)
    if isinstance(grid, pd.DataFrame):
        grid_df = grid.reset_index(drop=True)
        values = [row for _, row in grid_df.iterrows()]
    else:
        name = grid_name or (set_fn if isinstance(set_fn, str) else "value")
        grid_df = pd.DataFrame({name: np.asarray(grid).ravel()})
        values = list(grid_df[name])
    reserved = {"metric", "pd", "sd", "n", "frac_out_of_range", "ice_id", "ice"}
    clash = reserved & set(grid_df.columns)
    if clash:
        raise ValueError(
            f"Grid column name(s) {sorted(clash)} clash with output columns"
        )
    n_base, n_grid = len(base_rows), len(values)
    if n_base == 0 or n_grid == 0:
        raise ValueError("base_rows and grid must both be non-empty")

    modified = pd.concat(
        [setter(base_rows.copy(), v) for v in values], ignore_index=True
    )
    metrics, names = _evaluate(metric_fn, modified, encoder, batch_size)
    cube = metrics.reshape(n_grid, n_base, len(names))
    in_range = encoder.in_range(modified).reshape(n_grid, n_base)

    grid_rep = grid_df.loc[grid_df.index.repeat(len(names))].reset_index(drop=True)
    pd_df = grid_rep.assign(
        metric=np.tile(names, n_grid),
        pd=np.nanmean(cube, axis=1).ravel(),
        sd=np.nanstd(cube, axis=1, ddof=1).ravel() if n_base > 1 else np.nan,
        n=n_base,
        frac_out_of_range=np.repeat(1.0 - in_range.mean(axis=1), len(names)),
    )
    ice_grid = grid_df.loc[grid_df.index.repeat(n_base * len(names))]
    ice_df = ice_grid.reset_index(drop=True).assign(
        ice_id=np.tile(np.repeat(np.asarray(base_rows.index), len(names)), n_grid),
        metric=np.tile(names, n_grid * n_base),
        ice=cube.ravel(),
    )
    return pd_df, ice_df


def accumulated_local_effects(
    metric_fn,
    rows: pd.DataFrame,
    variable: str,
    set_fn: SetFn | None,
    encoder: ScalarEncoder,
    n_bins: int = 10,
    batch_size: int = 256,
    values: Any = None,
) -> pd.DataFrame:
    """First-order accumulated local effects (ALE) of one variable.

    The variable's observed values are cut into quantile bins.  Each row is
    moved to the lower and upper edge of its own bin (by *set_fn*, so coupled
    variables follow) and the metric difference is averaged within the bin.
    Only observed rows are used, each only near its own value, so correlated
    inputs are never pushed into combinations far from the data.  The
    differences are accumulated across bins and centred so the count-weighted
    mean effect is zero.

    Parameters
    ----------
    metric_fn, encoder, batch_size:
        As for :func:`partial_dependence`.
    rows:
        Observed natural-unit rows.
    variable:
        Name of the variable (a column of *rows*, unless *values* is given).
    set_fn:
        ``set_fn(rows, value)``, where *value* is a per-row array; None means
        plain assignment of *variable*.
    n_bins:
        Number of quantile bins (fewer if quantiles coincide).
    values:
        The variable's value per row when it is not a column (e.g. a derived
        "heat stress" axis).

    Returns
    -------
    pd.DataFrame
        One row per bin edge and metric: *variable* (the edge value),
        ``metric``, ``ale``, ``n`` (rows in the bin ending at this edge; 0 for
        the first edge).
    """
    setter = _setter(variable if set_fn is None else set_fn)
    x = np.asarray(rows[variable] if values is None else values, dtype=np.float64)
    if x.shape != (len(rows),):
        raise ValueError("values must have one entry per row")
    keep = np.isfinite(x)
    rows, x = rows.loc[keep], x[keep]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        raise ValueError(f"{variable} has a single value; no ALE to compute")
    n_edges = len(edges)
    # Bin j (1..n_edges-1) holds edges[j-1] < x <= edges[j]; the minimum goes in bin 1
    bins = np.clip(np.searchsorted(edges, x, side="left"), 1, n_edges - 1)

    lower = setter(rows.copy(), edges[bins - 1])
    upper = setter(rows.copy(), edges[bins])
    both = pd.concat([lower, upper], ignore_index=True)
    metrics, names = _evaluate(metric_fn, both, encoder, batch_size)
    diff = metrics[len(rows) :] - metrics[: len(rows)]

    counts = np.bincount(bins, minlength=n_edges)[1:]
    mean_diff = np.zeros((n_edges - 1, len(names)))
    for j in range(1, n_edges):
        in_bin = bins == j
        if in_bin.any():
            mean_diff[j - 1] = np.nanmean(diff[in_bin], axis=0)
    ale = np.vstack([np.zeros((1, len(names))), np.cumsum(mean_diff, axis=0)])
    # Centre: subtract the count-weighted mean of each bin's midpoint effect
    mid = (ale[:-1] + ale[1:]) / 2
    ale -= (counts[:, None] * mid).sum(axis=0) / counts.sum()

    return pd.DataFrame(
        {
            variable: np.repeat(edges, len(names)),
            "metric": np.tile(names, n_edges),
            "ale": ale.ravel(),
            "n": np.repeat(np.r_[0, counts], len(names)),
        }
    )


def _format_edge(v: float) -> str:
    return f"{float(f'{v:.4g}'):g}"


def binned_average(
    metric_fn,
    rows: pd.DataFrame,
    variable: str,
    encoder: ScalarEncoder,
    n_bins: int = 10,
    zero_bin_threshold: float = 0.1,
    min_n: int = 50,
    batch_size: int = 256,
    metrics: pd.DataFrame | np.ndarray | None = None,
) -> pd.DataFrame:
    """Metric averaged over observed rows in bins of one variable (an M-plot).

    The bins follow ``bin_weather()`` in the empirical weather analysis:

    * a variable with at most *n_bins* distinct values is treated as
      categorical (one bin per value);
    * if more than *zero_bin_threshold* of the values are exactly 0, the zeros
      get a bin of their own (``"0"``) and the non-zero values are cut into
      quantile bins;
    * otherwise quantile bins, ``(a, b]`` with the lowest one closed, edges
      rounded to 6 significant figures and de-duplicated;
    * bins with ``n <= min_n`` rows are dropped.

    Unlike a partial dependence plot, nothing is changed: each bin averages the
    model's output at the conditions actually observed, including whatever the
    other variables were doing at the time.  That is what makes it comparable
    with the empirical binned plots.

    Parameters
    ----------
    metric_fn, encoder, batch_size:
        As for :func:`partial_dependence`; not used when *metrics* is given.
    rows:
        Observed natural-unit rows.
    variable:
        The column to bin by.
    metrics:
        Precomputed ``metric_fn`` output for *rows* (reuse one evaluation for
        several variables).

    Returns
    -------
    pd.DataFrame
        One row per bin and metric: ``bin``, ``lo``, ``hi``, ``x_median``,
        ``n``, ``metric``, ``mean``, ``sd``, ``se``.
    """
    if metrics is None:
        values, names = _evaluate(metric_fn, rows, encoder, batch_size)
    else:
        values, names = _as_metric_array(metrics, len(rows), None)
    x = pd.to_numeric(rows[variable], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(x)
    x, values = x[keep], values[keep]

    labels = np.empty(len(x), dtype=object)
    lo = np.full(len(x), np.nan)
    hi = np.full(len(x), np.nan)
    uniques = np.unique(x)
    if len(uniques) <= n_bins:
        for u in uniques:
            sel = x == u
            labels[sel], lo[sel], hi[sel] = _format_edge(u), u, u
    else:
        zero_bin = np.mean(x == 0) > zero_bin_threshold
        to_bin = x[x > 0] if zero_bin else x
        qs = np.quantile(to_bin, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(np.array([float(f"{q:.6g}") for q in qs]))
        idx = np.clip(np.searchsorted(edges, x, side="left"), 1, len(edges) - 1)
        for j in range(1, len(edges)):
            sel = idx == j
            opener = "[" if j == 1 else "("
            labels[sel] = (
                f"{opener}{_format_edge(edges[j - 1])},{_format_edge(edges[j])}]"
            )
            lo[sel], hi[sel] = edges[j - 1], edges[j]
        if zero_bin:
            zero = x == 0
            labels[zero], lo[zero], hi[zero] = "0", 0.0, 0.0

    df = pd.DataFrame(values, columns=names)
    df["bin"], df["lo"], df["hi"], df["_x"] = labels, lo, hi, x
    rows_out = []
    for (label, b_lo, b_hi), sub in df.groupby(["bin", "lo", "hi"], sort=False):
        n = len(sub)
        if n <= min_n:
            continue
        for name in names:
            vals = sub[name].to_numpy()
            sd = np.nanstd(vals, ddof=1) if n > 1 else np.nan
            rows_out.append(
                {
                    "bin": label,
                    "lo": b_lo,
                    "hi": b_hi,
                    "x_median": float(np.median(sub["_x"])),
                    "n": n,
                    "metric": name,
                    "mean": float(np.nanmean(vals)),
                    "sd": sd,
                    "se": sd / np.sqrt(n),
                }
            )
    columns = ["bin", "lo", "hi", "x_median", "n", "metric", "mean", "sd", "se"]
    out = pd.DataFrame(rows_out, columns=columns)
    if len(out):
        out = out.sort_values(["lo", "hi", "metric"], kind="stable").reset_index(
            drop=True
        )
    return out


def h_statistic(
    metric_fn,
    rows: pd.DataFrame,
    encoder: ScalarEncoder,
    var_a: str,
    var_b: str,
    set_fn_a: SetFn | None = None,
    set_fn_b: SetFn | None = None,
    values_a: Any = None,
    values_b: Any = None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Friedman's H statistic for the interaction of two variables.

    With centred partial dependence functions evaluated at the sample's own
    values, ``H² = Σ (PD_ab − PD_a − PD_b)² / Σ PD_ab²``: the share of the
    joint effect's variance not explained by the two main effects (0 = no
    interaction).  Costs ``3 n²`` metric evaluations for ``n`` rows, so pass
    a small sample.

    Parameters
    ----------
    rows:
        Natural-unit sample; both the evaluation points and the averaging set.
    var_a, var_b:
        Variable names (columns of *rows* unless *values_a*/*values_b* given).
    set_fn_a, set_fn_b:
        As for :func:`partial_dependence` (default: plain assignment).
    values_a, values_b:
        Per-row values of derived variables.

    Returns
    -------
    pd.DataFrame
        One row per metric: ``metric``, ``h2``, ``h``.
    """
    set_a = _setter(var_a if set_fn_a is None else set_fn_a)
    set_b = _setter(var_b if set_fn_b is None else set_fn_b)
    xa = np.asarray(rows[var_a] if values_a is None else values_a)
    xb = np.asarray(rows[var_b] if values_b is None else values_b)
    n = len(rows)

    def mean_over_rows(frames: list[pd.DataFrame]) -> tuple[np.ndarray, list[str]]:
        metrics, names = _evaluate(
            metric_fn, pd.concat(frames, ignore_index=True), encoder, batch_size
        )
        return metrics.reshape(len(frames), n, -1).mean(axis=1), names

    pd_a, names = mean_over_rows([set_a(rows.copy(), v) for v in xa])
    pd_b, _ = mean_over_rows([set_b(rows.copy(), v) for v in xb])
    pd_ab, _ = mean_over_rows(
        [set_b(set_a(rows.copy(), va), vb) for va, vb in zip(xa, xb, strict=True)]
    )
    pd_a -= pd_a.mean(axis=0)
    pd_b -= pd_b.mean(axis=0)
    pd_ab -= pd_ab.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h2 = ((pd_ab - pd_a - pd_b) ** 2).sum(axis=0) / (pd_ab**2).sum(axis=0)
    return pd.DataFrame({"metric": names, "h2": h2, "h": np.sqrt(np.clip(h2, 0, None))})


# ---------------------------------------------------------------------------
# Per-step likelihood and permutation importance
# ---------------------------------------------------------------------------

HEADS = ("habitat", "movement", "joint")


def _target_cols_rows(targets: Any, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(cols, rows) long tensors from a [B, 2] (col, row) tensor or a (px, py) pair."""
    if isinstance(targets, (tuple, list)):
        px, py = targets
    else:
        t = torch.as_tensor(targets)
        px, py = t[:, 0], t[:, 1]
    return (
        torch.as_tensor(px).to(device=device, dtype=torch.long),
        torch.as_tensor(py).to(device=device, dtype=torch.long),
    )


def step_nll(
    model: nn.Module,
    spatial: torch.Tensor,
    scalars: torch.Tensor,
    bearing: torch.Tensor,
    targets: Any,
) -> dict[str, torch.Tensor]:
    """Negative log-likelihood of each observed step, per head.

    Parameters
    ----------
    model:
        Fitted ConvJointModel (evaluated as-is; call ``model.eval()`` first).
    spatial, scalars, bearing:
        A batch of model inputs ([B, C, D, D], [B, S], [B, 1]), on the
        model's device.
    targets:
        Next-step pixel within each window: ``(px, py)`` (as
        :class:`deepssf.MovementDataset` items collate) or a [B, 2] tensor of
        (column, row).

    Returns
    -------
    dict of [B] tensors
        ``"habitat"`` and ``"movement"``: minus each head's log-probability
        at the target; ``"joint"``: minus ``log_softmax(habitat + movement)``
        at the target — the per-step terms of
        :class:`deepssf.negativeLogLikeLoss`.
    """
    with torch.no_grad():
        out = model((spatial, scalars, bearing))
    hab, move = out[..., 0], out[..., 1]
    joint = hab + move
    joint = joint - torch.logsumexp(joint, dim=(1, 2), keepdim=True)
    cols, rows = _target_cols_rows(targets, out.device)
    b = torch.arange(out.shape[0], device=out.device)
    return {
        "habitat": -hab[b, rows, cols],
        "movement": -move[b, rows, cols],
        "joint": -joint[b, rows, cols],
    }


def _materialise(dataset: Any, batch_size: int) -> tuple[torch.Tensor, ...]:
    """(spatial, scalars, bearing, targets [N, 2]) on the CPU from a dataset."""
    if isinstance(dataset, (tuple, list)) and len(dataset) == 4:
        spatial, scalars, bearing, targets = dataset
        cols, rows = _target_cols_rows(targets, "cpu")
        return (
            torch.as_tensor(spatial).float(),
            torch.as_tensor(scalars).float(),
            torch.as_tensor(bearing).float().reshape(-1, 1),
            torch.stack([cols, rows], dim=1),
        )
    from torch.utils.data import DataLoader

    parts: list[list[torch.Tensor]] = [[], [], [], []]
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        spatial, scalars, bearing, target = batch[:4]
        cols, rows = _target_cols_rows(target, "cpu")
        parts[0].append(spatial.float())
        parts[1].append(scalars.float())
        parts[2].append(bearing.float().reshape(-1, 1))
        parts[3].append(torch.stack([cols, rows], dim=1))
    return tuple(torch.cat(p, dim=0) for p in parts)


def _apply_permutation(
    spatial: torch.Tensor,
    scalars: torch.Tensor,
    idx: torch.Tensor,
    source: torch.Tensor,
    scalar_idx: Sequence[int],
    raster_idx: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs of steps *idx*, with the group's channels taken from steps *source*.

    One permutation is applied to all of a group's channels together, so
    values that belong together (e.g. a sin/cos pair, or the bands of a
    one-hot raster) stay paired.  Each step keeps its own target.
    """
    sp = spatial[idx].clone()
    sc = scalars[idx].clone()
    if len(raster_idx):
        r = torch.as_tensor(list(raster_idx), dtype=torch.long, device=spatial.device)
        sp[:, r] = spatial[source][:, r]
    if len(scalar_idx):
        s = torch.as_tensor(list(scalar_idx), dtype=torch.long, device=scalars.device)
        sc[:, s] = scalars[source][:, s]
    return sp, sc


def permutation_importance(
    model: nn.Module,
    dataset: Any,
    groups: Mapping[str, Sequence[str]],
    scalar_cols: Sequence[str],
    raster_channel_names: Sequence[str],
    n_repeats: int = 5,
    batch_size: int = 256,
    seed: int = 0,
    device: str | torch.device | None = None,
    keep_on_device: bool = True,
) -> pd.DataFrame:
    """Grouped permutation importance: how much worse each head's fit gets.

    For each group, one random permutation of the steps is applied to all of
    the group's inputs together — scalar columns, or whole raster channels
    (windows are swapped between steps, each step keeping its own target) —
    and the mean per-step NLL (:func:`step_nll`) is recomputed.  The increase
    over the unpermuted NLL measures how much the fitted model *relies* on the
    group.  Correlated inputs share credit: permuting one of two correlated
    variables leaves the other to stand in for it.

    Parameters
    ----------
    model:
        Fitted ConvJointModel.
    dataset:
        A :class:`deepssf.MovementDataset` (or ``Subset`` of one), or any
        dataset of ``(spatial, scalars, bearing, target, ...)`` items, or a
        tuple of tensors ``(spatial, scalars, bearing, targets)``.  It is
        loaded into CPU memory once (``N * C * D² * 4`` bytes of windows).
    groups:
        ``{name: [input names]}``; names are scalar columns or raster channel
        names.  A group may mix both.
    scalar_cols:
        The model's scalar columns, in order.
    raster_channel_names:
        One name per raster channel, in order.
    n_repeats:
        Permutations per group; the increase is averaged and its SD reported.
    batch_size:
        Steps per forward pass.
    seed:
        Seeds the permutations (each group gets its own stream, so results do
        not depend on which other groups are requested).
    device:
        Defaults to the model's device.
    keep_on_device:
        Move the loaded steps to *device* once (fast; needs the memory there)
        rather than copying each batch from the CPU on every pass.

    Returns
    -------
    pd.DataFrame
        One row per group and head (``habitat``, ``movement``, ``joint``):
        ``group, head, baseline_nll, permuted_nll, increase, sd, n_repeats``.
    """
    scalar_cols = list(scalar_cols)
    raster_names = list(raster_channel_names)
    ambiguous = set(scalar_cols) & set(raster_names)
    if ambiguous:
        raise ValueError(
            f"Names are both scalar and raster inputs: {sorted(ambiguous)}"
        )
    resolved = {}
    for name, members in groups.items():
        unknown = [m for m in members if m not in scalar_cols and m not in raster_names]
        if unknown or not members:
            raise ValueError(
                f"Group {name!r}: unknown input(s) {unknown}"
                if unknown
                else f"Group {name!r} is empty"
            )
        resolved[name] = (
            [scalar_cols.index(m) for m in members if m in scalar_cols],
            [raster_names.index(m) for m in members if m in raster_names],
        )

    spatial, scalars, bearing, targets = _materialise(dataset, batch_size)
    if spatial.shape[1] != len(raster_names) or scalars.shape[1] != len(scalar_cols):
        raise ValueError(
            f"Data has {spatial.shape[1]} raster channels and {scalars.shape[1]} "
            f"scalars; names given for {len(raster_names)} and {len(scalar_cols)}"
        )
    n = spatial.shape[0]
    device = (
        torch.device(device) if device is not None else next(model.parameters()).device
    )
    model.eval()
    if keep_on_device:
        spatial, scalars, bearing, targets = (
            t.to(device) for t in (spatial, scalars, bearing, targets)
        )
    data_device = spatial.device

    def mean_nll(source: torch.Tensor | None, s_idx=(), r_idx=()) -> dict[str, float]:
        totals = dict.fromkeys(HEADS, 0.0)
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n), device=data_device)
            if source is None:
                sp, sc = spatial[idx], scalars[idx]
            else:
                sp, sc = _apply_permutation(
                    spatial, scalars, idx, source[idx], s_idx, r_idx
                )
            nll = step_nll(
                model,
                sp.to(device),
                sc.to(device),
                bearing[idx].to(device),
                targets[idx].to(device),
            )
            for h in HEADS:
                totals[h] += float(nll[h].detach().cpu().double().sum())
        return {h: totals[h] / n for h in HEADS}

    baseline = mean_nll(None)
    out = []
    for g, (name, (s_idx, r_idx)) in enumerate(resolved.items()):
        rng = np.random.default_rng([seed, g])
        runs = []
        for _ in range(n_repeats):
            perm = torch.from_numpy(rng.permutation(n)).to(data_device)
            runs.append(mean_nll(perm, s_idx, r_idx))
        for h in HEADS:
            vals = np.array([r[h] for r in runs])
            out.append(
                {
                    "group": name,
                    "head": h,
                    "baseline_nll": baseline[h],
                    "permuted_nll": vals.mean(),
                    "increase": vals.mean() - baseline[h],
                    "sd": vals.std(ddof=1) if n_repeats > 1 else np.nan,
                    "n_repeats": n_repeats,
                }
            )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Movement kernel: parameters and exact summaries
# ---------------------------------------------------------------------------

#: The parameter layout of Params_to_Grid_Block that movement_parameters decodes
_MOVEMENT_LAYOUT = {
    "gamma_shape": (0, 3),
    "gamma_scale": (1, 4),
    "gamma_weight_logit": (2, 5),
    "vm_mu_offset": (6, 9),
    "vm_kappa": (7, 10),
    "vm_weight_logit": (8, 11),
}
_N_MOVEMENT_PARAMS = 12


def movement_parameters(
    model: nn.Module, spatial: torch.Tensor, scalars: torch.Tensor
) -> dict[str, torch.Tensor]:
    """The movement head's raw and decoded kernel parameters for a batch of windows.

    Runs the movement branch exactly as ``ConvJointModel.forward`` does
    (``scalar_grid_output`` → concatenate → ``conv_movement`` →
    ``fcn_movement_all``) and decodes the raw outputs with
    ``model.movement_grid_output``'s own ``_positive`` and
    ``gamma2_scale_factor``.

    Parameters
    ----------
    model:
        Fitted ConvJointModel whose ``movement_grid_output`` is a
        ``Params_to_Grid_Block`` (2-component gamma step lengths, 2-component
        von Mises turning angles; 12 parameters).
    spatial, scalars:
        [B, C, D, D] windows and [B, S] scalar rows, on the model's device.

    Returns
    -------
    dict of tensors
        ``raw`` [B, 12]; and [B, 2] per component: ``gamma_shape``,
        ``gamma_scale`` (CRS units, e.g. m), ``gamma_weight``,
        ``vm_mu_offset`` (radians, relative to the previous bearing),
        ``vm_kappa``, ``vm_weight``; and ``polar_jacobian`` (bool): whether the
        block divides the gamma density by r (``Params_to_Grid_Block_ChV``).

    Raises
    ------
    TypeError, ValueError
        If the kernel block or the number of parameters is not the layout
        decoded here.
    """
    from deepssf.model import Params_to_Grid_Block, Params_to_Grid_Block_ChV

    block = model.movement_grid_output
    if not isinstance(block, Params_to_Grid_Block):
        raise TypeError(
            f"movement_grid_output is {type(block).__name__}; movement_parameters "
            "decodes Params_to_Grid_Block (gamma + von Mises mixtures) only"
        )
    with torch.no_grad():
        x = torch.cat([spatial, model.scalar_grid_output(scalars)], dim=1)
        raw = model.fcn_movement_all(model.conv_movement(x))
    if raw.ndim != 2 or raw.shape[1] != _N_MOVEMENT_PARAMS:
        raise ValueError(
            f"The movement head outputs {tuple(raw.shape[1:])} parameters; the known "
            f"layout has {_N_MOVEMENT_PARAMS} (2 gamma + 2 von Mises components). "
            "Update movement_parameters for a different kernel."
        )

    def pair(name):
        i, j = _MOVEMENT_LAYOUT[name]
        return torch.stack([raw[:, i], raw[:, j]], dim=1)

    scale = block._positive(pair("gamma_scale"))
    scale = scale * torch.tensor([1.0, block.gamma2_scale_factor], device=raw.device)
    return {
        "raw": raw,
        "gamma_shape": block._positive(pair("gamma_shape")),
        "gamma_scale": scale,
        "gamma_weight": torch.softmax(pair("gamma_weight_logit"), dim=1),
        "vm_mu_offset": pair("vm_mu_offset"),
        "vm_kappa": block._positive(pair("vm_kappa")),
        "vm_weight": torch.softmax(pair("vm_weight_logit"), dim=1),
        "polar_jacobian": isinstance(block, Params_to_Grid_Block_ChV),
    }


def mask_window_edges(
    log_surfaces: torch.Tensor | np.ndarray, width: int, renormalise: bool = True
) -> torch.Tensor:
    """Exclude the outer *width* cells of window surfaces.

    The habitat CNN zero-pads every convolution, so within a model window the
    habitat surface (and the joint surface built from it) carries artefacts in
    the outer ``habitat_edge_buffer(model)`` cells, where the receptive field
    reaches past the window.  This sets those cells to ``-inf`` (probability
    0) and, by default, renormalises over the interior.

    Parameters
    ----------
    log_surfaces:
        [..., D, D] log-probability (or log-intensity) surfaces.
    width:
        Border width in cells; usually :func:`deepssf.habitat_edge_buffer`.
    renormalise:
        Subtract the log-sum-exp over the interior, so each surface sums to 1
        over the cells kept.

    Returns
    -------
    torch.Tensor
        Same shape, on the input's device.
    """
    s = torch.as_tensor(log_surfaces)
    d1, d2 = s.shape[-2:]
    w = int(width)
    if w < 0 or 2 * w >= min(d1, d2):
        raise ValueError(f"Edge width {w} leaves nothing of a {d1}x{d2} window")
    out = torch.full_like(s, float("-inf"))
    out[..., w : d1 - w, w : d2 - w] = s[..., w : d1 - w, w : d2 - w]
    if renormalise:
        out = out - torch.logsumexp(out, dim=(-2, -1), keepdim=True)
    return out


def next_step_surfaces(
    model: nn.Module,
    spatial: torch.Tensor,
    scalars: torch.Tensor,
    bearing: torch.Tensor,
    edge_buffer: int | None = None,
) -> dict[str, torch.Tensor]:
    """Habitat, movement and joint next-step log-surfaces for model windows.

    Runs ``model.forward`` and removes the habitat CNN's zero-padding
    artefacts: the outer *edge_buffer* cells of the habitat and joint surfaces
    are excluded (see :func:`mask_window_edges`) and both are renormalised over
    the interior.  The movement kernel is parametric, with no padding
    artefacts, and is returned over the whole window.

    Parameters
    ----------
    model:
        Fitted ConvJointModel (evaluated as-is; call ``model.eval()`` first).
    spatial, scalars, bearing:
        [B, C, D, D], [B, S] and [B, 1] inputs on the model's device.
    edge_buffer:
        Border width to exclude; defaults to
        :func:`deepssf.habitat_edge_buffer`.  0 keeps the whole window.

    Returns
    -------
    dict of [B, D, D] tensors
        ``"habitat"``, ``"movement"`` and ``"joint"`` (habitat + movement,
        renormalised), log-probabilities, ``-inf`` in excluded cells.
    """
    width = habitat_edge_buffer(model) if edge_buffer is None else int(edge_buffer)
    with torch.no_grad():
        out = model((spatial, scalars, bearing))
    return {
        "habitat": mask_window_edges(out[..., 0], width),
        "movement": out[..., 1],
        "joint": mask_window_edges(out[..., 0] + out[..., 1], width),
    }


def kernel_summary(
    log_kernel: torch.Tensor | np.ndarray,
    distance_grid: torch.Tensor | np.ndarray,
    bearing_grid: torch.Tensor | np.ndarray,
    bearing_tm1: Any = None,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75, 0.9),
    stay_radius: float | None = None,
    class_windows: torch.Tensor | np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    edge: int = 0,
) -> pd.DataFrame:
    """Exact summaries of next-step kernels on the model's grid.

    Works for any log-surface over the window: the movement kernel alone, or
    the joint next-step surface (habitat + movement, renormalised).  Nothing
    is sampled; every number is a sum over the grid cells.

    Parameters
    ----------
    log_kernel:
        [B, D, D] (or [D, D]) log-probabilities (renormalised here).
    distance_grid, bearing_grid:
        [D, D] distance from the centre (CRS units) and bearing (radians,
        0 = east, counter-clockwise) of each cell — use
        ``model.movement_grid_output.distance_layer`` and ``.bearing_layer``.
    bearing_tm1:
        [B] previous bearings (radians); turning angles are measured relative
        to them.  None measures absolute bearings.
    quantiles:
        Step-length quantiles, read off the cumulative mass sorted by distance.
    stay_radius:
        Steps shorter than this count as "stay".  Defaults to the smallest
        non-centre distance (one pixel), so P(stay) is the centre cell's mass.
    class_windows, class_names:
        Optional [B, K, D, D] class membership of each window's cells, to
        report the probability that the next step lands in each class.
    edge:
        Width of the border excluded from the kernel (e.g. the
        ``habitat_edge_buffer`` used by :func:`next_step_surfaces` for a joint
        surface); the outer-ring mass is measured just inside it.

    Returns
    -------
    pd.DataFrame, one row per kernel
        ``expected_distance``, ``sd_distance``, ``q<quantile>``, ``p_stay``,
        ``outer_ring_mass`` (mass in the outermost ring of cells the kernel
        covers, *edge* cells in from the window border: a large value means
        the kernel is truncated),
        ``turn_resultant``
        (mean resultant length of the turning angle, 0–1, excluding the
        centre cell) and ``turn_mean_angle`` (radians), and ``p_<class>``.
    """
    lk = torch.as_tensor(log_kernel)
    if lk.ndim == 2:
        lk = lk.unsqueeze(0)
    device = lk.device
    dtype = torch.float32 if device.type == "mps" else torch.float64
    lk = lk.to(dtype)
    n, d1, d2 = lk.shape
    dist = torch.as_tensor(distance_grid).to(device=device, dtype=dtype)
    brg = torch.as_tensor(bearing_grid).to(device=device, dtype=dtype)
    if tuple(dist.shape) != (d1, d2) or tuple(brg.shape) != (d1, d2):
        raise ValueError(
            "distance_grid and bearing_grid must match the kernel's [D, D]"
        )

    p = torch.softmax(lk.reshape(n, -1), dim=1)  # [B, cells]
    dflat = dist.reshape(-1)
    mean = p @ dflat
    var = (p @ dflat**2 - mean**2).clamp_min(0)

    order = torch.argsort(dflat)
    cum = torch.cumsum(p[:, order], dim=1)
    sorted_d = dflat[order]
    out: dict[str, Any] = {"expected_distance": mean, "sd_distance": var.sqrt()}
    for q in quantiles:
        k = torch.searchsorted(cum, torch.full((n, 1), q, dtype=dtype, device=device))
        out[f"q{q:g}"] = sorted_d[k.clamp_max(len(sorted_d) - 1).squeeze(1)]

    centre_d = dflat.min()
    if stay_radius is None:
        stay_radius = float(dflat[dflat > centre_d].min())
    out["p_stay"] = p[:, dflat < stay_radius].sum(dim=1)

    # Outermost ring of cells the kernel covers: `edge` cells in from the window
    # border (0 = the border itself; the excluded width for edge-masked surfaces)
    rows_i = torch.arange(d1, device=device)[:, None].expand(d1, d2)
    cols_i = torch.arange(d2, device=device)[None, :].expand(d1, d2)
    from_edge = torch.minimum(
        torch.minimum(rows_i, d1 - 1 - rows_i), torch.minimum(cols_i, d2 - 1 - cols_i)
    )
    ring = from_edge.reshape(-1) == int(edge)
    out["outer_ring_mass"] = p[:, ring].sum(dim=1)

    moving = dflat > centre_d
    pm = p[:, moving]
    pm = pm / pm.sum(dim=1, keepdim=True).clamp_min(1e-300)
    prev = (
        torch.zeros(n, dtype=dtype, device=device)
        if bearing_tm1 is None
        else torch.as_tensor(bearing_tm1).to(device=device, dtype=dtype).reshape(n)
    )
    turn = brg.reshape(-1)[moving][None, :] - prev[:, None]
    c, s = (pm * torch.cos(turn)).sum(dim=1), (pm * torch.sin(turn)).sum(dim=1)
    out["turn_resultant"] = torch.sqrt(c**2 + s**2)
    out["turn_mean_angle"] = torch.atan2(s, c)

    if class_windows is not None:
        cw = torch.as_tensor(class_windows).to(device=device, dtype=dtype)
        if cw.ndim == 3:
            cw = cw.unsqueeze(0).expand(n, -1, -1, -1)
        k = cw.shape[1]
        names = [f"class_{i}" for i in range(k)] if class_names is None else class_names
        probs = torch.einsum("bc,bkc->bk", p, cw.reshape(n, k, -1))
        for i, name in enumerate(names):
            out[f"p_{name}"] = probs[:, i]

    return pd.DataFrame({k: v.detach().cpu().double().numpy() for k, v in out.items()})


def mixture_moments(decoded: Mapping[str, torch.Tensor]) -> pd.DataFrame:
    """Untruncated moments of the decoded step-length and turning-angle mixtures.

    With the polar Jacobian (``Params_to_Grid_Block_ChV``, the default model)
    the grid density is ``gamma(r) / r × von Mises(θ)`` per unit area, so the
    step length *r* itself follows the gamma mixture.  Without it
    (``Params_to_Grid_Block``) the density per unit area is ``gamma(r)``, so
    *r* follows the size-biased gamma, i.e. shape + 1 (``decoded
    ["polar_jacobian"]`` says which).  These are the
    moments of that mixture on an unbounded plane; the grid summaries in
    :func:`kernel_summary` are the same distribution truncated to the window
    and discretised to cells (and, for a joint kernel, reweighted by habitat).
    Large differences mean the window truncates the kernel.

    Parameters
    ----------
    decoded:
        Output of :func:`movement_parameters`.

    Returns
    -------
    pd.DataFrame, one row per kernel
        ``gamma_mean``, ``gamma_sd`` (mixture), ``gamma1_mean``,
        ``gamma2_mean``, ``gamma2_weight``; ``vm_resultant`` and
        ``vm_mean_turn`` (radians) of the turning-angle mixture relative to
        the previous bearing, using ``A(κ) = I1(κ)/I0(κ)``.
    """
    k = decoded["gamma_shape"].detach().cpu().double()
    if not decoded.get("polar_jacobian", True):
        k = k + 1.0  # r ~ size-biased gamma: density ∝ r * gamma(r)
    th = decoded["gamma_scale"].detach().cpu().double()
    w = decoded["gamma_weight"].detach().cpu().double()
    comp_mean = k * th
    mean = (w * comp_mean).sum(dim=1)
    second = (w * k * (k + 1) * th**2).sum(dim=1)
    kappa = decoded["vm_kappa"].detach().cpu().double()
    a = torch.special.i1e(kappa) / torch.special.i0e(kappa)
    vw = decoded["vm_weight"].detach().cpu().double()
    mu = decoded["vm_mu_offset"].detach().cpu().double()
    c = (vw * a * torch.cos(mu)).sum(dim=1)
    s = (vw * a * torch.sin(mu)).sum(dim=1)
    as_np = lambda t: t.detach().cpu().double().numpy()  # noqa: E731
    return pd.DataFrame(
        {
            "gamma_mean": as_np(mean),
            "gamma_sd": as_np((second - mean**2).clamp_min(0).sqrt()),
            "gamma1_mean": as_np(comp_mean[:, 0]),
            "gamma2_mean": as_np(comp_mean[:, 1]),
            "gamma2_weight": as_np(w[:, 1]),
            "vm_resultant": as_np(torch.sqrt(c**2 + s**2)),
            "vm_mean_turn": as_np(torch.atan2(s, c)),
        }
    )
