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

__all__: list[str] = []
