# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Fixed
- **NaN movement probability surfaces during training.** In float32 the von
  Mises normaliser `I₀(κ)` overflows to `inf` at κ ≈ 89 — reachable from a raw
  FCN output of only `log(89) ≈ 4.5`. With both mixture components overflowing,
  the log-sum-exp evaluated `-inf - (-inf)` and the entire surface became NaN;
  with only one overflowing, the forward pass stayed finite but produced NaN
  gradients that silently propagated into every weight on the next optimiser
  step. `Params_to_Grid_Block._vonmises_log` now uses the exponentially-scaled
  Bessel function, `log I₀(κ) = κ + log(i0e(κ))`, giving
  `κ·(cos(θ-μ) - 1) - log(2π) - log(i0e(κ))`, whose exponent is ≤ 0 for every κ.
- Mixture components are combined with `torch.logsumexp` instead of a manual
  max/exp/log, which returned NaN when both components were `-inf`.
- Raw FCN outputs are clamped (`Params_to_Grid_Block.raw_clamp`, default 10.0)
  before exponentiation. Previously `exp` overflowed to `inf` at a raw value of
  ≈ 88, making `lgamma(inf) - inf` NaN, and scale underflowed to 0 at ≈ -88.
- `torch.special.i0e` is called on the active device. The previous `.cpu()`
  round-trip for MPS forced a host sync on every forward pass; `i0`/`i0e` now
  have CPU, CUDA and MPS kernels registered (verified on torch 2.12).
- `distance_layer` and `bearing_layer` are registered buffers, so `model.to()`
  moves them once instead of copying them host→device on every forward pass.
  They use `persistent=False`: both are constants derived from `image_dim` and
  `pixel_size`, so they stay out of `state_dict` and checkpoint keys are
  unchanged.
- `train_loop` skips batches whose loss or gradients are not finite rather than
  stepping the optimiser on them, and excludes them from the epoch mean. A
  single bad batch no longer makes a run unrecoverable.
- The epoch mean is now averaged over contributing batches, not `num_batches`.
- Corrected the `Params_to_Grid_Block_ChV` docstring: it subtracts `log(r)` from
  the log-density (dividing the density by `r`), not "by `log(r)`".

### Changed
- **Breaking (checkpoints):** mixture weights are now `log_softmax` over the raw
  FCN outputs. Previously `softmax(exp(raw))` was applied — a double exponential
  that saturated to exactly `(1, 0)` by a raw value of ≈ 4.75, zeroing the weight
  gradient and putting `log(0) = -inf` into the surface. Movement parameters
  2, 5, 8 and 11 therefore have a different meaning, so checkpoints written by
  ≤ 0.2.3 will be reinterpreted rather than failing to load. The step-length and
  turning-angle densities themselves are unchanged (agreement to ~6e-6 when the
  two weights are equal).
- `negativeLogLikeLoss` raises a `RuntimeWarning` once per non-finite surface
  instead of printing on every batch, and performs one device sync per batch
  rather than three.
- `_expand` uses `expand` rather than `repeat`, returning a view instead of
  materialising a `[B, H, W]` copy per parameter.

### Added
- **Versioned checkpoints.** `EarlyStopping` now writes a dict with
  `deepssf_checkpoint_format`, `deepssf_version`, `val_loss` and `state_dict`
  keys instead of a bare `state_dict` (`CHECKPOINT_FORMAT = 2`). Because the
  `state_dict` keys themselves did not change, a pre-0.3.0 file would otherwise
  load without error while silently misreading movement parameters 2, 5, 8
  and 11 — the marker makes that case fail loudly.
- `load_checkpoint(path, model=None, *, map_location=None, allow_legacy=False,
  strict=True)` reads a checkpoint, verifies its format, and optionally loads it
  into a model. Legacy (format 1) files raise `RuntimeError` unless
  `allow_legacy=True`; files from a future format are refused with an
  upgrade message. Exported from the package root alongside `CHECKPOINT_FORMAT`.
- `grad_clip` argument on `fit` and `train_loop` for global gradient-norm
  clipping (`None` by default, so existing behaviour is unchanged).
- Regression tests covering forward and backward finiteness at extreme movement
  parameters, log-normalisation of the surface, and non-vanishing mixture-weight
  gradients.
- Initial package scaffold.

## [0.1.0] - TBD
- First release.
