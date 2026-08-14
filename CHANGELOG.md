# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added
- **`deepssf.predict.predict_habitat_landscape`** — run the trained habitat
  filters over an entire raster in one pass, giving a landscape-scale habitat
  selection surface. The habitat sub-network is fully convolutional (four 3x3
  convolutions, stride 1, padding 1, no pooling or dense layer), so it is
  translation-equivariant: a pixel's value is exactly what the model would
  produce for a window centred on it. The scalar covariates are broadcast to
  full-landscape layers, since `Scalar_to_Grid_Block`'s output size is pinned to
  the training window. Processing is chunked by rows with an `edge_buffer`
  overlap — bit-identical to a single pass, but without the several-hundred-MB
  activations that can exhaust a GPU. `habitat_edge_buffer` reports the exact
  border width that zero-padding contaminates (1 px per conv layer, so 4), which
  is masked to NaN rather than the -inf earlier scripts multiplied through.
- **`EarlyStopping(checkpoint_on=...)`** and **`EarlyStopping(head_paths=...)`**
  — see *Fixed* below.
- **`load_head_weights`** — copy just one head's weights out of a checkpoint,
  leaving the other head untouched. `HEAD_MODULE_PREFIXES` records the
  module-to-head mapping, mirroring the split `set_trainable` freezes on.
- **`simulate_trajectories`** — simulate many trajectories in one call, returning
  them stacked in a long-format DataFrame with `trajectory_id` and `step`
  columns. Three methods: `"batched"` (the default), `"sequential"` (a plain loop
  over `simulate_trajectory`) and `"parallel"` (that loop over a thread pool).
- **`simulate_trajectories_batched`** — steps every trajectory together so the
  model sees a batch of windows per forward pass instead of one. Measured on an
  Apple M-series GPU at `window_size=75`, 200 steps: 2.2× faster than the
  sequential loop at 4 trajectories, 6.9× at 16 and 34× at 64, since the cost per
  step is nearly flat in the number of trajectories. Window extraction is a
  single advanced-indexing gather over a landscape padded once per month, and
  sampling uses `torch.multinomial` over the whole batch. The trade-off is a
  shared clock: `starting_yday`/`starting_hour` must be scalars (start
  *locations* may still vary per trajectory).
  Note that the sub-pixel jitter is scaled from the raster resolution implied by
  the transform, whereas `simulate_next_step` hard-codes a 25 m cell — so batched
  and sequential runs differ slightly on rasters of any other resolution.
- **`save_trajectories`** — write simulated trajectories to CSV under a name that
  records trajectory count, step count and date, appending a run counter rather
  than ever overwriting an existing file. `split=True` writes one file per
  trajectory.
- **`deepssf.plot.plot_trajectories_folium`** — draw simulated and observed
  trajectories over a satellite (or topographic/OSM) basemap, reprojecting from
  the data's projected CRS to EPSG:4326 via `rasterio.warp` (no pyproj axis-order
  trap). Each trajectory is its own toggleable layer. `folium` is an optional
  dependency: `pip install "deepSSF[maps]"`.

### Fixed
- **Sub-pixel jitter was hard-coded to a 25 m cell.** `simulate_next_step`
  scattered each sampled location by a truncated normal on [0, 25] m regardless
  of the raster's actual resolution. On a 20 m raster that put up to 1.25 cells
  of noise on every simulated step and pushed locations outside the cell the
  model had sampled; on a 100 m raster it never left the first quarter of the
  cell. The cell size now comes from the transform, which is the authority on
  the grid the model was trained against. **Simulated trajectories will differ
  from previous versions on any raster that is not 25 m.**
- **A habitat-only training stage saved the wrong epoch.** The combined loss is
  `habitat + movement + logZ`, and `logZ = logsumexp(habitat + movement)` — the
  overlap between the habitat surface and the movement kernel — moves with the
  habitat weights. The combined loss and the habitat loss are the same
  used-versus-available contrast measured against different availability sets:
  the combined loss scores habitat against the cells within reach (~109 effective
  cells of a 75x75 window on a feral-pig model), the habitat loss against the
  whole window (~5460). So with movement frozen the combined loss can keep
  falling — habitat discriminating better among near neighbours — while the
  density it gives the observed location gets *worse*. Since checkpointing keyed
  on the combined loss unconditionally, every epoch of such a stage looked like an
  improvement and overwrote the checkpoint, ending on a habitat surface well past
  its best — in one feral-pig run, habitat bottomed out at epoch 12 of a 10-epoch
  stage and then gave back three quarters of what it had learned while the
  combined loss kept falling.
  `EarlyStopping(checkpoint_on='active')` now keys the checkpoint on whichever
  head is training during a single-head stage, falling back to the combined loss
  when both are; `head_paths=` additionally keeps each head's own best epoch in
  its own file. The default remains `'total'` — for the finished *joint* model,
  and for a habitat surface intended as a *selection* map, that is still the
  right criterion; `'active'` is mainly a diagnostic for whether the habitat head
  is learning at all. `negativeLogLikeLoss`'s docstring now spells out what `logZ`
  is, gives the gradients of both criteria, and notes that neither is a spatial
  operation — the habitat CNN has no positional input and is
  translation-equivariant, so the difference between them is entirely in which
  covariate values enter the available side of the contrast.
- **Environmental layers were not scaled to [0, 1].** `load_environmental_layers`
  divided by the maximum rather than the range — `(data - lo) / hi` instead of
  `(data - lo) / (hi - lo)` — which only lands in [0, 1] when the minimum happens
  to be zero. Sentinel-2 indices stored as ×10 000 (`lo=-10000, hi=10000`) came
  out in [0, 2]; an all-negative layer came out negative *and* axis-flipped,
  reversing the sign of everything the habitat CNN learned from it; a layer with
  `hi == 0` divided by zero. A constant layer now maps to a flat 0 instead of
  raising. **Checkpoints trained on the old scaling will not transfer** — the
  inputs have changed, so retrain.
- **`conv_movement` was registered with no optimiser.** `make_optimisers` covered
  only `fcn_movement_all` and `conv_habitat`, so the movement CNN trunk received
  gradients on every `backward()` but was never zeroed and never stepped. Its
  `.grad` accumulated monotonically across the whole run: the trunk stayed at its
  random initialisation, and once the buffer overflowed `_grads_are_finite` —
  which iterates over every parameter — began returning `False`, silently
  skipping *every* subsequent update, habitat's included. `conv_movement` now
  sits in the movement optimiser alongside `fcn_movement_all`.
- `train_loop` zeroes gradients with `model.zero_grad(set_to_none=True)` rather
  than per-optimiser, so no parameter can be left out of the reset however the
  optimisers are configured. The manual stash-and-restore of habitat gradients
  around the two optimiser steps has been removed: the optimisers own disjoint
  parameter sets and `Adam.step()` only reads `.grad` for its own parameters, so
  it was a no-op.
- **Both learning-rate schedulers stepped on the combined validation loss.** That
  loss is `habitat + movement + logZ` and is dominated by the movement component
  (~1.2 nats of movement against ~0.05 of habitat over a typical run), so the
  habitat learning rate was being cut whenever the *movement* head plateaued.
  Each scheduler now steps on its own component, and only while that component is
  training — stepping a frozen head's scheduler would penalise it for an
  improvement it was never given the chance to make.
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
- **Breaking:** `skip_epoch0_training` has been removed from `train_loop`. It had
  no epoch awareness — it was a per-call switch that `fit` never passed, so its
  default applied to *every* epoch, and with that default at `True` an entire run
  executed forward-only under `torch.set_grad_enabled(False)`. It also measured a
  training-set baseline when the useful number is a validation one. `fit` now runs
  one validation pass before the first epoch and records it at index 0 of every
  history list, with `train_losses[0] = nan`.
- `history` gains a `stage` key recording which stage each epoch belonged to
  (`-1` for the pre-training baseline row).
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
- **`EarlyStopping(monitor='both')`.** Keeps a separate patience budget for the
  habitat and movement components and stops only once every active component has
  plateaued, so a converged movement head can no longer end a run in which habitat
  is still improving. Checkpointing still keys on the combined loss — the joint
  likelihood remains the right model-selection criterion; only the stopping rule
  changes. `reset()` clears the counters at stage boundaries while leaving the
  best score and saved checkpoint intact.
- **`fit(stages=...)`** runs the sub-networks as a coordinate ascent on the same
  joint likelihood, e.g. movement alone to establish an availability kernel, then
  habitat alone to its own convergence, then both. A frozen branch keeps
  contributing its surface to the loss as a fixed offset; only its weights stop
  moving, so the objective is unchanged rather than reweighted. This is not what
  `negativeLogLikeLoss(freeze_movement=True)` does — that flag *drops* the movement
  surface from the loss, which would make the habitat head learn a distance-decay
  bump in order to explain the movement structure itself. Early stopping ends the
  current stage and advances to the next. `reset_optimiser_state` (default `True`)
  clears the Adam moments of a branch when it becomes active again.
- `set_trainable(model, *, habitat, movement)` toggles `requires_grad` per
  sub-network. Exported from the package root.
- See `Habitat_movement_loss_balancing.md` for why the fix for the
  habitat/movement asymmetry is a monitoring change rather than a loss weight.
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
