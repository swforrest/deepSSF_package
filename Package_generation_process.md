# Package Generation Process

This document records what was done to convert the deepSSF research codebase
(originally a collection of Jupyter notebooks) into the `deepssf` installable
Python package.

---

## 1. Starting Point

The repo contained:

- A monolithic analysis notebook (`deepSSF_train_validate_s2.ipynb`) with all
  model definition, data loading, training, simulation, and validation code
  written inline.
- A separate simulation notebook (`deepSSF_simulations.ipynb`).
- No `pyproject.toml`, no `src/` layout, no tests, no installable package.

---

## 2. Package Scaffold

### Directory layout

The standard `src`-layout was adopted so the package is importable only after
installation (editable or otherwise), preventing accidental imports from the
working directory:

```
deepSSF_package/
├── src/
│   └── deepssf/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── simulate.py
│       ├── validate.py
│       ├── utils.py
│       └── datasets/data/      # bundled test dataset (CSV + 2 GeoTIFFs)
├── tests/
│   ├── __init__.py
│   ├── test_deepssf.py
│   └── test_smoke.py
├── examples/
│   └── deepssf_train_validate_example.ipynb
├── pyproject.toml
├── environment.yml
├── README.md
├── CHANGELOG.md
├── CITATION.cff
└── LICENSE
```

### `pyproject.toml`

A single `pyproject.toml` replaces `setup.py` / `setup.cfg` / `requirements.txt`.
Key decisions:

- Build backend: `hatchling` (version read dynamically from `__init__.py`).
- Runtime deps declared once: `numpy`, `torch`, `pandas`, `matplotlib`,
  `rasterio`, `imageio`.
- Optional dep groups:
  - `[dev]` — `pytest`, `ruff`, `pre-commit`, `build`, `twine`.
  - `[examples]` — `jupyterlab`, `ipykernel` (used by `environment.yml`).
- `requires-python = ">=3.10"`.

---

## 3. Code Ported from Notebooks

Each notebook section was extracted into its own module with type annotations,
docstrings, and cleaned-up interfaces.

### `src/deepssf/model.py`

- `ModelParams` — thin wrapper around a config dict; attributes accessible by
  name.
- `Conv2d_block_spatial` — habitat sub-network (conv stack → log-softmax over
  H×W).
- `Scalar_to_Grid_Block` — broadcasts scalar covariates into spatial maps.
- `Conv2d_block_toFC` — movement sub-network (conv stack → flattened dense
  layer).
- `ConvJointModel` — combines all three blocks; forward pass returns
  `[B, H, W, 2]` (habitat channel + movement channel).

### `src/deepssf/train.py`

- `negativeLogLikeLoss` — factory returning a loss function that reads the
  observed next-step pixel `(px2, py2)` from the batch; supports `reduction`
  in `{"mean", "sum", "none", "median"}` and a `freeze_movement` flag for
  curriculum training.
- `EarlyStopping` — patience-based stopping with checkpoint saving.
- `make_optimisers` — returns two `Adam` optimisers (one per sub-network) with
  `ReduceLROnPlateau` schedulers.
- `train_loop` / `test_loop` — single-epoch passes.
- `fit` — full training loop with early stopping, snapshot saving, and loss
  history dict.

### `src/deepssf/data.py`

- `load_s2_data` — loads all `S2_*.tif` monthly composites from a directory;
  scales DN → surface reflectance.
- `load_environmental_layers` — dispatches to `load_s2_data` for the S2
  directory; loads other layers from TIFF (scaled to [0, 1]) or `.npy` files.
- `MovementDataset` — `torch.utils.data.Dataset`; `__getitem__` extracts a
  spatial patch centred on the departure pixel and returns
  `(spatial, scalars, bearing_tm1, next_step_pixel, transform)`.
- `prepare_movement_df` — converts raw telemetry (one row per fix) to step
  format (one row per consecutive pair). Outputs: `x1_`, `y1_`, `x2_`, `y2_`,
  `t1_`, `dx`, `dy`, `bearing`, `bearing_tm1`, `dt_hour`, `hour_t1`,
  `yday_t1`, and four cyclic encodings.
- `filter_steps_by_window` *(new)* — drops steps whose displacement `|dx|` or
  `|dy|` exceeds `(window_size - 1) * pixel_size / 2`. Must be called after
  `prepare_movement_df` and before `make_dataloaders` to prevent out-of-bounds
  pixel indices in the loss function (see §6 below).
- `make_dataloaders` — convenience wrapper that builds `MovementDataset` and
  splits it into train / val / test `DataLoader`s.

### `src/deepssf/utils.py`

- `get_device` — selects MPS → CUDA → CPU at runtime.
- `clear_memory` — frees GPU/MPS cache.
- `subset_raster_with_padding_torch` / `_npy` — crops a raster window with
  −1 padding at boundaries.
- `subset_raster_all_bands_torch` — multi-band variant.
- `subset_layer_vectorized` — fast NumPy/Torch crop used inside
  `MovementDataset.__getitem__`.
- `recover_hour` / `recover_yday` — invert cyclic sine/cosine encodings.
- `create_gif` — assembles training snapshot PNGs into an animated GIF.

### `src/deepssf/simulate.py`

Ported from `deepSSF_simulations.ipynb`:

- `make_simulation_inputs` — pre-computes cyclic scalar arrays for N steps
  given a starting day-of-year and hour.
- `simulate_next_step` — runs one forward pass; samples the next location from
  the joint probability map; returns new coordinates and three log-prob tensors.
- `simulate_trajectory` — loops `simulate_next_step` for N steps and returns a
  DataFrame of `(x, y, hour, yday, month_index)`.

### `src/deepssf/validate.py`

- `validate_next_step_probs` — iterates over observed steps, runs the model,
  and appends `habitat_prob`, `move_prob`, and `next_step_prob` columns to the
  input DataFrame.

### `src/deepssf/__init__.py`

Curated public API: re-exports every user-facing name from all five modules,
plus `__version__ = "0.1.0"`. `__all__` is explicit.

---

## 4. Example Notebook

`examples/deepssf_train_validate_example.ipynb` is a cleaned-up, reproducible
version of the original analysis notebook. It uses only the public package API:
no inline model or data-prep code remains. Key changes from the original:

- Imports `from deepssf import ...` throughout.
- Defines `WINDOW_SIZE = 25` and `PIXEL_SIZE = 25` as named constants.
- Calls `filter_steps_by_window(step_df, window_size=WINDOW_SIZE, pixel_size=PIXEL_SIZE)`
  immediately after `prepare_movement_df`, before `make_dataloaders`.
- Uses `PIXEL_SIZE` (not a bare literal) in `ModelParams`.
- Saves all outputs under `examples/outputs/`.

---

## 5. Test Suite

`tests/test_deepssf.py` — 48 tests covering every public function:

| Area | Tests |
|------|-------|
| `deepssf.utils` | `get_device`, `recover_hour/yday` roundtrips, `subset_raster_*` shapes and padding, `clear_memory` |
| `deepssf.data` | `extract_year_month_regex`, `day_to_month_index`, `prepare_movement_df` columns / row count / bearing / cyclic range / dx/dy / bearing_tm1, `filter_steps_by_window`, `MovementDataset.__getitem__` shapes, `make_dataloaders` (df= and prepare= variants) |
| `deepssf.model` | `ModelParams` construction, `Conv2d_block_spatial` output shape, `Scalar_to_Grid_Block` output shape, full forward pass shape, habitat log-normalisation |
| `deepssf.train` | `negativeLogLikeLoss` (mean, sum, none, median, freeze_movement, invalid reduction), `EarlyStopping` counter / reset, `make_optimisers`, `fit` loss history |
| `deepssf.simulate` | `make_simulation_inputs` shape / cyclic encoding / hour wrap, `simulate_next_step` return types and shapes, `simulate_trajectory` DataFrame shape and columns |
| `deepssf.validate` | `validate_next_step_probs` column presence and row-0 == 0 rule, `_day_to_s2_month` range |

Ruff (linting + import sorting) is configured in `pyproject.toml` and passes
clean on all source and test files.

---

## 6. Issues Encountered and Resolved

### OpenMP duplicate-library crash on macOS (MPS)

**Symptom**: `OMP Error #15: Initializing libomp.dylib, but found libomp.dylib
already initialized` — process crashed on the first MPS forward pass.

**Root cause**: Two separate copies of libomp were loaded:
1. pip-installed `torch` bundles its own `libomp.dylib`.
2. conda-forge's `rasterio` pulled in `libopenblas` (OpenMP build) →
   `llvm-openmp`, a second libomp.

Additionally, the system `~/.condarc` had `channel_priority: flexible` plus a
`defaults` channel entry, which caused conda to pre-install the full
conda-forge pytorch ecosystem as orphaned packages before pip had a chance to
install torch.

**Fix** (two parts):
1. Added `- nodefaults` to `environment.yml` channels to prevent the system
   `~/.condarc` channels from being merged into the solve.
2. Moved `rasterio` from the conda section to the pip section. The macOS ARM64
   pip wheel for rasterio bundles GDAL/PROJ internally, so conda-forge never
   installs libopenblas or llvm-openmp. The conda section now installs only
   pure-Python packages (jupyterlab, ipykernel) with no C extension or BLAS
   dependencies.

### torch 2.12 stricter MPS bounds checking

**Symptom**: `AcceleratorError: index 25 is out of bounds for dimension 1 with
size 25` during the first training epoch on MPS.

**Root cause**: `MovementDataset.__getitem__` computes the next-step pixel
index as `px2 - col_start`. For `window_size=25`, valid indices are 0–24. A
step whose displacement exceeds 12 pixels (i.e. 300 m at 25 m/pixel) produces
a local index of 25, which is out of bounds. torch 2.5.1 silently allowed
this; torch 2.12 on MPS correctly raises an error.

**Fix**: Rather than clamping the index (which would silently corrupt the loss
signal), the correct fix is to remove those steps before training. This
matches the original notebook's pattern. The solution was to:
- Add `dx` and `dy` columns to `prepare_movement_df` output (the actual x/y
  displacements in CRS units).
- Add `hour_t1` column (needed by the simulation cell).
- Add the `filter_steps_by_window(df, window_size, pixel_size)` function.
- Call it in the example notebook between `prepare_movement_df` and
  `make_dataloaders`.

### macOS case-insensitive filesystem collision

The original development conda environment was named `deepSSF`. Running
`conda env remove -n deepssf` (lowercase) on macOS's case-insensitive APFS
filesystem removed `/opt/miniconda3/envs/deepSSF` — the two names are
identical to the OS. The new `deepssf` environment (created from
`environment.yml`) is now the primary development environment.

---

## 7. Environment Setup (`environment.yml`)

A conda/pip hybrid environment for reproducible setup on any platform:

```yaml
name: deepssf

channels:
  - conda-forge
  - nodefaults    # prevent system ~/.condarc channels from being merged in

dependencies:
  - python=3.11
  - pip

  # Jupyter — installed via conda for reliable kernel registration
  - jupyterlab>=4.0
  - ipykernel>=6.0

  - pip:
    # Plain torch — pip selects MPS on macOS, CUDA on NVIDIA, CPU elsewhere
    - torch>=2.0

    # rasterio via pip — ARM64/Linux wheels bundle GDAL/PROJ internally,
    # avoiding the conda-forge openblas → llvm-openmp conflict with torch's libomp
    - rasterio>=1.3

    # Install the package + Jupyter extras; all other deps come from pyproject.toml
    - -e ".[examples]"
```

### Verification (Apple Silicon, MPS)

After `conda env create -f environment.yml && conda activate deepssf`:

| Check | Result |
|-------|--------|
| `torch.backends.mps.is_available()` | `True` |
| `import deepssf, rasterio` | OK |
| `pytest` | 48 passed, 1 warning |
| `ruff check src/ tests/` | All checks passed |
| Example notebook end-to-end | Completed; produced `best_model.pt`, `loss_history.png`, `simulated_trajectory.png`, `training_progress.gif`, `validation_probs.png` |

---

## 8. README

A "Setting up (for users new to Python)" section was added as the first major
section, aimed at R users unfamiliar with Python packaging:

- Frames a conda environment as equivalent to an `renv` project library.
- Links to the official Miniconda page; notes Windows users should use the
  Anaconda Prompt and avoid paths with spaces.
- Mentions Miniforge as an alternative for those who prefer to avoid
  Anaconda's default channel.
- Copy-pasteable commands: `conda env create`, `conda activate`, optional
  `ipykernel install`, `jupyter lab`.
- Notes that PyTorch auto-selects MPS / CUDA / CPU with no manual configuration.
