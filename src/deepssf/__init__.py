"""deepssf — deep learning step selection functions for animal movement.

This top-level module defines the *public API*: the names users get when they
do ``import deepssf`` or ``from deepssf import ...``. Keep it curated — only
expose the things you want people to rely on. Internal helpers stay private.
"""

# Single source of truth for the version. hatchling reads this string at build
# time (see pyproject.toml [tool.hatch.version]). Bump it when you release.
__version__ = "0.1.0"

from deepssf.data import (
    MovementDataset,
    load_environmental_layers,
    load_s2_data,
    make_dataloaders,
)
from deepssf.model import ConvJointModel, ModelParams
from deepssf.simulate import make_simulation_inputs, simulate_next_step, simulate_trajectory
from deepssf.validate import validate_next_step_probs
from deepssf.train import EarlyStopping, negativeLogLikeLoss, test_loop, train_loop
from deepssf.utils import (
    clear_memory,
    create_gif,
    get_device,
    recover_hour,
    recover_yday,
    subset_layer_vectorized,
    subset_raster_all_bands_torch,
    subset_raster_with_padding_npy,
    subset_raster_with_padding_torch,
)

__all__ = [
    "__version__",
    # model
    "ConvJointModel",
    "ModelParams",
    # train
    "negativeLogLikeLoss",
    "EarlyStopping",
    "train_loop",
    "test_loop",
    # data
    "MovementDataset",
    "load_environmental_layers",
    "load_s2_data",
    "make_dataloaders",
    # simulate
    "make_simulation_inputs",
    "simulate_next_step",
    "simulate_trajectory",
    # validate
    "validate_next_step_probs",
    # utils
    "get_device",
    "clear_memory",
    "create_gif",
    "recover_hour",
    "recover_yday",
    "subset_layer_vectorized",
    "subset_raster_with_padding_torch",
    "subset_raster_all_bands_torch",
    "subset_raster_with_padding_npy",
]
