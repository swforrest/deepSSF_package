"""deepssf — deep learning step selection functions for animal movement.

This top-level module defines the *public API*: the names users get when they
do ``import deepssf`` or ``from deepssf import ...``. Keep it curated — only
expose the things you want people to rely on. Internal helpers stay private.
"""

# Single source of truth for the version. hatchling reads this string at build
# time (see pyproject.toml [tool.hatch.version]). Bump it when you release.
__version__ = "0.1.0"

# Re-export the key pieces so users can write `deepssf.DeepSSF` instead of
# `deepssf.model.DeepSSF`. Uncomment as you implement each one.
# from deepssf.model import DeepSSF
# from deepssf.data import prepare_data
# from deepssf.train import train
# from deepssf.simulate import simulate_trajectory
# from deepssf.validate import validate

__all__ = [
    "__version__",
    # "DeepSSF",
    # "prepare_data",
    # "train",
    # "simulate_trajectory",
    # "validate",
]
