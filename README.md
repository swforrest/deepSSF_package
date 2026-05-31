# deepSSF

Predicting animal movement with a deep learning step-selection framework.

This package provides the installable implementation of the deepSSF 
approach in Python. Accompanying tutorials and example code live at the
[deepSSF project site](https://swforrest.github.io/deepSSF/). 

There is a package walkthrough script that outlines an implementation of the functions with some example GPS tracking data (a single water buffalo) and two spatial covariates (NDVI and slope). You can access the script as a Jupyter notebook or as a knitted html in the [examples](https://github.com/swforrest/deepSSF_package/tree/main/examples) directory. The example script will not download with the package installation, but the GPS and spatial datasets will.

![](icons/both_icons.png)

The Python package can be viewed on [PyPi](https://pypi.org/project/deepSSF/).

The paper can be found at:

Forrest, S. W., Pagendam, D., Hassan, C., Potts, J. R., Drovandi, C., Bode, M., & Hoskins, A. J. (2026). **Predicting animal movement with deepSSF : A deep learning step selection framework**. Methods in Ecology and Evolution, 17(2), 371–391. [https://doi.org/10.1111/2041-210x.70136](https://doi.org/10.1111/2041-210x.70136).

## Installation (pip only)
 
If you manage your own Python environment, install deepSSF with:

```bash
pip install deepssf
```

Development install (editable, with linting and testing tools):

```bash
git clone https://github.com/swforrest/deepssf
cd deepssf
pip install -e ".[dev]"
```

## Quick start

```python
import deepssf
print(deepssf.__version__)
```

## Setting up (for users new to Python)

If you are coming from R, think of a conda environment the way you think of
an `renv` project library — it is a self-contained Python installation that
keeps this project's packages separate from everything else on your computer.
The steps below create one for deepSSF and should take about five minutes.

### 1. Install Miniconda (once, system-wide)

Download and run the installer from the
[official Miniconda page](https://docs.anaconda.com/miniconda/).

If you click 'Download' towards the top right, the links to download 
Miniconda are towards the bottom of the page - the links at the top are
for the Anaconda Distribution, which has thousands of pacakges and is
not necessary to get things up and running initially.

- **Windows**: use the **Anaconda Prompt** for all subsequent commands, and
  choose an install path that contains **no spaces** (e.g. `C:\miniconda3`).
- **macOS / Linux**: a normal terminal works fine.

> **Miniforge alternative**: if you prefer to avoid Anaconda's default channel
> entirely, [Miniforge](https://github.com/conda-forge/miniforge) is a
> drop-in replacement that ships with `conda-forge` as the only channel.

### 2. Create the environment

```bash
git clone https://github.com/swforrest/deepssf_package
cd deepssf
conda env create -f environment.yml
```

This installs Python 3.11, the geospatial libraries (rasterio / GDAL / PROJ),
Jupyter Lab, and the deepSSF package itself with all of its dependencies.
PyTorch is installed via pip with no extra flags — pip automatically picks the
right build for your hardware: **MPS on Apple Silicon, CUDA on NVIDIA GPUs,
CPU everywhere else**. No configuration is needed; the package selects the
correct backend at runtime.

### 3. Activate the environment

```bash
conda activate deepssf
```

You will need to run this once per terminal session before using deepSSF.

### 4. (Optional) Register the Jupyter kernel

If you use VS Code or another editor that manages its own Jupyter kernel list,
register the environment so it appears as a kernel option:

```bash
python -m ipykernel install --user --name deepssf --display-name "Python (deepssf)"
```

### 5. Launch Jupyter Lab

```bash
jupyter lab
```

Then open `examples/deepssf_train_validate_example.ipynb` to get started.

---

## Documentation

Tutorials and walkthroughs: https://swforrest.github.io/deepSSF/

## Citation

If you use deepssf in your research, please cite the paper. See `CITATION.cff` or use the citation and link to paper below.

Forrest, S. W., Pagendam, D., Hassan, C., Potts, J. R., Drovandi, C., Bode, M., & Hoskins, A. J. (2026). **Predicting animal movement with deepSSF : A deep learning step selection framework**. Methods in Ecology and Evolution, 17(2), 371–391. https://doi.org/10.1111/2041-210x.70136

## License

MIT — see [LICENSE](LICENSE).
