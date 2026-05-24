# deepssf

Deep learning step selection functions for predicting animal movement.

This package provides the reusable, installable implementation of the deepSSF
method. The accompanying paper, tutorials, and reproducibility code live at the
[deepSSF project site](https://swforrest.github.io/deepSSF/).

## Installation

```bash
pip install deepssf
```

Development install (editable, with tooling):

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

## Documentation

Tutorials and walkthroughs: https://swforrest.github.io/deepSSF/

## Citation

If you use deepssf in your research, please cite the paper. See `CITATION.cff` or use the citation and link to paper below.

Forrest, S. W., Pagendam, D., Hassan, C., Potts, J. R., Drovandi, C., Bode, M., & Hoskins, A. J. (2026). **Predicting animal movement with deepSSF : A deep learning step selection framework**. Methods in Ecology and Evolution, 17(2), 371–391. https://doi.org/10.1111/2041-210x.70136

## License

MIT — see [LICENSE](LICENSE).
