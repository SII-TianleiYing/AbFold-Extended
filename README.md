# AbFold Extended

AbFold Extended packages the `abfold` Python modules used by the AbDiff
pipeline. The PyPI distribution name is `abfold-extended`, while the Python
import package remains `abfold` for compatibility with existing AbDiff code.

This repository is based on the original
[HK-GSAS/AbFold](https://github.com/HK-GSAS/AbFold) implementation, with
engineering changes for packaging and AbDiff integration.

## Installation

Install the package from PyPI:

```bash
pip install abfold-extended==0.2.0
```

Code should continue to import `abfold`:

```python
from abfold.config import config
from abfold.model import AbFold
```

For the AbDiff H3-mask stage, install the optional ANARCI extra:

```bash
pip install "abfold-extended[h3]==0.2.0"
```

Optional extras:

```bash
pip install "abfold-extended[diffusion]==0.2.0"
pip install "abfold-extended[train]==0.2.0"
pip install "abfold-extended[all]==0.2.0"
```

## Package Scope

This MVP package is intended to make AbDiff's AbFold-facing imports stable:

```python
from abfold.config import config
from abfold.model import AbFold
from abfold.data.data_process import process_repr, process_fasta
from abfold.data.data_process import get_CDRs_mask_with_anarci
from abfold.np import protein
from abfold.np.residue_constants import str_sequence_to_aatype
from abfold.train_ema import tensor_dict_to_device
from abfold.training_config import config as training_config
```

The package does not include pretrained checkpoints, benchmark data, AF2
representations, IgFold embeddings, or a standalone `predict.py` entry point.
Those assets must be supplied by the calling workflow.

## AbDiff Usage

AbDiff can depend on this package by installing `abfold-extended` while keeping
its existing source imports as `from abfold ...`.

For a full AbDiff pipeline environment, use the optional extras needed by the
stages you run. In particular, Stage 4 requires ANARCI:

```bash
pip install "abfold-extended[h3]==0.2.0"
```

## Development Checks

Run the lightweight smoke tests:

```bash
python -m unittest discover -s tests -v
```

Run the wheel build and install smoke test:

```bash
ABFOLD_WHEEL_SMOKE=1 python -m unittest discover -s tests -v
```

On Windows PowerShell:

```powershell
$env:ABFOLD_WHEEL_SMOKE = "1"
python -m unittest discover -s tests -v
```

## Notes

- The distribution name and import name are intentionally different:
  `pip install abfold-extended`, then `import abfold`.
- Optional dependencies such as `diffusers`, `deepspeed`, and `anarci` are
  loaded only by the features that need them.
- Published PyPI files are immutable. If a release is flawed, publish a new
  version and yank the flawed release instead of trying to replace it.

## Citation

```bibtex
@article{abfold,
    title = {AbFold -- an AlphaFold Based Transfer Learning Model for Accurate Antibody Structure Prediction},
    author = {Peng, Chao and Wang, Zelong and Zhao, Peize and Ge, Weifeng and Huang, Charles},
    journal = {bioRxiv},
    year = {2023}
}
```
