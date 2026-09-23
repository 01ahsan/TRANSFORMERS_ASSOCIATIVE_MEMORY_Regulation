# Head Geometry and Associative Memory

Training and analysis code for measuring how attention head size and count
affect associative recall. Includes saved result tables for plotting without
rerunning training.

- `experiments/`: training programs and controls.
- `analysis/`: statistical analysis and paper plots.
- `results/`: saved CSV/JSON inputs.

## Setup

Use Python 3.11. Run the commands below from the repository root.

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then install:

```sh
python -m pip install -r requirements.txt
```

Plotting and analysis run on CPU. Training needs CUDA: two GPUs for
`wikitext_capacity.py`, one or two for the other experiments. Install a
PyTorch build compatible with your CUDA environment.

## Plot the saved results

```sh
python analysis/paper_assets.py --project-root results
```

Writes tables and plots to `outputs/paper/`. Use `--output-dir` to choose
another directory.

The WikiText capacity point-summary CSV is missing. The command reports this
and skips that curve; the capacity-boundary plot still works. Add
`--strict-main` to fail on missing main-paper inputs instead.

## Run the experiments

The first run downloads the datasets and encoder. Caches, checkpoints, and
results go under `outputs/`. Set `HEAD_GEOMETRY_ROOT` to change the run root;
use a fresh directory for an independent run.

Run the WikiText stages in this order:

```sh
python experiments/wikitext_capacity.py
python experiments/qk_intervention.py
python experiments/head_factorial.py
python analysis/regime_dynamics.py
python experiments/wikitext_confirmation.py
```

Then run the AG News stages:

```sh
python experiments/agnews_transfer.py
python experiments/agnews_capacity.py
python experiments/agnews_confirmation.py
python experiments/head_partition.py
```

To plot a new run:

```sh
python analysis/paper_assets.py --project-root outputs --output-dir outputs/paper
```

Each program supports `--help`. See [reproducibility](docs/reproducibility.md)
for stage dependencies, path overrides, and environment details.

## Data and scope

The experiments use WikiText-103 and AG News, with frozen `all-MiniLM-L6-v2`
embeddings. Dataset loading, preprocessing, and citations are in
[data.md](docs/data.md); BibTeX entries are in [references.bib](references.bib).

Datasets, model weights, and checkpoints are downloaded or generated locally.
Some historical synthetic programs and the final integrity-audit utility are
not included. The code and result records for the main confirmatory claims
are retained; [provenance](docs/provenance.md) documents the gaps and checks.
Full GPU training was not rerun for this release.

See [LICENSE](LICENSE) for license status.
