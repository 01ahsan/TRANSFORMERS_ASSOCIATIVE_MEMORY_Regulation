# Head geometry and associative memory

Code for our submission on how the number and size of attention heads affect
associative recall. The saved results in `results/` are enough to regenerate
all figures and tables in the paper without retraining anything.

    experiments/   training scripts and controls
    analysis/      statistics and figure generation
    results/       saved CSV/JSON results used in the paper
    docs/          notes on data, reproducibility and provenance

## Setup

We used Python 3.11.

    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Install whichever PyTorch build matches your CUDA setup. Analysis and plotting
run fine on CPU. Training needs CUDA: `wikitext_capacity.py` expects two GPUs,
and the other experiments use one or two.

## Regenerating the figures

    python analysis/paper_assets.py --project-root results

Output goes to `outputs/paper/` by default (`--output-dir` to change it).

One input is missing: the point-summary CSV for the WikiText capacity sweep.
The script warns and skips that curve. Everything else is still produced,
including the capacity-boundary plot. Pass `--strict-main` if you'd rather
it fail on missing main-paper inputs.

## Training from Beginning

The first run downloads the datasets and the encoder. Caches, checkpoints and
results are written under `outputs/`. Set `HEAD_GEOMETRY_ROOT` to put them
elsewhere, and point it at an empty directory if you want a clean,
independent run.

Later stages read earlier outputs, so order matters. WikiText first:

    python experiments/wikitext_capacity.py
    python experiments/qk_intervention.py
    python experiments/head_factorial.py
    python analysis/regime_dynamics.py
    python experiments/wikitext_confirmation.py

then AG News:

    python experiments/agnews_transfer.py
    python experiments/agnews_capacity.py
    python experiments/agnews_confirmation.py
    python experiments/head_partition.py

and plot the new run with

    python analysis/paper_assets.py --project-root outputs --output-dir outputs/paper

Every script takes `--help`. Stage dependencies, path overrides and our
environment are described in [docs/reproducibility.md](docs/reproducibility.md).

## Data

WikiText-103 and AG News, embedded with a frozen `all-MiniLM-L6-v2`.
Loading, preprocessing and dataset citations are in [docs/data.md](docs/data.md);
BibTeX is in [references.bib](references.bib).
