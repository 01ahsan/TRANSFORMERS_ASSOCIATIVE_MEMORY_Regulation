# Head Geometry and Associative Memory

Code for experiments separating attention head dimension, head count, and
total attention width in associative memory. This is the anonymous review
release. Training programs are in `experiments/`; statistical analysis and
paper asset generation are in `analysis/`.

The real-text experiments use WikiText-103
([Merity et al., 2017](https://openreview.net/forum?id=Byj72udxe)) and AG News
([Zhang et al., 2015](https://papers.nips.cc/paper_files/paper/2015/hash/250cf8b51c773f3f8dc8b4be867a9a02-Abstract.html)).
Both corpora are encoded with the frozen `all-MiniLM-L6-v2` sentence encoder
([Reimers and Gurevych, 2019](https://aclanthology.org/D19-1410/);
[Wang et al., 2020](https://papers.nips.cc/paper_files/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)),
producing 384-dimensional normalized representations. Queries are deterministic
10% word-drop views of their keys. AG News category labels are unused; the
corpus provides an independent text-representation distribution. See
[data preparation](docs/data.md) and [BibTeX references](references.bib).

## Setup

Use Python 3.11. From the repository root:

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then install:

```sh
python -m pip install -r requirements.txt
```

Training requires CUDA and a compatible PyTorch installation. The WikiText
capacity experiment requires two visible GPUs; the other training programs
support one or two. Analysis runs on CPU.

## Paper results

The compact CSV/JSON inputs are included in `results/`. Generate the supported
paper tables and plots without training:

```sh
python analysis/paper_assets.py
```

Outputs are written to `outputs/paper/`. The available inputs generate 67
assets. The WikiText capacity point-summary CSV is missing, so its
capacity-curves figure is skipped; the capacity-boundary figure is supported.
WikiText summary tables retained from paper exports are identified separately
from run-archive records in [result provenance](results/provenance.json).

## Experiments

Run from the repository root. Each program downloads its public inputs and
writes caches and run artifacts beneath `outputs/`. Set `HEAD_GEOMETRY_ROOT`
to change this location. Use a fresh directory for an independent run.

The WikiText sequence is:

```sh
python experiments/wikitext_capacity.py
python experiments/qk_intervention.py
python experiments/head_factorial.py
python analysis/regime_dynamics.py
python experiments/wikitext_confirmation.py
```

The AG News sequence is:

```sh
python experiments/agnews_transfer.py
python experiments/agnews_capacity.py
python experiments/agnews_confirmation.py
python experiments/head_partition.py
```

After completing the relevant experiments, generate paper tables and plots:

```sh
python analysis/paper_assets.py --project-root outputs --output-dir outputs/paper
```

Run any program with `--help` for its options. Dependencies between stages,
path overrides, and reproduction limits are described in
[reproducibility](docs/reproducibility.md).

Datasets, checkpoints, embedding caches, and generated figures are not
included. Some historical synthetic-discovery programs and the final integrity-audit
utility are not included in this release. These omissions do not affect the
code or result records underlying the main confirmatory claims; see
`docs/provenance.md` for details.

No full GPU training was rerun for this release. License status is recorded
in [LICENSE](LICENSE).
