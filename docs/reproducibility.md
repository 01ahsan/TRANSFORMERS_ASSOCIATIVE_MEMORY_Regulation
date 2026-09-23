# Reproducibility

Run commands from the repository root. `HEAD_GEOMETRY_ROOT` selects the run
directory and defaults to `./outputs`. For example, use
`export HEAD_GEOMETRY_ROOT=./runs/replication` in a POSIX shell or
`$env:HEAD_GEOMETRY_ROOT = './runs/replication'` in PowerShell. Each study has
its own subdirectory and an optional environment override:

| Program | Directory under the run root | Override |
| --- | --- | --- |
| `experiments/wikitext_capacity.py` | `wikitext/capacity` | `WIKITEXT_CAPACITY_DIR` |
| `experiments/qk_intervention.py` | `wikitext/qk_intervention` | `QK_INTERVENTION_DIR` |
| `experiments/head_factorial.py` | `wikitext/head_factorial` | `HEAD_FACTORIAL_DIR` |
| `analysis/regime_dynamics.py` | `wikitext/regime_dynamics` | `REGIME_DYNAMICS_DIR` |
| `experiments/wikitext_confirmation.py` | `wikitext/confirmation` | `WIKITEXT_CONFIRMATION_DIR` |
| `experiments/agnews_transfer.py` | `agnews/transfer` | `AGNEWS_TRANSFER_DIR` |
| `experiments/agnews_capacity.py` | `agnews/capacity` | `AGNEWS_CAPACITY_DIR` |
| `experiments/agnews_confirmation.py` | `agnews/confirmation` | `AGNEWS_CONFIRMATION_DIR` |
| `experiments/head_partition.py` | `agnews/head_partition` | `HEAD_PARTITION_DIR` |

The WikiText capacity study prepares representations that the Q/K
intervention, head factorial, and confirmation studies can reuse. Regime
analysis requires the factorial run table and learning curves:

```sh
python analysis/regime_dynamics.py --input-dir outputs/wikitext/head_factorial --output-dir outputs/wikitext/regime_dynamics
```

Run the AG News stages in this order: transfer, capacity, confirmation, head
partition. Capacity mapping reuses compatible transfer data and fixes the
confirmatory load. Confirmation checks the capacity study's design records;
the head-partition control checks records from the preceding AG News stages.
Keep these stages under one run root or configure their source directories
consistently. The code checks cache compatibility and source records before
reusing them.

All retained training designs use 20,000 steps with batch size 256. WikiText
capacity requires two visible CUDA GPUs. The other training coordinators
support one or two GPUs. Full grids are substantial training jobs; `--help`
displays options without starting training.

Generate paper tables and plots from the bundled compact results:

```sh
python analysis/paper_assets.py
```

The default input is `results/`, unless `HEAD_GEOMETRY_ROOT` is set. To select
the bundled results explicitly, pass `--project-root results`. For new runs:

```sh
python analysis/paper_assets.py --project-root outputs --output-dir outputs/paper
```

Use `--help` for per-study input overrides. `--strict-main` requires the
source records needed for the main-paper assets and fails when they are
missing. The bundled inputs lack the WikiText capacity point-summary CSV, so
strict mode fails on that missing file. Default mode generates the supported
67 assets and reports the skipped capacity-curves figure. It does not infer
missing point-level results from fitted boundaries.

The AG News F=6 confirmation and fixed-width head-partition control each use
25,447,936 trainable parameters. The learned positional table has 13 rows;
the capacity-mapping model covering F up to 7 has two additional rows. The
output classifier shares the value-embedding weights and is counted once.

The source was checked under Python 3.11. `requirements.txt` lists the
dependencies; the historical package lockfile and dataset/model revision
pins were unavailable. Reproduction therefore depends on the installed
training stack and retrieved input revisions. Record these versions with new
runs, and retain the supplied seeds, architecture definitions, thresholds,
episode construction, and statistical procedures.

Full GPU training was not rerun during repository preparation.
[Provenance](provenance.md) records which original sources are integrated,
which recovered sources still need to be supplied, and the historical
optimization diagnostic that cannot be reproduced from the retained files.
