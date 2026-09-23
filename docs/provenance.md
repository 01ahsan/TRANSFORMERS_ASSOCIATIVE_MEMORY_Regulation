# Source provenance

The repository contains eight experiment programs, the regime-dynamics
analysis, and the supplied paper-asset generator. Each program was extracted
from the selected source cell or file; duplicate implementations and notebook
setup, launch, debug, and output cells were discarded.

[provenance.json](provenance.json) records the original source hashes and the
current program hashes. Source filenames are omitted because they belonged to
the local notebook workflow. Cell numbers include markdown cells.

Paths, command-line options, output names, and source-lock metadata use the
study names in this repository. Computational definitions, seeds, training
budgets, architecture settings, and statistical procedures are unchanged.
Use one output root throughout an experiment sequence so downstream stages
read the matching caches and frozen-design records.

Dataset files, checkpoints, and embedding caches are not distributed. Compact
result tables are included in `results/`; new run outputs and generated assets
are written beneath `outputs/` or a configured path.

## Result records

[Result provenance](../results/provenance.json) records source and released
hashes for all 29 CSV/JSON files. Numerical values are preserved; identifying
paths and local workflow labels are sanitized. The AG News capacity,
confirmation, and head-partition records come from their run archives.
The WikiText capacity boundary and confirmation tables come from previously
exported paper tables, not recovered per-seed run records.

The WikiText capacity point-summary CSV was not located. Its curve is not
reconstructed from the boundary estimates. The archive originally labeled as
the WikiText confirmation instead contained duplicate AG News transfer
records; those records are not presented as WikiText evidence.

## Source integration status

The release-correction notes report recovery of the synthetic discovery
notebook, final ten-seed head-partition top-up, and final integrity auditor.
Those source files were not included with the notes and have not been located
in the available project files. Their integration remains pending. This is a
statement about this checkout, not a claim that the reported recovered sources
do not exist elsewhere. No replacement historical programs have been written.

### Historical optimization diagnostic

The original source for the final 20-seed continuation of the
`d_model=64`, `F=17`, batch-size-512 diagnostic was not recovered. The recovery
notes describe an earlier version with the same configuration and fewer
seeds; that notebook is not present in this checkout. The reported `3/20`
aggregate is documented only as historical discovery evidence and is **not
used in any main confirmatory result**.

Validation covered Python syntax, every command's `--help` interface, scientific
function comparisons, and the renamed input/output references. The
regime-dynamics analysis was rerun on the available factorial records; all nine
numeric tables matched the prior outputs. Training was not rerun.
The bundled summaries generate 67 paper assets. The F=6 run records confirm
25,447,936 trainable parameters for both confirmatory comparisons. The missing
WikiText point-summary input is reported by default and rejected in strict
asset-generation mode.
