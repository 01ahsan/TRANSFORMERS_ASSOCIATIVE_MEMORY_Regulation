# Paper inputs

These compact tables support paper-asset generation without GPU training:

```sh
python analysis/paper_assets.py --project-root results
```

The confirmation and capacity directories contain the final endpoints,
continuous outcomes, load-selection records, and compact run tables where
available. Additional factorial, regime-dynamics, and transfer summaries
support the appendix assets.

`wikitext_capacity/confirmatory_CAPACITY_FSTAR.csv` and the two
`wikitext_confirmation/` tables are preserved paper exports. The other tables
come from run archives. [provenance.json](provenance.json) distinguishes these
sources and records original and released SHA256 hashes. Metadata sanitization
does not change numerical observations.

The WikiText capacity point-summary CSV is absent. Its curve cannot be
regenerated from this bundle. No synthetic result files have been reconstructed.
These tables are analysis inputs, not caches or checkpoints for resuming runs.
