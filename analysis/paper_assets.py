from __future__ import annotations
import argparse
import io
import json
import math
import os
import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
STUDY_NAMES = {12: 'wikitext_capacity', 14: 'qk_intervention', 15: 'head_factorial', 16: 'regime_dynamics', 17: 'wikitext_confirmation', 18: 'agnews_transfer', 19: 'agnews_capacity', 20: 'agnews_confirmation', 21: 'head_partition'}
SESSION_DIRS = {12: 'wikitext/capacity', 14: 'wikitext/qk_intervention', 15: 'wikitext/head_factorial', 16: 'wikitext/regime_dynamics', 17: 'wikitext/confirmation', 18: 'agnews/transfer', 19: 'agnews/capacity', 20: 'agnews/confirmation', 21: 'agnews/head_partition'}
FILE_ALIASES = {
    'confirmatory_POINT_SUMMARY.csv': 'POINT_SUMMARY.csv',
    'confirmatory_PAIRED_CONTRASTS.csv': 'PAIRED_CONTRASTS.csv',
    'confirmatory_FACTORIAL_EFFECTS.csv': 'FACTORIAL_EFFECTS.csv',
    'AGNEWS_CONFIRMATION_FROZEN_LOAD_SELECTION.json': 'FROZEN_LOAD_SELECTION.json',
}
for prefix in ('agnews_capacity_mapping_', 'AGNEWS_CONFIRMATION_', 'HEAD_PARTITION_',
               'confirmatory_replication_', 'external_corpus_replication_'):
    for suffix in ('POINT_SUMMARY', 'F50', 'F50_DIFFERENCE', 'PAIRED_EFFECTS_BY_F',
                   'PRIMARY_ENDPOINT', 'CONTINUOUS_OUTCOMES', 'CROSS_CORPUS_SYNTHESIS',
                   'CONTROL_SEQUENCE_SYNTHESIS', 'ALL_RUNS', 'HELDOUT_BINARY_ENDPOINT',
                   'CUMULATIVE_REGIME_ENTRY'):
        FILE_ALIASES[prefix + suffix + '.csv'] = suffix + '.csv'

def artifact_label(value, project_root):
    if value is None or str(value) == '':
        return ''
    parts = str(value).split('::', 1)
    p = Path(parts[0])
    try:
        label = p.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        label = 'external/' + p.name
    return label + ('::' + parts[1] if len(parts) == 2 else '')
RNG_SEED = 20260810
BOOTSTRAP_REPLICATES = 5000
NORMALIZED_BINDING_THRESHOLD = 5.0 / 12.0
ARCH_LABEL = {'H8_D32': '$H=8,\\ d_h=32$', 'H8_D64': '$H=8,\\ d_h=64$', 'H16_D32': '$H=16,\\ d_h=32$', 'H16_D64': '$H=16,\\ d_h=64$'}
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X']
LINESTYLES = ['-', '--', '-.', ':']
QK_INTERVENTION_EXPECTED_DIMS = {16, 32, 64, 128}
QK_INTERVENTION_EXPECTED_PAIRS = {(32, 16), (64, 16), (128, 16), (64, 32), (128, 32), (128, 64)}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--project-root', default=os.environ.get('HEAD_GEOMETRY_ROOT', 'results'),
                   help='Bundled result tables or a fresh experiment output root.')
    p.add_argument('--audit-manifest', default=None)
    p.add_argument('--output-dir', default=os.environ.get('HEAD_GEOMETRY_ASSETS', 'outputs/paper'))
    for i in [12, 14, 15, 16, 17, 18, 19, 20, 21]:
        p.add_argument(f'--{STUDY_NAMES[i].replace("_", "-")}-dir', default=None)
    p.add_argument('--strict-main', action='store_true', help='Fail if a main-paper source cannot be resolved from the locked session.')
    return p.parse_args()

def choose_output_dir(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir) if args.output_dir else Path('outputs/paper')
    for sub in ['main/figures', 'main/tables', 'appendix/figures', 'appendix/tables', 'audit']:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out

def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    aliases = {'f': 'F', 'f_star': 'F_star', 'f_star_ci_low': 'F_star_ci_low', 'f_star_ci_high': 'F_star_ci_high', 'dhead': 'd_head', 'head_dim': 'd_head', 'architecture': 'architecture_id', 'arch': 'architecture_id', 'maximum_validation_binding': 'maximum_logged_validation_binding', 'max_validation_binding': 'maximum_logged_validation_binding', 'maximum_validation_accuracy': 'maximum_logged_validation_accuracy', 'max_validation_accuracy': 'maximum_logged_validation_accuracy'}
    lower_map = {str(c).strip().lower(): c for c in d.columns}
    rename = {}
    for low, canonical in aliases.items():
        if low in lower_map and canonical not in d.columns:
            rename[lower_map[low]] = canonical
    d = d.rename(columns=rename)
    return d

def load_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    return canonicalize_columns(pd.read_csv(path))

def load_json(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def load_locked_sessions(args: argparse.Namespace) -> tuple[dict[int, Path], Optional[Path]]:
    project_root = Path(args.project_root)
    manifest_path = Path(args.audit_manifest) if args.audit_manifest else None
    locked = {}
    for study, relative in SESSION_DIRS.items():
        candidates = [project_root / relative, project_root / STUDY_NAMES[study]]
        available = [p for p in candidates if p.is_dir()]
        if len(available) > 1:
            raise ValueError(f'Ambiguous input directories for {STUDY_NAMES[study]}: {available}')
        if available:
            locked[study] = available[0]
    if manifest_path is not None and manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding='utf-8'))
        scripts = data.get('scripts', {})
        for s, rec in scripts.items():
            try:
                script = int(s)
            except Exception:
                continue
            p = str(rec.get('recommended_session_dir', '')).strip()
            if p:
                locked[script] = Path(p) if Path(p).is_absolute() else project_root / p
    for script in [12, 14, 15, 16, 17, 18, 19, 20, 21]:
        value = getattr(args, f'{STUDY_NAMES[script]}_dir')
        if value:
            locked[script] = Path(value)
    return (locked, manifest_path)

def validate_locked_dir(script: int, p: Optional[Path], required: bool) -> Optional[Path]:
    if p is None:
        if required:
            raise FileNotFoundError(f'No input directory resolved for {STUDY_NAMES[script]}.')
        return None
    if not p.exists():
        if required:
            raise FileNotFoundError(f'Input directory for {STUDY_NAMES[script]} does not exist: {p}')
        return None
    return p

def find_locked_file(session: Optional[Path], filename: str, *, required: bool=False) -> Optional[Path]:
    if session is None:
        if required:
            raise FileNotFoundError(filename)
        return None
    names = [filename]
    if filename in FILE_ALIASES:
        names.append(FILE_ALIASES[filename])
    for name in names:
        direct = session / name
        if direct.is_file():
            return direct
    candidates = [p for name in names for p in session.rglob(name) if p.is_file()]
    if candidates:
        candidates.sort(key=lambda p: (0 if '/final/' in str(p).replace('\\', '/').lower() else 1, len(p.parts), str(p)))
        return candidates[0]
    if required:
        raise FileNotFoundError(f'Required file {filename} not found under locked session {session}')
    return None

def _qk_intervention_zip_candidates(session: Path) -> list[Path]:
    zips = list(session.rglob('*.zip'))
    zips.extend(session.parent.glob('*qk_intervention*.zip'))
    zips.extend(session.parent.glob('*qk*RESULTS*.zip'))
    return sorted(set(zips))

def read_qk_intervention_artifact(session: Optional[Path], filename: str, *, required: bool=False) -> tuple[Optional[pd.DataFrame], str]:
    if session is None:
        if required:
            raise FileNotFoundError(f'Q/K intervention session unavailable for {filename}')
        return (None, '')
    direct = find_locked_file(session, filename, required=False)
    if direct is not None:
        return (canonicalize_columns(pd.read_csv(direct)), str(direct))
    for z in _qk_intervention_zip_candidates(session):
        try:
            with zipfile.ZipFile(z, 'r') as ar:
                matches = [member for member in ar.namelist() if Path(member).name.lower() == filename.lower()]
                if matches:
                    data = ar.read(matches[0])
                    df = pd.read_csv(io.BytesIO(data))
                    return (canonicalize_columns(df), f'{z}::{matches[0]}')
        except zipfile.BadZipFile:
            continue
    if required:
        raise FileNotFoundError(f'Q/K intervention artifact {filename} not found directly or inside ZIPs for session {session}')
    return (None, '')

def validate_qk_intervention_assets(point: Optional[pd.DataFrame], runs: Optional[pd.DataFrame], paired: Optional[pd.DataFrame]) -> None:
    if point is None or runs is None or paired is None:
        raise RuntimeError('Q/K intervention validation requires point summary, all-runs, and paired contrasts.')
    required_point = {'n_heads', 'value_head_dim', 'qk_max_dim', 'active_qk_dim', 'F', 'n_runs', 'p_success', 'parameter_count'}
    miss = required_point - set(point.columns)
    if miss:
        raise RuntimeError(f'Q/K intervention point summary missing {sorted(miss)}')
    dims = set(pd.to_numeric(point['active_qk_dim'], errors='coerce').dropna().astype(int))
    if dims != QK_INTERVENTION_EXPECTED_DIMS:
        raise RuntimeError(f'Q/K intervention active_qk_dim mismatch: {sorted(dims)}')
    if len(point) != 4:
        raise RuntimeError(f'Q/K intervention point summary has {len(point)} rows; expected 4.')
    if not (pd.to_numeric(point['n_runs'], errors='coerce') == 10).all():
        raise RuntimeError('Q/K intervention summary does not have 10 runs for every Q/K condition.')
    run_cols = {str(c).lower(): c for c in runs.columns}
    dim_col = run_cols.get('active_qk_dim')
    seed_col = run_cols.get('seed') or run_cols.get('random_seed') or run_cols.get('run_seed')
    if dim_col is None or seed_col is None:
        raise RuntimeError('Q/K intervention ALL_RUNS lacks active_qk_dim and/or seed.')
    x = runs[[dim_col, seed_col]].copy()
    x[dim_col] = pd.to_numeric(x[dim_col], errors='coerce')
    x[seed_col] = pd.to_numeric(x[seed_col], errors='coerce')
    x = x.dropna().drop_duplicates()
    x[dim_col] = x[dim_col].astype(int)
    x[seed_col] = x[seed_col].astype(int)
    counts = x.groupby(dim_col)[seed_col].nunique().to_dict()
    if set(counts) != QK_INTERVENTION_EXPECTED_DIMS or any((counts[d] != 10 for d in counts)):
        raise RuntimeError(f'Q/K intervention ALL_RUNS seed counts invalid: {counts}')
    if len(x) != 40:
        raise RuntimeError(f'Q/K intervention has {len(x)} unique Q/K×seed runs; expected 40.')
    needed = {'active_qk_high', 'active_qk_low', 'paired_seeds'}
    miss = needed - set(paired.columns)
    if miss:
        raise RuntimeError(f'Q/K intervention paired contrasts missing {sorted(miss)}')
    pairs = {(int(h), int(l)) for h, l in zip(paired['active_qk_high'], paired['active_qk_low'])}
    if pairs != QK_INTERVENTION_EXPECTED_PAIRS:
        raise RuntimeError(f'Q/K intervention contrast pairs mismatch: {sorted(pairs)}')
    if len(paired) != 6:
        raise RuntimeError('Q/K intervention must contain six pairwise contrasts.')
    if not (pd.to_numeric(paired['paired_seeds'], errors='coerce') == 10).all():
        raise RuntimeError('Q/K intervention paired contrasts do not all use 10 paired seeds.')

def save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> tuple[Path, Path]:
    pdf = out_dir / f'{stem}.pdf'
    png = out_dir / f'{stem}.png'
    fig.savefig(pdf, bbox_inches='tight', metadata={'Creator': None, 'Producer': None, 'CreationDate': None})
    fig.savefig(png, dpi=400, bbox_inches='tight', metadata={'Software': None})
    plt.close(fig)
    return (pdf, png)

def format_p(p: float) -> str:
    if not np.isfinite(p):
        return '--'
    if p < 0.0001:
        return f'{p:.2e}'
    return f'{p:.4f}'

def wilson_interval(k: int, n: int, z: float=1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    den = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / den
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)

def paired_bootstrap_rd(wide: np.ndarray, control: np.ndarray, rng: np.random.Generator, reps: int=BOOTSTRAP_REPLICATES) -> tuple[float, float, float]:
    wide = np.asarray(wide, dtype=float)
    control = np.asarray(control, dtype=float)
    diffs = wide - control
    n = len(diffs)
    idx = rng.integers(0, n, size=(reps, n))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return (float(diffs.mean()), float(lo), float(hi))

def fig_realtext_capacity_curves(point: pd.DataFrame, out: Path) -> list[Path]:
    required = {'d_head', 'F', 'p_success', 'n_success', 'n_runs'}
    missing = required - set(point.columns)
    if missing:
        raise ValueError(f'WikiText capacity mapping point summary missing: {sorted(missing)}')
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    for i, dhead in enumerate(sorted(point['d_head'].unique(), reverse=True)):
        sub = point[point['d_head'] == dhead].sort_values('F')
        x = sub['F'].to_numpy(float)
        y = sub['p_success'].to_numpy(float)
        lows, highs = ([], [])
        for _, r in sub.iterrows():
            lo, hi = wilson_interval(int(r['n_success']), int(r['n_runs']))
            lows.append(lo)
            highs.append(hi)
        lower = np.maximum(0.0, y - np.asarray(lows))
        upper = np.maximum(0.0, np.asarray(highs) - y)
        ax.errorbar(x, y, yerr=np.vstack([lower, upper]), marker=MARKERS[i % len(MARKERS)], linestyle=LINESTYLES[i % len(LINESTYLES)], capsize=2, label=f'$d_h={int(dhead)}$')
    ax.set_xlabel('Associative load $F$')
    ax.set_ylabel('Probability of successful binding')
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, 'main_fig1a_realtext_capacity_curves'))

def fig_realtext_capacity_boundary(capacity: pd.DataFrame, out: Path) -> list[Path]:
    required = {'d_head', 'F_star', 'F_star_ci_low', 'F_star_ci_high'}
    missing = required - set(capacity.columns)
    if missing:
        raise ValueError(f'WikiText capacity mapping capacity file missing: {sorted(missing)}')
    c = capacity.sort_values('d_head')
    x = c['d_head'].to_numpy(float)
    y = c['F_star'].to_numpy(float)
    lo = c['F_star_ci_low'].to_numpy(float)
    hi = c['F_star_ci_high'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.errorbar(x, y, yerr=np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)]), marker='o', linestyle='-', capsize=3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel('Head dimension $d_h$')
    ax.set_ylabel('Estimated capacity boundary $F_{50}$')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, 'main_fig1b_realtext_capacity_boundary'))

def fig_agnews_load_map(point: pd.DataFrame, selection: Optional[dict], out: Path) -> list[Path]:
    required = {'architecture_id', 'F', 'p_normalized_regime_success', 'success_ci_low', 'success_ci_high'}
    missing = required - set(point.columns)
    if missing:
        raise ValueError(f'AG News capacity mapping point summary missing: {sorted(missing)}')
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    architectures = [a for a in ['H8_D32', 'H8_D64'] if a in set(point['architecture_id'])]
    for i, arch in enumerate(architectures):
        sub = point[point['architecture_id'] == arch].sort_values('F')
        x = sub['F'].to_numpy(float)
        y = sub['p_normalized_regime_success'].to_numpy(float)
        lo = sub['success_ci_low'].to_numpy(float)
        hi = sub['success_ci_high'].to_numpy(float)
        ax.errorbar(x, y, yerr=np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)]), marker=MARKERS[i], linestyle=LINESTYLES[i], capsize=2, label=ARCH_LABEL.get(arch, arch))
    selected = selection.get('selected_F_for_agnews_confirmation') if selection else None
    if selected is not None:
        ax.axvline(float(selected), linestyle=':', linewidth=1.2)
        ax.text(float(selected) + 0.03, 0.5, f'frozen confirmation load $F={int(selected)}$', rotation=90, va='center', fontsize=8.5)
    ax.set_xlabel('AG News associative load $F$')
    ax.set_ylabel('Normalized-regime success probability')
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(sorted(point['F'].unique()))
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, 'main_fig2_agnews_load_normalization'))

def effect_rows(cross_corpus: pd.DataFrame, head_partition_primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if cross_corpus is not None and (not cross_corpus.empty):
        for _, r in cross_corpus.iterrows():
            corpus = str(r.get('corpus', ''))
            label = 'WikiText independent confirmation' if 'wiki' in corpus.lower() else 'AG News external confirmation'
            rows.append({'study': label, 'rd': float(r['paired_risk_difference']), 'lo': float(r['ci_low']), 'hi': float(r['ci_high']), 'p': float(r['exact_mcnemar_p_value'])})
    if head_partition_primary is not None and (not head_partition_primary.empty):
        r = head_partition_primary.iloc[0]
        rows.append({'study': 'AG News fixed-width partition control', 'rd': float(r['paired_risk_difference']), 'lo': float(r['bootstrap_ci_low']), 'hi': float(r['bootstrap_ci_high']), 'p': float(r['exact_mcnemar_p_value'])})
    return pd.DataFrame(rows)

def fig_primary_forest(effects: pd.DataFrame, out: Path) -> list[Path]:
    if effects.empty:
        raise ValueError('No effects available for primary forest plot.')
    e = effects.copy().reset_index(drop=True)
    y = np.arange(len(e))[::-1]
    fig, ax = plt.subplots(figsize=(6.7, 3.6))
    for yi, (_, r) in zip(y, e.iterrows()):
        ax.errorbar(r['rd'], yi, xerr=np.array([[r['rd'] - r['lo']], [r['hi'] - r['rd']]]), fmt='o', capsize=3)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(e['study'].tolist())
    ax.set_xlabel('Paired risk difference in regime entry')
    ax.set_xlim(min(-0.05, float(e['lo'].min()) - 0.03), max(0.6, float(e['hi'].max()) + 0.03))
    ax.grid(True, axis='x', alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, 'main_fig3_primary_effect_forest'))

def _extract_continuous_row(df: Optional[pd.DataFrame], outcome: str) -> Optional[pd.Series]:
    if df is None or df.empty or 'outcome' not in df.columns:
        return None
    sub = df[df['outcome'].astype(str) == outcome]
    return None if sub.empty else sub.iloc[0]

def _difference_column(row: pd.Series) -> Optional[str]:
    cols = [c for c in row.index if c.startswith('mean_difference_')]
    return cols[0] if cols else None

def continuous_rows(s17: Optional[pd.DataFrame], s20: Optional[pd.DataFrame], s21: Optional[pd.DataFrame], outcome: str) -> pd.DataFrame:
    rows = []
    for label, df in [('WikiText independent confirmation', s17), ('AG News external confirmation', s20), ('AG News fixed-width partition control', s21)]:
        row = _extract_continuous_row(df, outcome)
        if row is None:
            continue
        col = _difference_column(row)
        if col is None:
            continue
        rows.append({'study': label, 'delta': float(row[col]), 'lo': float(row['bootstrap_ci_low']), 'hi': float(row['bootstrap_ci_high']), 'p': float(row['exact_sign_flip_p_value'])})
    return pd.DataFrame(rows)

def fig_continuous_forest(rows: pd.DataFrame, out: Path, stem: str, xlabel: str) -> list[Path]:
    if rows.empty:
        return []
    r = rows.copy().reset_index(drop=True)
    y = np.arange(len(r))[::-1]
    fig, ax = plt.subplots(figsize=(6.7, 3.6))
    for yi, (_, row) in zip(y, r.iterrows()):
        ax.errorbar(row['delta'], yi, xerr=np.array([[row['delta'] - row['lo']], [row['hi'] - row['delta']]]), fmt='o', capsize=3)
    ax.axvline(0, linestyle='--', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(r['study'].tolist())
    ax.set_xlabel(xlabel)
    ax.grid(True, axis='x', alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, stem))

def threshold_sensitivity(all_runs: Optional[pd.DataFrame], wide_arch: str, control_arch: str, out: Path, table_out: Path, stem: str) -> list[Path]:
    if all_runs is None or all_runs.empty:
        return []
    required = {'architecture_id', 'seed', 'maximum_logged_validation_binding'}
    if not required.issubset(all_runs.columns):
        return []
    wide = all_runs[all_runs['architecture_id'] == wide_arch][['seed', 'maximum_logged_validation_binding']].rename(columns={'maximum_logged_validation_binding': 'wide'})
    ctrl = all_runs[all_runs['architecture_id'] == control_arch][['seed', 'maximum_logged_validation_binding']].rename(columns={'maximum_logged_validation_binding': 'control'})
    paired = wide.merge(ctrl, on='seed', validate='one_to_one').sort_values('seed')
    if paired.empty:
        return []
    rng = np.random.default_rng(RNG_SEED)
    thresholds = np.linspace(0.1, 0.8, 29)
    records = []
    for t in thresholds:
        w = (paired['wide'].to_numpy(float) >= t).astype(float)
        c = (paired['control'].to_numpy(float) >= t).astype(float)
        rd, lo, hi = paired_bootstrap_rd(w, c, rng)
        records.append((t, rd, lo, hi))
    frame = pd.DataFrame(records, columns=['threshold', 'rd', 'lo', 'hi'])
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.plot(frame['threshold'], frame['rd'], marker='o', markersize=3)
    ax.fill_between(frame['threshold'], frame['lo'], frame['hi'], alpha=0.18)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.axvline(NORMALIZED_BINDING_THRESHOLD, linestyle=':', linewidth=1)
    ax.set_xlabel('Normalized-binding threshold')
    ax.set_ylabel('Paired risk difference')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    files = list(save_fig(fig, out, stem))
    frame.to_csv(table_out / f'{stem}.csv', index=False)
    return files

def fig_qk_negative_control(point: Optional[pd.DataFrame], out: Path) -> list[Path]:
    if point is None or point.empty:
        return []
    required = {'active_qk_dim', 'F', 'p_success'}
    if not required.issubset(point.columns):
        return []
    p = point.sort_values('active_qk_dim')
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(p['active_qk_dim'].to_numpy(float), p['p_success'].to_numpy(float), marker='o', linestyle='-')
    ax.set_xscale('log', base=2)
    ax.set_xticks(sorted(p['active_qk_dim'].unique()))
    ax.set_xticklabels([str(int(v)) for v in sorted(p['active_qk_dim'].unique())])
    ax.set_xlabel('Active Q/K dimension $d_{QK}$')
    ax.set_ylabel('Success probability at $F=7$')
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, 'app_fig_qk_only_negative_control'))

def fig_factorial_summary(point: Optional[pd.DataFrame], out: Path) -> list[Path]:
    if point is None or point.empty:
        return []
    required = {'architecture_id', 'p_success'}
    if not required.issubset(point.columns):
        return []
    agg = point.groupby('architecture_id', as_index=False).agg(mean_p_success=('p_success', 'mean'))
    order = [a for a in ['H8_D32', 'H8_D64', 'H16_D32', 'H16_D64'] if a in set(agg['architecture_id'])]
    agg['architecture_id'] = pd.Categorical(agg['architecture_id'], order, ordered=True)
    agg = agg.sort_values('architecture_id')
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    x = np.arange(len(agg))
    ax.bar(x, agg['mean_p_success'].to_numpy(float))
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL.get(str(a), str(a)) for a in agg['architecture_id']])
    ax.set_ylabel('Mean success probability across tested loads')
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, 'app_fig_factorial_exploration_summary'))

def fig_raw_external_failure(primary18: Optional[pd.DataFrame], out: Path) -> list[Path]:
    if primary18 is None or primary18.empty:
        return []
    r = primary18.iloc[0]
    needed = {'p_success_H8_D64', 'p_success_H8_D32'}
    if not needed.issubset(r.index):
        return []
    vals = [float(r['p_success_H8_D32']), float(r['p_success_H8_D64'])]
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    x = np.arange(2)
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCH_LABEL['H8_D32'], ARCH_LABEL['H8_D64']])
    ax.set_ylabel('Primary regime-success probability')
    ax.set_ylim(0, max(0.12, max(vals) + 0.08))
    ax.set_title('Direct AG News transfer at raw $F=7$')
    ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, 'app_fig_agnews_transfer_raw_F7_floor'))

def fig_cumulative(cum: Optional[pd.DataFrame], out: Path, stem: str) -> list[Path]:
    if cum is None or cum.empty:
        return []
    required = {'architecture_id', 'step', 'cumulative_regime_entry_probability'}
    if not required.issubset(cum.columns):
        return []
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for i, (arch, sub) in enumerate(cum.groupby('architecture_id')):
        sub = sub.sort_values('step')
        ax.plot(sub['step'], sub['cumulative_regime_entry_probability'], marker=MARKERS[i % len(MARKERS)], markevery=max(1, len(sub) // 8), linestyle=LINESTYLES[i % len(LINESTYLES)], label=ARCH_LABEL.get(str(arch), str(arch)))
    ax.set_xlabel('Training step')
    ax.set_ylabel('Cumulative regime-entry probability')
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return list(save_fig(fig, out, stem))

def tex_escape(s: object) -> str:
    text = str(s)
    replacements = {'\\': '\\textbackslash{}', '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#', '_': '\\_', '{': '\\{', '}': '\\}'}
    for a, b in replacements.items():
        text = text.replace(a, b)
    return text

def dataframe_to_booktabs(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        latex = df.to_latex(index=False, escape=False)
    wrapped = f'\\begin{{table}}[t]\n\\centering\n\\small\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\resizebox{{\\linewidth}}{{!}}{{%\n' + latex + '}\n\\end{table}\n'
    path.write_text(wrapped, encoding='utf-8')

def write_main_design_table(path: Path) -> pd.DataFrame:
    df = pd.DataFrame([['Capacity map', 'WikiText-103', '$d_h\\in\\{16,32,64,128\\}$', 'multiple', '10/point', 'capacity boundary'], ['Independent confirmation', 'WikiText-103', 'H8_D64 vs H8_D32', '7', '40 pairs', 'confirmatory'], ['External load map', 'AG News', 'H8_D64 vs H8_D32', '3--7', '10 pairs/load', 'load selection'], ['External confirmation', 'AG News', 'H8_D64 vs H8_D32', '6', '120 pairs', 'confirmatory'], ['Fixed-width control', 'AG News', 'H8_D64 vs H16_D32', '6', '120 pairs', 'confirmatory control']], columns=['Study', 'Corpus', 'Comparison', '$F$', 'Seeds', 'Role'])
    dataframe_to_booktabs(df, path, caption='Core experimental sequence. The AG News confirmation load was selected from pooled difficulty rather than architecture separation.', label='tab:study_design')
    return df

def write_main_results_table(effects: pd.DataFrame, cont_acc: pd.DataFrame, path: Path) -> pd.DataFrame:
    acc_lookup = {}
    if cont_acc is not None and (not cont_acc.empty):
        for _, r in cont_acc.iterrows():
            acc_lookup[str(r['study'])] = (float(r['delta']), float(r['lo']), float(r['hi']))
    rows = []
    for _, r in effects.iterrows():
        study = str(r['study'])
        delta_text = '--'
        if study in acc_lookup:
            d, lo, hi = acc_lookup[study]
            delta_text = f'{d:.3f} [{lo:.3f}, {hi:.3f}]'
        rows.append([tex_escape(study), f"{float(r['rd']):.3f} [{float(r['lo']):.3f}, {float(r['hi']):.3f}]", format_p(float(r['p'])), delta_text])
    df = pd.DataFrame(rows, columns=['Study', 'Paired RD [95\\% CI]', 'McNemar $p$', '$\\Delta$ test accuracy [95\\% CI]'])
    dataframe_to_booktabs(df, path, caption='Independent confirmatory effects. Risk differences use the frozen binary regime endpoint; continuous accuracy differences are paired.', label='tab:confirmatory_results')
    return df

def save_raw_table(df: Optional[pd.DataFrame], out_dir: Path, stem: str) -> list[Path]:
    if df is None or df.empty:
        return []
    csv = out_dir / f'{stem}.csv'
    tex = out_dir / f'{stem}.tex'
    df.to_csv(csv, index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        text = df.to_latex(index=False, escape=True)
    tex.write_text(text, encoding='utf-8')
    return [csv, tex]

def write_placement(out: Path) -> None:
    text = '# ICLR 2027 paper asset placement\n\n## Main paper\n\n**Figure 1** — Real-text WikiText capacity map\n- `main_fig1a_realtext_capacity_curves.pdf`\n- `main_fig1b_realtext_capacity_boundary.pdf`\n\n**Figure 2** — AG News load normalization\n- `main_fig2_agnews_load_normalization.pdf`\n\n**Figure 3** — Independent confirmations + fixed-width control\n- `main_fig3_primary_effect_forest.pdf`\n\n**Figure 4** — Continuous test-accuracy effects\n- `main_fig4_continuous_test_accuracy_forest.pdf`\nUse in main paper if space permits; otherwise move to appendix.\n\n**Table 1** — Core experimental sequence\n- `table_main_1_study_design.tex`\n\n**Table 2** — Final confirmatory evidence\n- `table_main_2_confirmatory_results.tex`\n\n## Appendix\n\n- Q/K intervention Q/K-only negative intervention\n- head factorial factorial exploratory control\n- regime dynamics regime-analysis tables\n- AG News transfer raw AG News F=7 floor result\n- AG News capacity mapping complete mapping statistics\n- AG News confirmation threshold sensitivity + cumulative entry\n- head partition control threshold sensitivity + cumulative entry\n- all exact source tables\n\nThe main paper contains all evidence essential to the central claims.\nThe appendix contains negative controls, robustness, full statistics, and\nreproducibility details.\n'
    (out / 'PAPER_ASSET_PLACEMENT.md').write_text(text, encoding='utf-8')

def write_latex_snippets(out: Path) -> None:
    text = '% MAIN FIGURE 1\n\\begin{figure}[t]\n\\centering\n\\begin{minipage}[t]{0.57\\linewidth}\n  \\centering\n  \\includegraphics[width=\\linewidth]{main/figures/main_fig1a_realtext_capacity_curves.pdf}\n  \\textbf{(a)}\n\\end{minipage}\n\\hfill\n\\begin{minipage}[t]{0.40\\linewidth}\n  \\centering\n  \\includegraphics[width=\\linewidth]{main/figures/main_fig1b_realtext_capacity_boundary.pdf}\n  \\textbf{(b)}\n\\end{minipage}\n\\caption{Real-text associative capacity on WikiText-103 representations.\n(a) Regime-success probability across associative loads.\n(b) Estimated 50\\% capacity boundary with bootstrap confidence intervals.}\n\\label{fig:realtext_capacity}\n\\end{figure}\n\n% MAIN FIGURE 2\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.82\\linewidth]{main/figures/main_fig2_agnews_load_normalization.pdf}\n\\caption{AG News capacity map used to select the transition operating point\nbefore independent external confirmation. The frozen load-selection rule uses\npooled difficulty rather than architecture separation.}\n\\label{fig:agnews_loadmap}\n\\end{figure}\n\n% MAIN FIGURE 3\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.88\\linewidth]{main/figures/main_fig3_primary_effect_forest.pdf}\n\\caption{Paired risk differences for the independent WikiText confirmation,\nindependently selected AG News confirmation, and fixed-attention-width\nhead-partition control. Error bars are paired-bootstrap 95\\% confidence intervals.}\n\\label{fig:primary_forest}\n\\end{figure}\n\n% MAIN FIGURE 4 — optional main / otherwise appendix\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=0.88\\linewidth]{main/figures/main_fig4_continuous_test_accuracy_forest.pdf}\n\\caption{Paired continuous test-accuracy effects, showing that the findings\nare not artifacts of the frozen binary regime threshold.}\n\\label{fig:continuous_forest}\n\\end{figure}\n\n% TABLES\n% \\input{main/tables/table_main_1_study_design.tex}\n% \\input{main/tables/table_main_2_confirmatory_results.tex}\n'
    (out / 'LATEX_INCLUDE_SNIPPETS.tex').write_text(text, encoding='utf-8')

def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    out = choose_output_dir(args)
    main_fig = out / 'main/figures'
    main_tab = out / 'main/tables'
    app_fig = out / 'appendix/figures'
    app_tab = out / 'appendix/tables'
    audit_dir = out / 'audit'
    locked, manifest_path = load_locked_sessions(args)
    required_main_scripts = [12, 17, 19, 20, 21]
    for script in required_main_scripts:
        locked[script] = validate_locked_dir(script, locked.get(script), required=args.strict_main)
    if 14 in locked:
        locked[14] = validate_locked_dir(14, locked.get(14), required=False)
    files: dict[str, Optional[Path]] = {}
    s12 = locked.get(12)
    s14 = locked.get(14)
    s15 = locked.get(15)
    s16 = locked.get(16)
    s17 = locked.get(17)
    s18 = locked.get(18)
    s19 = locked.get(19)
    s20 = locked.get(20)
    s21 = locked.get(21)
    files['s12_point'] = find_locked_file(s12, 'confirmatory_POINT_SUMMARY.csv', required=args.strict_main)
    files['s12_capacity'] = find_locked_file(s12, 'confirmatory_CAPACITY_FSTAR.csv', required=args.strict_main)
    files['s19_point'] = find_locked_file(s19, 'agnews_capacity_mapping_POINT_SUMMARY.csv', required=args.strict_main)
    files['s19_selection'] = find_locked_file(s19, 'AGNEWS_CONFIRMATION_FROZEN_LOAD_SELECTION.json', required=args.strict_main)
    files['s20_primary'] = find_locked_file(s20, 'AGNEWS_CONFIRMATION_PRIMARY_ENDPOINT.csv', required=args.strict_main)
    files['s20_cont'] = find_locked_file(s20, 'AGNEWS_CONFIRMATION_CONTINUOUS_OUTCOMES.csv', required=args.strict_main)
    files['s20_cross'] = find_locked_file(s20, 'AGNEWS_CONFIRMATION_CROSS_CORPUS_SYNTHESIS.csv', required=args.strict_main)
    files['s20_runs'] = find_locked_file(s20, 'AGNEWS_CONFIRMATION_ALL_RUNS.csv', required=False)
    files['s20_heldout'] = find_locked_file(s20, 'AGNEWS_CONFIRMATION_HELDOUT_BINARY_ENDPOINT.csv', required=False)
    files['s20_cumulative'] = find_locked_file(s20, 'AGNEWS_CONFIRMATION_CUMULATIVE_REGIME_ENTRY.csv', required=False)
    files['s21_primary'] = find_locked_file(s21, 'HEAD_PARTITION_PRIMARY_ENDPOINT.csv', required=args.strict_main)
    files['s21_cont'] = find_locked_file(s21, 'HEAD_PARTITION_CONTINUOUS_OUTCOMES.csv', required=args.strict_main)
    files['s21_control'] = find_locked_file(s21, 'HEAD_PARTITION_CONTROL_SEQUENCE_SYNTHESIS.csv', required=False)
    files['s21_runs'] = find_locked_file(s21, 'HEAD_PARTITION_ALL_RUNS.csv', required=False)
    files['s21_heldout'] = find_locked_file(s21, 'HEAD_PARTITION_HELDOUT_BINARY_ENDPOINT.csv', required=False)
    files['s21_cumulative'] = find_locked_file(s21, 'HEAD_PARTITION_CUMULATIVE_REGIME_ENTRY.csv', required=False)
    if s15:
        files['s15_point'] = find_locked_file(s15, 'confirmatory_POINT_SUMMARY.csv')
        files['s15_paired'] = find_locked_file(s15, 'confirmatory_PAIRED_CONTRASTS.csv')
        files['s15_factorial'] = find_locked_file(s15, 'confirmatory_FACTORIAL_EFFECTS.csv')
    else:
        files['s15_point'] = files['s15_paired'] = files['s15_factorial'] = None
    if s16:
        files['s16_summary'] = find_locked_file(s16, 'architecture_transition_summary.csv')
        files['s16_numeric'] = find_locked_file(s16, 'paired_numeric_contrasts.csv')
        files['s16_binary'] = find_locked_file(s16, 'paired_binary_contrasts.csv')
    else:
        files['s16_summary'] = files['s16_numeric'] = files['s16_binary'] = None
    if s17:
        files['s17_primary'] = find_locked_file(s17, 'confirmatory_replication_PRIMARY_ENDPOINT.csv', required=args.strict_main)
        files['s17_cont'] = find_locked_file(s17, 'confirmatory_replication_CONTINUOUS_OUTCOMES.csv', required=args.strict_main)
    else:
        files['s17_primary'] = files['s17_cont'] = None
    if s18:
        files['s18_primary'] = find_locked_file(s18, 'external_corpus_replication_PRIMARY_ENDPOINT.csv')
        files['s18_cont'] = find_locked_file(s18, 'external_corpus_replication_CONTINUOUS_OUTCOMES.csv')
    else:
        files['s18_primary'] = files['s18_cont'] = None
    if s19:
        files['s19_f50'] = find_locked_file(s19, 'agnews_capacity_mapping_F50.csv')
        files['s19_f50diff'] = find_locked_file(s19, 'agnews_capacity_mapping_F50_DIFFERENCE.csv')
        files['s19_paired'] = find_locked_file(s19, 'agnews_capacity_mapping_PAIRED_EFFECTS_BY_F.csv')
    else:
        files['s19_f50'] = files['s19_f50diff'] = files['s19_paired'] = None
    s14_point = s14_runs = s14_paired = None
    s14_sources = {}
    if s14 is not None:
        s14_runs, src = read_qk_intervention_artifact(s14, 'confirmatory_ALL_RUNS.csv', required=False)
        s14_sources['s14_runs'] = src
        s14_point, src = read_qk_intervention_artifact(s14, 'confirmatory_POINT_SUMMARY.csv', required=False)
        s14_sources['s14_point'] = src
        s14_paired, src = read_qk_intervention_artifact(s14, 'confirmatory_PAIRED_QK_CONTRASTS.csv', required=False)
        s14_sources['s14_paired'] = src
        if any((x is not None for x in [s14_runs, s14_point, s14_paired])):
            validate_qk_intervention_assets(s14_point, s14_runs, s14_paired)
    result_provenance = project_root / 'provenance.json'
    source_kinds = {}
    if result_provenance.is_file():
        for record in json.loads(result_provenance.read_text(encoding='utf-8')).get('artifacts', []):
            relative = Path(record['file'])
            if relative.parts and relative.parts[0] == 'results':
                relative = Path(*relative.parts[1:])
            source_kinds[(project_root / relative).resolve()] = record['source_kind']
    source_rows = []
    for script, session in sorted(locked.items()):
        source_rows.append({'study': STUDY_NAMES[script], 'locked_session': artifact_label(session, project_root), 'source': 'explicit override' if getattr(args, f'{STUDY_NAMES[script]}_dir', None) else 'selected session'})
    for key, path in files.items():
        source_rows.append({'study': key, 'locked_session': '' if path is None else artifact_label(path.parent, project_root), 'source': '' if path is None else artifact_label(path, project_root), 'source_kind': '' if path is None else source_kinds.get(path.resolve(), 'experiment_output')})
    for key, source in s14_sources.items():
        source_rows.append({'study': key, 'locked_session': artifact_label(s14, project_root) if s14 else '', 'source': artifact_label(source, project_root)})
    source_audit = pd.DataFrame(source_rows)
    source_audit.to_csv(out / 'SOURCE_LOCK_AUDIT.csv', index=False)
    input_rows = []
    for key, path in files.items():
        input_rows.append({'key': key, 'found': path is not None, 'path': '' if path is None else artifact_label(path, project_root), 'source_kind': '' if path is None else source_kinds.get(path.resolve(), 'experiment_output')})
    for key, source in s14_sources.items():
        input_rows.append({'key': key, 'found': bool(source), 'path': artifact_label(source, project_root)})
    pd.DataFrame(input_rows).to_csv(audit_dir / 'input_file_audit.csv', index=False)
    s12_point = load_csv(files['s12_point'])
    s12_capacity = load_csv(files['s12_capacity'])
    s19_point = load_csv(files['s19_point'])
    s19_selection = load_json(files['s19_selection'])
    s20_primary = load_csv(files['s20_primary'])
    s20_cont = load_csv(files['s20_cont'])
    s20_cross = load_csv(files['s20_cross'])
    s21_primary = load_csv(files['s21_primary'])
    s21_cont = load_csv(files['s21_cont'])
    s17_cont = load_csv(files['s17_cont'])
    manifest = []

    def add(paths: Iterable[Path], placement: str, section: str, purpose: str) -> None:
        for p in paths:
            manifest.append({'file': str(p.relative_to(out)), 'placement': placement, 'section': section, 'purpose': purpose})
    if s12_point is not None:
        add(fig_realtext_capacity_curves(s12_point, main_fig), 'MAIN', 'Sec. 4', 'Real-text capacity curves across head dimensions')
    if s12_capacity is not None:
        add(fig_realtext_capacity_boundary(s12_capacity, main_fig), 'MAIN', 'Sec. 4', 'Estimated real-text capacity boundary versus head dimension')
    if s19_point is not None:
        add(fig_agnews_load_map(s19_point, s19_selection, main_fig), 'MAIN', 'Sec. 5', 'External-corpus load normalization and frozen F selection')
    effects = pd.DataFrame()
    if s20_cross is not None and s21_primary is not None:
        effects = effect_rows(s20_cross, s21_primary)
        add(fig_primary_forest(effects, main_fig), 'MAIN', 'Secs. 5-6', 'Independent confirmations and fixed-width causal-control effect')
    cont_acc = continuous_rows(s17_cont, s20_cont, s21_cont, 'test_accuracy')
    if not cont_acc.empty:
        add(fig_continuous_forest(cont_acc, main_fig, 'main_fig4_continuous_test_accuracy_forest', 'Paired difference in held-out test accuracy'), 'MAIN/OPTIONAL', 'Sec. 6', 'Continuous support showing the result is not threshold-only')
    cont_binding = continuous_rows(s17_cont, s20_cont, s21_cont, 'test_binding_score')
    if not cont_binding.empty:
        add(fig_continuous_forest(cont_binding, app_fig, 'app_fig_continuous_test_binding_forest', 'Paired difference in held-out normalized binding'), 'APPENDIX', 'App. H-I', 'Continuous normalized-binding effects')
    design = write_main_design_table(main_tab / 'table_main_1_study_design.tex')
    design.to_csv(main_tab / 'table_main_1_study_design.csv', index=False)
    add([main_tab / 'table_main_1_study_design.tex', main_tab / 'table_main_1_study_design.csv'], 'MAIN', 'Sec. 3', 'Core study sequence and confirmatory separation')
    if not effects.empty:
        results = write_main_results_table(effects, cont_acc, main_tab / 'table_main_2_confirmatory_results.tex')
        results.to_csv(main_tab / 'table_main_2_confirmatory_results.csv', index=False)
        add([main_tab / 'table_main_2_confirmatory_results.tex', main_tab / 'table_main_2_confirmatory_results.csv'], 'MAIN', 'Secs. 5-6', 'Final paired confirmatory results')
    s20_runs = load_csv(files['s20_runs'])
    s21_runs = load_csv(files['s21_runs'])
    add(threshold_sensitivity(s20_runs, 'H8_D64', 'H8_D32', app_fig, app_tab, 'app_fig_threshold_sensitivity_agnews_confirmation'), 'APPENDIX', 'App. H', 'Threshold sensitivity for external confirmation')
    add(threshold_sensitivity(s21_runs, 'H8_D64', 'H16_D32', app_fig, app_tab, 'app_fig_threshold_sensitivity_head_partition'), 'APPENDIX', 'App. I', 'Threshold sensitivity for fixed-width control')
    add(fig_qk_negative_control(s14_point, app_fig), 'APPENDIX', 'App. D', 'Q/K-only negative intervention')
    add(fig_factorial_summary(load_csv(files['s15_point']), app_fig), 'APPENDIX', 'App. E', 'Parameter-matched factorial exploratory evidence')
    add(fig_raw_external_failure(load_csv(files['s18_primary']), app_fig), 'APPENDIX', 'App. G', 'Direct raw-load AG News floor effect')
    add(fig_cumulative(load_csv(files['s20_cumulative']), app_fig, 'app_fig_agnews_confirmation_cumulative_regime_entry'), 'APPENDIX', 'App. H', 'Training-time regime-entry dynamics')
    add(fig_cumulative(load_csv(files['s21_cumulative']), app_fig, 'app_fig_head_partition_cumulative_regime_entry'), 'APPENDIX', 'App. I', 'Training-time regime-entry dynamics under fixed-width control')
    appendix_tables = [('s12_capacity', 'app_table_realtext_capacity_F50'), ('s15_paired', 'app_table_factorial_paired_contrasts'), ('s15_factorial', 'app_table_factorial_effects'), ('s16_summary', 'app_table_regime_architecture_summary'), ('s16_numeric', 'app_table_regime_numeric_contrasts'), ('s16_binary', 'app_table_regime_binary_contrasts'), ('s17_primary', 'app_table_wikitext_confirmation_primary'), ('s17_cont', 'app_table_wikitext_confirmation_continuous'), ('s18_primary', 'app_table_agnews_rawF7_primary'), ('s18_cont', 'app_table_agnews_rawF7_continuous'), ('s19_point', 'app_table_agnews_load_map'), ('s19_f50', 'app_table_agnews_F50'), ('s19_f50diff', 'app_table_agnews_F50_difference'), ('s19_paired', 'app_table_agnews_paired_effects_by_F'), ('s20_primary', 'app_table_agnews_confirmation_primary'), ('s20_heldout', 'app_table_agnews_confirmation_heldout'), ('s20_cont', 'app_table_agnews_confirmation_continuous'), ('s21_primary', 'app_table_fixed_width_primary'), ('s21_heldout', 'app_table_fixed_width_heldout'), ('s21_cont', 'app_table_fixed_width_continuous')]
    for key, stem in appendix_tables:
        paths = save_raw_table(load_csv(files.get(key)), app_tab, stem)
        add(paths, 'APPENDIX', 'Appendix', f'Source table: {stem}')
    add(save_raw_table(s14_point, app_tab, 'app_table_qk_point_summary'), 'APPENDIX', 'App. D', 'Q/K intervention point summary')
    add(save_raw_table(s14_paired, app_tab, 'app_table_qk_paired_contrasts'), 'APPENDIX', 'App. D', 'Q/K intervention paired contrasts')
    if s14_runs is not None:
        s14_runs.to_csv(app_tab / 'app_table_qk_all_runs.csv', index=False)
        add([app_tab / 'app_table_qk_all_runs.csv'], 'APPENDIX/SUPPLEMENT', 'App. D', 'Complete Q/K intervention run-level audit table')
    if files['s19_selection'] is not None:
        dest = app_tab / 'AGNEWS_CONFIRMATION_FROZEN_LOAD_SELECTION.json'
        shutil.copy2(files['s19_selection'], dest)
        add([dest], 'APPENDIX', 'App. G', 'Frozen external-confirmation load-selection rule')
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(out / 'PAPER_ASSET_MANIFEST.csv', index=False)
    write_placement(out)
    write_latex_snippets(out)
    provenance = {'project_root': '.', 'audit_manifest': '' if manifest_path is None else artifact_label(manifest_path, project_root), 'locked_sessions': {STUDY_NAMES[k]: artifact_label(v, project_root) for k, v in sorted(locked.items())}, 'qk_intervention_validated_in_asset_generator': bool(s14_point is not None and s14_runs is not None and (s14_paired is not None)), 'normalized_binding_threshold': NORMALIZED_BINDING_THRESHOLD, 'bootstrap_replicates_for_appendix_threshold_sensitivity': BOOTSTRAP_REPLICATES, 'no_model_training': True}
    (out / 'ASSET_GENERATION_PROVENANCE.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(f'Generated {len(manifest_df)} assets.')
    if files['s12_point'] is None:
        print('WikiText capacity point-summary data are absent; the capacity-curves figure was not generated.')
if __name__ == '__main__':
    main()
