"""Success regime and optimization dynamics analysis."""

import argparse
import json
import math
import os
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

ARCHS = ["H8_D32", "H8_D64", "H16_D32", "H16_D64"]
BASELINE = 1 / 7
THRESHOLD_30 = 0.30
THRESHOLD_50 = 0.50
SUCCESS_80 = 0.80
EARLY_END = 5000
MID_END = 10000
LATE_START = 15000
BOOTSTRAPS = 20000
RNG_SEED = 20260720

OUTPUT_ROOT = Path(os.environ.get("HEAD_GEOMETRY_ROOT", "./outputs"))
DEFAULT_OUTPUT = Path(os.environ.get("REGIME_DYNAMICS_DIR", str(OUTPUT_ROOT / "wikitext" / "regime_dynamics")))
DIAGNOSTICS = [
    "training_loss",
    "validation_accuracy",
    "validation_binding_score",
    "gradient_norm_before_clipping",
    "attention_entropy",
    "attention_logit_std",
    "attention_max_probability",
    "q_rms",
    "k_rms",
    "v_rms",
]


def locate_head_factorial(explicit=None):
    default = OUTPUT_ROOT / "wikitext" / "head_factorial"
    source = Path(explicit or os.environ.get("HEAD_FACTORIAL_DIR", str(default)))
    if (
        (source / "final" / "confirmatory_ALL_RUNS.csv").is_file()
        and (source / "results" / "learning_curves").is_dir()
    ):
        return source
    raise FileNotFoundError("Completed head factorial outputs are required; pass --input-dir.")


def exact_sign_flip(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    observed = abs(values.mean())
    n = len(values)
    if n <= 20:
        count = 0
        total = 2 ** n
        for mask in range(total):
            signs = np.array(
                [1 if (mask >> bit) & 1 else -1 for bit in range(n)]
            )
            if abs(np.mean(values * signs)) >= observed - 1e-15:
                count += 1
        return count / total
    rng = np.random.default_rng(RNG_SEED)
    signs = rng.choice([-1.0, 1.0], size=(100000, n))
    return float(np.mean(np.abs((signs * values).mean(1)) >= observed))


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[idx].mean(1)
    return tuple(np.quantile(means, [0.025, 0.975]).astype(float))


def first_crossing(steps, values, threshold):
    idx = np.flatnonzero(values >= threshold)
    return float(steps[idx[0]]) if len(idx) else np.nan


def sustained_crossing(steps, values, threshold, consecutive=2):
    above = values >= threshold
    for i in range(len(above) - consecutive + 1):
        if np.all(above[i : i + consecutive]):
            return float(steps[i])
    return np.nan


def last_at_or_before(frame, column, step):
    x = frame.loc[frame.step <= step, ["step", column]].dropna()
    return float(x.iloc[-1][column]) if len(x) else np.nan


def mean_window(frame, column, lo=None, hi=None):
    x = frame
    if lo is not None:
        x = x[x.step >= lo]
    if hi is not None:
        x = x[x.step <= hi]
    x = x[column].dropna()
    return float(x.mean()) if len(x) else np.nan


def slope_window(frame, column, lo=None, hi=None):
    x = frame
    if lo is not None:
        x = x[x.step >= lo]
    if hi is not None:
        x = x[x.step <= hi]
    x = x[["step", column]].dropna()
    if len(x) < 2:
        return np.nan
    return float(np.polyfit(x.step, x[column], 1)[0] * 1000)


def normalized_auc(steps, values):
    if len(steps) < 2 or steps[-1] <= steps[0]:
        return np.nan
    return float(np.trapezoid(values, steps) / (steps[-1] - steps[0]))


def load_data(head_factorial):
    final_path = head_factorial / "final" / "confirmatory_ALL_RUNS.csv"
    runs = pd.read_csv(final_path)
    required = {
        "architecture_id",
        "seed",
        "test_accuracy",
        "validation_accuracy",
        "success",
    }
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Missing final-run columns: {sorted(missing)}")
    runs["seed"] = runs.seed.astype(int)

    files = sorted((head_factorial / "results" / "learning_curves").glob("*.csv"))
    if not files:
        raise FileNotFoundError("No learning-curve CSV files found.")

    parts = []
    for f in files:
        x = pd.read_csv(f)
        needed = {"architecture_id", "seed", "step", "validation_accuracy"}
        if needed <= set(x.columns):
            x["source_file"] = f.name
            parts.append(x)
        else:
            warnings.warn(f"Skipping invalid learning curve: {f.name}")

    if not parts:
        raise RuntimeError("No valid learning curves loaded.")

    curves = pd.concat(parts, ignore_index=True)
    curves["seed"] = curves.seed.astype(int)
    curves["step"] = curves.step.astype(int)
    curves = curves.sort_values(["architecture_id", "seed", "step"])

    dup = curves.duplicated(["architecture_id", "seed", "step"])
    if dup.any():
        raise RuntimeError("Duplicate architecture/seed/step curve rows found.")

    return runs, curves


def integrity_table(runs, curves):
    rows = []
    for arch in sorted(set(runs.architecture_id) | set(curves.architecture_id)):
        a = sorted(runs.loc[runs.architecture_id == arch, "seed"].unique())
        b = sorted(curves.loc[curves.architecture_id == arch, "seed"].unique())
        rows.append(
            {
                "architecture_id": arch,
                "n_final_runs": len(a),
                "n_curve_runs": len(b),
                "final_seeds": ",".join(map(str, a)),
                "curve_seeds": ",".join(map(str, b)),
                "missing_curve_seeds": ",".join(map(str, sorted(set(a) - set(b)))),
                "orphan_curve_seeds": ",".join(map(str, sorted(set(b) - set(a)))),
            }
        )
    return pd.DataFrame(rows)


def run_metrics(runs, curves):
    lookup = runs.set_index(["architecture_id", "seed"])
    rows = []

    for (arch, seed), frame in curves.groupby(["architecture_id", "seed"]):
        key = (arch, int(seed))
        if key not in lookup.index:
            continue
        final = lookup.loc[key]
        if isinstance(final, pd.DataFrame):
            final = final.iloc[0]

        frame = frame.sort_values("step").reset_index(drop=True)
        steps = frame.step.to_numpy(float)
        val = frame.validation_accuracy.to_numpy(float)
        imax = int(np.nanargmax(val))
        s30 = first_crossing(steps, val, THRESHOLD_30)
        s50 = first_crossing(steps, val, THRESHOLD_50)
        final_curve = float(val[-1])

        row = {
            "architecture_id": arch,
            "seed": int(seed),
            "n_curve_points": len(frame),
            "first_logged_step": int(steps[0]),
            "last_logged_step": int(steps[-1]),
            "final_test_accuracy": float(final.test_accuracy),
            "final_validation_accuracy": float(final.validation_accuracy),
            "final_curve_validation_accuracy": final_curve,
            "entered_regime_030": bool(np.isfinite(s30)),
            "entered_regime_050": bool(np.isfinite(s50)),
            "final_success_threshold_080": bool(float(final.test_accuracy) >= SUCCESS_80),
            "first_crossing_step_030": s30,
            "first_crossing_step_050": s50,
            "sustained_crossing_step_030": sustained_crossing(
                steps, val, THRESHOLD_30
            ),
            "sustained_crossing_step_050": sustained_crossing(
                steps, val, THRESHOLD_50
            ),
            "maximum_validation_accuracy": float(val[imax]),
            "step_of_maximum_validation_accuracy": float(steps[imax]),
            "normalized_validation_auc": normalized_auc(steps, val),
            "accuracy_at_5000": last_at_or_before(
                frame, "validation_accuracy", EARLY_END
            ),
            "accuracy_at_10000": last_at_or_before(
                frame, "validation_accuracy", MID_END
            ),
            "early_accuracy_slope_per_1000_steps": slope_window(
                frame, "validation_accuracy", hi=EARLY_END
            ),
            "mid_accuracy_slope_per_1000_steps": slope_window(
                frame, "validation_accuracy", lo=EARLY_END, hi=MID_END
            ),
            "late_accuracy_slope_per_1000_steps": slope_window(
                frame, "validation_accuracy", lo=LATE_START
            ),
            "late_gain_15000_to_final": (
                final_curve
                - last_at_or_before(frame, "validation_accuracy", LATE_START)
            ),
            "peak_minus_final_accuracy": float(val[imax] - final_curve),
        }

        for col in DIAGNOSTICS:
            if col not in frame.columns:
                continue
            row[f"early_mean_{col}"] = mean_window(frame, col, hi=EARLY_END)
            row[f"mid_mean_{col}"] = mean_window(
                frame, col, lo=EARLY_END, hi=MID_END
            )
            row[f"late_mean_{col}"] = mean_window(frame, col, lo=LATE_START)
            row[f"early_slope_{col}_per_1000_steps"] = slope_window(
                frame, col, hi=EARLY_END
            )

        for col in ["n_heads", "head_dim", "inner_dim", "d_ff", "parameter_count"]:
            if col in final.index:
                row[col] = final[col]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["architecture_id", "seed"])


def architecture_summary(metrics):
    out = (
        metrics.groupby("architecture_id")
        .agg(
            n_runs=("seed", "count"),
            mean_final_test_accuracy=("final_test_accuracy", "mean"),
            median_final_test_accuracy=("final_test_accuracy", "median"),
            std_final_test_accuracy=("final_test_accuracy", "std"),
            min_final_test_accuracy=("final_test_accuracy", "min"),
            max_final_test_accuracy=("final_test_accuracy", "max"),
            p_entered_regime_030=("entered_regime_030", "mean"),
            p_entered_regime_050=("entered_regime_050", "mean"),
            p_success_080=("final_success_threshold_080", "mean"),
            median_transition_step_030=("first_crossing_step_030", "median"),
            median_transition_step_050=("first_crossing_step_050", "median"),
            mean_max_validation_accuracy=("maximum_validation_accuracy", "mean"),
            mean_validation_auc=("normalized_validation_auc", "mean"),
            mean_accuracy_at_5000=("accuracy_at_5000", "mean"),
            mean_accuracy_at_10000=("accuracy_at_10000", "mean"),
            mean_late_gain=("late_gain_15000_to_final", "mean"),
        )
        .reset_index()
    )
    order = {a: i for i, a in enumerate(ARCHS)}
    return out.sort_values("architecture_id", key=lambda s: s.map(order))


def paired_numeric(metrics):
    rng = np.random.default_rng(RNG_SEED)
    specs = [
        ("H8_D64", "H8_D32", "width_effect_at_8_heads"),
        ("H16_D64", "H16_D32", "width_effect_at_16_heads"),
        ("H16_D32", "H8_D32", "head_effect_at_width_32"),
        ("H16_D64", "H8_D64", "head_effect_at_width_64"),
    ]
    outcomes = [
        "final_test_accuracy",
        "maximum_validation_accuracy",
        "normalized_validation_auc",
        "accuracy_at_5000",
        "accuracy_at_10000",
        "late_gain_15000_to_final",
    ]
    rows = []
    for a, b, label in specs:
        for outcome in outcomes:
            xa = metrics[metrics.architecture_id == a][["seed", outcome]].rename(
                columns={outcome: "a"}
            )
            xb = metrics[metrics.architecture_id == b][["seed", outcome]].rename(
                columns={outcome: "b"}
            )
            p = xa.merge(xb, on="seed").dropna()
            d = p.a.to_numpy(float) - p.b.to_numpy(float)
            lo, hi = bootstrap_ci(d, rng)
            rows.append(
                {
                    "contrast": label,
                    "outcome": outcome,
                    "architecture_a": a,
                    "architecture_b": b,
                    "paired_seeds": len(d),
                    "mean_difference_a_minus_b": np.mean(d) if len(d) else np.nan,
                    "median_difference_a_minus_b": np.median(d) if len(d) else np.nan,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "exact_sign_flip_p_value": exact_sign_flip(d),
                    "wins_a": int(np.sum(d > 0)),
                    "ties": int(np.sum(d == 0)),
                    "losses_a": int(np.sum(d < 0)),
                }
            )
    return pd.DataFrame(rows)


def paired_binary(metrics):
    specs = [
        ("H8_D64", "H8_D32", "width_effect_at_8_heads"),
        ("H16_D64", "H16_D32", "width_effect_at_16_heads"),
        ("H16_D32", "H8_D32", "head_effect_at_width_32"),
        ("H16_D64", "H8_D64", "head_effect_at_width_64"),
    ]
    outcomes = [
        "entered_regime_030",
        "entered_regime_050",
        "final_success_threshold_080",
    ]
    rows = []
    for a, b, label in specs:
        for outcome in outcomes:
            xa = metrics[metrics.architecture_id == a][["seed", outcome]].rename(
                columns={outcome: "a"}
            )
            xb = metrics[metrics.architecture_id == b][["seed", outcome]].rename(
                columns={outcome: "b"}
            )
            p = xa.merge(xb, on="seed")
            av = p.a.astype(bool).to_numpy()
            bv = p.b.astype(bool).to_numpy()
            a_only = int(np.sum(av & ~bv))
            b_only = int(np.sum(~av & bv))
            discordant = a_only + b_only
            mcnemar = (
                stats.binomtest(
                    min(a_only, b_only), discordant, 0.5, alternative="two-sided"
                ).pvalue
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "contrast": label,
                    "outcome": outcome,
                    "architecture_a": a,
                    "architecture_b": b,
                    "paired_seeds": len(p),
                    "p_a": av.mean(),
                    "p_b": bv.mean(),
                    "risk_difference_a_minus_b": av.mean() - bv.mean(),
                    "a_only_success": a_only,
                    "b_only_success": b_only,
                    "both_success": int(np.sum(av & bv)),
                    "neither_success": int(np.sum(~av & ~bv)),
                    "exact_mcnemar_p_value": float(mcnemar),
                }
            )
    return pd.DataFrame(rows)


def success_failure_table(metrics, arch="H8_D64"):
    x = metrics[metrics.architecture_id == arch].copy()
    x["group"] = np.where(
        x.entered_regime_050,
        "entered_0.50_regime",
        "never_entered_0.50_regime",
    )
    cols = [
        "final_test_accuracy",
        "maximum_validation_accuracy",
        "normalized_validation_auc",
        "accuracy_at_5000",
        "accuracy_at_10000",
        "early_accuracy_slope_per_1000_steps",
        "mid_accuracy_slope_per_1000_steps",
        "late_gain_15000_to_final",
    ] + [
        c
        for c in x.columns
        if c.startswith("early_mean_") or c.startswith("early_slope_")
    ]
    rows = []
    for col in cols:
        s = x.loc[x.group == "entered_0.50_regime", col].dropna().to_numpy(float)
        f = x.loc[
            x.group == "never_entered_0.50_regime", col
        ].dropna().to_numpy(float)
        if not len(s) or not len(f):
            continue
        p = (
            stats.mannwhitneyu(s, f, alternative="two-sided").pvalue
            if len(s) >= 2 and len(f) >= 2
            else np.nan
        )
        pooled = np.r_[s, f]
        sd = np.std(pooled, ddof=1) if len(pooled) > 1 else np.nan
        rows.append(
            {
                "architecture_id": arch,
                "metric": col,
                "n_success_group": len(s),
                "n_failure_group": len(f),
                "success_group_mean": s.mean(),
                "failure_group_mean": f.mean(),
                "mean_difference_success_minus_failure": s.mean() - f.mean(),
                "standardized_mean_difference": (
                    (s.mean() - f.mean()) / sd if np.isfinite(sd) and sd > 0 else np.nan
                ),
                "mann_whitney_p_value": p,
            }
        )
    return pd.DataFrame(rows)


def trajectory_summary(curves, metrics):
    labels = metrics[
        ["architecture_id", "seed", "entered_regime_050"]
    ]
    x = curves.merge(labels, on=["architecture_id", "seed"])
    x["regime_group"] = np.where(
        x.entered_regime_050,
        "entered_0.50_regime",
        "never_entered_0.50_regime",
    )
    out = (
        x.groupby(["architecture_id", "regime_group", "step"])
        .validation_accuracy.agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci95_low"] = out["mean"] - 1.96 * out["sem"]
    out["ci95_high"] = out["mean"] + 1.96 * out["sem"]
    return out


def mixture_table(metrics):
    rows = []
    for arch, g in metrics.groupby("architecture_id"):
        values = g.final_test_accuracy.dropna().to_numpy(float).reshape(-1, 1)
        if len(values) < 4:
            continue
        models = {}
        for k in [1, 2]:
            model = GaussianMixture(
                k,
                covariance_type="full",
                n_init=50,
                random_state=RNG_SEED,
                reg_covar=1e-6,
            ).fit(values)
            models[k] = model
        m2 = models[2]
        labels = m2.predict(values)
        means = m2.means_.ravel()
        weights = m2.weights_.ravel()
        sds = np.sqrt(m2.covariances_.reshape(-1))
        order = np.argsort(means)
        silhouette = (
            silhouette_score(values, labels)
            if len(np.unique(labels)) == 2
            else np.nan
        )
        rows.append(
            {
                "architecture_id": arch,
                "n_runs": len(values),
                "bic_one_component": models[1].bic(values),
                "bic_two_components": models[2].bic(values),
                "delta_bic_one_minus_two": models[1].bic(values) - models[2].bic(values),
                "aic_one_component": models[1].aic(values),
                "aic_two_components": models[2].aic(values),
                "delta_aic_one_minus_two": models[1].aic(values) - models[2].aic(values),
                "low_component_mean": means[order][0],
                "high_component_mean": means[order][1],
                "low_component_weight": weights[order][0],
                "high_component_weight": weights[order][1],
                "low_component_sd": sds[order][0],
                "high_component_sd": sds[order][1],
                "two_component_silhouette": silhouette,
                "two_component_preferred_by_bic": bool(
                    models[2].bic(values) < models[1].bic(values)
                ),
            }
        )
    return pd.DataFrame(rows)


def early_correlations(metrics):
    cols = [
        "accuracy_at_5000",
        "accuracy_at_10000",
        "early_accuracy_slope_per_1000_steps",
        "mid_accuracy_slope_per_1000_steps",
    ] + [
        c
        for c in metrics.columns
        if c.startswith("early_mean_") or c.startswith("early_slope_")
    ]
    rows = []
    for arch, g in metrics.groupby("architecture_id"):
        for col in cols:
            if col not in g:
                continue
            x = g[[col, "final_test_accuracy"]].dropna()
            if len(x) < 4:
                continue
            sr = stats.spearmanr(x[col], x.final_test_accuracy)
            pr = stats.pearsonr(x[col], x.final_test_accuracy)
            rows.append(
                {
                    "architecture_id": arch,
                    "early_metric": col,
                    "n_runs": len(x),
                    "spearman_rho": sr.statistic,
                    "spearman_p_value": sr.pvalue,
                    "pearson_r": pr.statistic,
                    "pearson_p_value": pr.pvalue,
                }
            )
    return pd.DataFrame(rows)


def make_plots(curves, metrics, traj, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)

    for arch in ARCHS:
        g = curves[curves.architecture_id == arch]
        if g.empty:
            continue
        plt.figure(figsize=(8.5, 5.5))
        for seed, run in g.groupby("seed"):
            plt.plot(
                run.step,
                run.validation_accuracy,
                linewidth=1.4,
                alpha=0.75,
                label=f"seed {seed}",
            )
        plt.axhline(BASELINE, linestyle="--", label="1/F baseline")
        plt.axhline(THRESHOLD_30, linestyle=":", label="0.30 threshold")
        plt.axhline(THRESHOLD_50, linestyle="-.", label="0.50 threshold")
        plt.xlabel("Training step")
        plt.ylabel("Validation accuracy")
        plt.title(f"Learning trajectories — {arch}")
        plt.ylim(0, 1.02)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(figure_dir / f"learning_curves_{arch}.png", dpi=300)
        plt.close()

    for arch in ARCHS:
        g = traj[traj.architecture_id == arch]
        if g.empty:
            continue
        plt.figure(figsize=(8.5, 5.5))
        for label, run in g.groupby("regime_group"):
            run = run.sort_values("step")
            plt.plot(run.step, run["mean"], linewidth=2.2, label=label)
            plt.fill_between(
                run.step.to_numpy(float),
                run.ci95_low.to_numpy(float),
                run.ci95_high.to_numpy(float),
                alpha=0.2,
            )
        plt.axhline(BASELINE, linestyle="--", label="1/F baseline")
        plt.axhline(THRESHOLD_50, linestyle=":", label="0.50 threshold")
        plt.xlabel("Training step")
        plt.ylabel("Mean validation accuracy")
        plt.title(f"Regime-group trajectories — {arch}")
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"regime_groups_{arch}.png", dpi=300)
        plt.close()

    data = [
        metrics.loc[metrics.architecture_id == a, "final_test_accuracy"].to_numpy()
        for a in ARCHS
    ]
    plt.figure(figsize=(8.5, 5.5))
    plt.boxplot(data, tick_labels=ARCHS, showmeans=True)
    rng = np.random.default_rng(RNG_SEED)
    for i, values in enumerate(data, 1):
        plt.scatter(
            np.full(len(values), i) + rng.normal(0, 0.035, len(values)),
            values,
            s=35,
            alpha=0.8,
        )
    plt.axhline(BASELINE, linestyle="--", label="1/F baseline")
    plt.axhline(SUCCESS_80, linestyle=":", label="0.80 success")
    plt.ylabel("Final test accuracy")
    plt.title("Final outcomes across matched seeds")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "final_accuracy_distribution.png", dpi=300)
    plt.close()

    pivot = metrics.pivot(
        index="seed", columns="architecture_id", values="final_test_accuracy"
    )
    plt.figure(figsize=(9, 5.5))
    xx = np.arange(len(ARCHS))
    for seed, row in pivot.iterrows():
        plt.plot(
            xx,
            [row.get(a, np.nan) for a in ARCHS],
            marker="o",
            alpha=0.7,
            label=f"seed {seed}",
        )
    plt.xticks(xx, ARCHS)
    plt.axhline(BASELINE, linestyle="--", label="1/F baseline")
    plt.ylabel("Final test accuracy")
    plt.title("Seed-matched architectural outcomes")
    plt.ylim(0, 1.02)
    plt.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "seed_matched_final_accuracy.png", dpi=300)
    plt.close()

    for col, name in [
        ("accuracy_at_5000", "step5000"),
        ("accuracy_at_10000", "step10000"),
    ]:
        plt.figure(figsize=(8, 5.5))
        for arch in ARCHS:
            g = metrics[metrics.architecture_id == arch].dropna(
                subset=[col, "final_test_accuracy"]
            )
            plt.scatter(
                g[col],
                g.final_test_accuracy,
                s=55,
                alpha=0.8,
                label=arch,
            )
        plt.xlabel(col.replace("_", " "))
        plt.ylabel("Final test accuracy")
        plt.title(f"Early versus final performance — {name}")
        plt.xlim(0, 1.02)
        plt.ylim(0, 1.02)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figure_dir / f"early_vs_final_{name}.png", dpi=300)
        plt.close()


def decide(metrics, mixtures, correlations):
    g = metrics[metrics.architecture_id == "H8_D64"]
    if g.empty:
        return {
            "decision": "insufficient_data",
            "rationale": "H8_D64 data are missing.",
            "next_experiment": "Recover head factorial outputs before new training.",
        }

    transitions = int(g.entered_regime_050.sum())
    late_transitions = int(
        np.sum(g.first_crossing_step_050.fillna(-np.inf) >= LATE_START)
    )
    mean_late_gain = float(g.late_gain_15000_to_final.mean())

    mix = mixtures[mixtures.architecture_id == "H8_D64"]
    bimodal = (
        bool(mix.iloc[0].two_component_preferred_by_bic)
        if len(mix)
        else False
    )
    corr = correlations[correlations.architecture_id == "H8_D64"]
    early_signal = (
        bool((corr.spearman_rho.abs() >= 0.60).any())
        if len(corr)
        else False
    )

    if transitions and (
        late_transitions >= max(1, math.ceil(transitions / 2))
        or mean_late_gain >= 0.10
    ):
        return {
            "decision": "extend_training_budget",
            "rationale": (
                f"H8_D64 crossed 0.50 in {transitions}/{len(g)} seeds; "
                f"{late_transitions} transitions occurred at/after {LATE_START}, "
                f"and mean late gain was {mean_late_gain:.3f}."
            ),
            "next_experiment": (
                "WikiText confirmation: H8_D32 versus H8_D64 only, 40,000 steps, "
                "at least 20 new matched seeds, dense late-stage logging."
            ),
        }

    if bimodal and transitions:
        return {
            "decision": "increase_seed_count",
            "rationale": (
                f"H8_D64 has a two-component mixture signal and crossed 0.50 "
                f"in {transitions}/{len(g)} seeds."
            ),
            "next_experiment": (
                "WikiText confirmation: probability-of-success replication using H8_D32 "
                "and H8_D64 only, at least 30 new matched seeds, same 20k budget."
            ),
        }

    if early_signal:
        return {
            "decision": "target_optimization_mechanism",
            "rationale": (
                "At least one early H8_D64 diagnostic has |Spearman rho| >= 0.60 "
                "with final test accuracy."
            ),
            "next_experiment": (
                "WikiText confirmation: one predeclared optimization intervention on H8_D64 "
                "(for example warmup or learning rate), not another architecture grid."
            ),
        }

    return {
        "decision": "stop_this_mechanism_branch",
        "rationale": (
            "No coherent late-transition, mixture, or early-warning signature "
            "justifies another expensive width/head experiment."
        ),
        "next_experiment": (
            "Keep Scripts 14–16 as negative mechanistic evidence and move to a "
            "different theoretically motivated causal axis."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    head_factorial = locate_head_factorial(args.input_dir)
    output = Path(args.output_dir)
    tables = output / "tables"
    figures = output / "figures"
    reports = output / "reports"
    for p in [output, tables, figures, reports]:
        p.mkdir(parents=True, exist_ok=True)

    print("Source:", head_factorial)
    print("Output:", output)

    runs, curves = load_data(head_factorial)
    audit = integrity_table(runs, curves)
    metrics = run_metrics(runs, curves)
    summary = architecture_summary(metrics)
    numeric = paired_numeric(metrics)
    binary = paired_binary(metrics)
    success_failure = success_failure_table(metrics)
    trajectories = trajectory_summary(curves, metrics)
    mixtures = mixture_table(metrics)
    correlations = early_correlations(metrics)
    decision = decide(metrics, mixtures, correlations)

    outputs = {
        "integrity_audit.csv": audit,
        "run_level_transition_metrics.csv": metrics,
        "architecture_transition_summary.csv": summary,
        "paired_numeric_contrasts.csv": numeric,
        "paired_binary_contrasts.csv": binary,
        "successful_vs_unsuccessful_H8_D64.csv": success_failure,
        "trajectory_group_summary.csv": trajectories,
        "mixture_bimodality_analysis.csv": mixtures,
        "early_warning_correlations.csv": correlations,
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / name, index=False)

    make_plots(curves, metrics, trajectories, figures)

    report = f"""# regime dynamics Decision Report

## Scope
Analysis-only evaluation of completed head factorial learning curves. No retraining.

## Primary question
Does architecture change the probability or timing of entering a successful
associative-binding regime rather than uniformly shifting final accuracy?

## Fixed definitions
- Present-value baseline: {BASELINE:.4f}
- Initial regime threshold: {THRESHOLD_30:.2f}
- Strong regime threshold: {THRESHOLD_50:.2f}
- High final-success threshold: {SUCCESS_80:.2f}
- Early window: through step {EARLY_END}
- Mid checkpoint: step {MID_END}
- Late window: step {LATE_START} onward

## Decision
**{decision["decision"]}**

{decision["rationale"]}

## Next experiment
{decision["next_experiment"]}

## Interpretation boundary
Mixture and diagnostic analyses use only 10 seeds per architecture and are
mechanism-generating. They must not be presented as definitive proof.
"""
    (reports / "REGIME_DYNAMICS_DECISION_REPORT.md").write_text(report, encoding="utf-8")
    (reports / "REGIME_DYNAMICS_DECISION.json").write_text(
        json.dumps(
            {
                **decision,
                "head_factorial_source": head_factorial.name,
                "regime_dynamics_output": output.name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    archive = shutil.make_archive(
        str(output.parent / "REGIME_DYNAMICS_RESULTS"),
        "zip",
        root_dir=output,
    )

    print("\nArchitecture summary:")
    print(summary.to_string(index=False))
    print("\nDecision:")
    print(json.dumps(decision, indent=2))
    print("\nArchive:", archive)


if __name__ == "__main__":
    main()
