"""M3 evaluation: metrics, per-patient breakdown, sensitivity subsets and seeds.

Input contract (project plan): one CSV per training seed,
``results/<experiment>/seed<k>.csv`` with columns
``pid, y_true, y_pred, p_N, p_S, p_V, p_F`` in the row order of the test NPZ.
Extra columns (for example ``sample``) are allowed and kept.

Conventions used throughout:
- Recall is the ECG literature's sensitivity (Se); precision is its positive
  predictivity (+P). F1 = 2TP / (2TP + FP + FN).
- Undefined values stay undefined (NaN): precision of a class that is never
  predicted, recall of a class with no support. A present class that is never
  detected has recall 0 and F1 0. Macro-F1 and balanced accuracy average only the
  defined per-class values.
- Seeds measure training variability: mean +- sample SD (ddof=1) over seeds.
- The patient bootstrap measures test-population variability: test records are
  resampled with replacement (the same draws for every seed and experiment), so
  intervals reflect which unseen patients happen to be in the test set.

Run from the repository root: ``python -m src.evaluate --help``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import json
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

from .data import CLASS_NAMES, load_split


K = len(CLASS_NAMES)
PROB_COLUMNS = tuple(f"p_{name}" for name in CLASS_NAMES)
REQUIRED_COLUMNS = ("pid", "y_true", "y_pred") + PROB_COLUMNS
# Record 232 holds 75% of the DS2 S beats and record 213 holds 93% of the DS2
# F beats, so S and F results are also reported without them. These are
# supplementary sensitivity analyses; "all" is always the main result.
SUBSETS = {"all": (), "without_232": (232,), "without_213": (213,)}
TEST_FILES = ("patient_test.npz", "random_test.npz")
DUMMY_COLUMN = "dummy"


# ---------------------------------------------------------------- loading

def load_predictions(path):
    """Read one prediction CSV and check it against the agreed contract."""
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df.empty:
        raise ValueError(f"{path}: no rows")
    for column in ("pid", "y_true", "y_pred"):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float64)
        if not (np.isfinite(values).all() and np.all(values == np.round(values))):
            raise ValueError(f"{path}: {column} must contain integers")
        df[column] = values.astype(np.int64)
        if column != "pid" and not df[column].isin(range(K)).all():
            raise ValueError(f"{path}: {column} must be in 0..{K - 1} ({', '.join(CLASS_NAMES)})")
    probs = df[list(PROB_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all() or probs.min() < 0 or probs.max() > 1 + 1e-6:
        raise ValueError(f"{path}: probabilities must be finite and within [0, 1]")
    if np.abs(probs.sum(axis=1) - 1).max() > 1e-3:
        raise ValueError(f"{path}: p_N..p_F must sum to 1 per row (softmax outputs)")
    disagree = int(np.count_nonzero(probs.argmax(axis=1) != df["y_pred"].to_numpy()))
    if disagree:
        warnings.warn(f"{path}: y_pred is not the argmax probability on {disagree} rows")
    return df


@lru_cache(maxsize=4)
def _cached_split(path):
    return load_split(path)


def find_test_split(df, processed_dir):
    """Return (file name, arrays) of the test NPZ matching these predictions.

    A match needs the same length, record IDs and labels in the same row order.
    patient_test and patient_strict_test are identical, so the first is used.
    Returns (None, None) if no generated test file matches.
    """
    for name in TEST_FILES:
        path = Path(processed_dir) / name
        if not path.is_file():
            continue
        arrays = _cached_split(str(path))
        if (len(arrays["y"]) == len(df)
                and np.array_equal(arrays["pid"], df["pid"].to_numpy())
                and np.array_equal(arrays["y"], df["y_true"].to_numpy())):
            return name, arrays
    return None, None


def attach_beat_identity(df, arrays):
    """Add beat identity and raw timing from the aligned test NPZ.

    ``sample``/``subject_id`` make each row traceable; ``pre_rr_s`` and
    ``rr_ratio`` (previous RR / mean of the last 10 RR) support failure analysis.
    """
    for column in ("sample", "subject_id"):
        if column in df.columns and not np.array_equal(df[column].to_numpy(), arrays[column]):
            raise ValueError(f"prediction '{column}' column disagrees with the test NPZ")
    df = df.copy()
    df["sample"] = arrays["sample"]
    df["subject_id"] = arrays["subject_id"]
    df["pre_rr_s"] = arrays["rr_raw"][:, 0]
    df["rr_ratio"] = arrays["rr_raw"][:, 3]
    return df


def _natural_key(text):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(text))]


def seed_files(exp_dir):
    """Test prediction files ``seed<k>.csv`` only. M2's training also writes
    ``seed<k>_val.csv`` and ``seed<k>_history.csv`` to the same folder."""
    return sorted((p for p in Path(exp_dir).glob("seed*.csv")
                   if re.fullmatch(r"seed\d+", p.stem)), key=lambda p: _natural_key(p.stem))


def load_experiment(exp_dir, processed_dir=None, allow_unmatched=False):
    """Load every ``seed<k>.csv`` of one experiment; all seeds share one test set.

    With ``processed_dir``, every file must match a test NPZ (same records and
    labels in the same order); ``allow_unmatched`` downgrades a mismatch to a
    warning for generic CSV analysis, never for reported results.
    """
    exp_dir = Path(exp_dir)
    paths = seed_files(exp_dir)
    if not paths:
        raise FileNotFoundError(f"No seed*.csv files in {exp_dir}")
    frames, test_name = {}, None
    for path in paths:
        df = load_predictions(path)
        if processed_dir is not None:
            name, arrays = find_test_split(df, processed_dir)
            if name is None:
                message = f"{path}: no matching test NPZ in {processed_dir}; row-order alignment not verified"
                if not allow_unmatched:
                    raise ValueError(message + " (pass allow_unmatched / --allow-unmatched to analyse anyway)")
                warnings.warn(message)
            else:
                df = attach_beat_identity(df, arrays)
                test_name = name
        frames[path.stem] = df
    first = next(iter(frames.values()))
    for seed, df in frames.items():
        if not (np.array_equal(df["pid"].to_numpy(), first["pid"].to_numpy())
                and np.array_equal(df["y_true"].to_numpy(), first["y_true"].to_numpy())):
            raise ValueError(f"{exp_dir}/{seed}: seeds do not share the same test rows")
    return {"name": exp_dir.name, "seeds": frames, "test": test_name,
            "dummy": any(DUMMY_COLUMN in df.columns for df in frames.values())}


def discover_experiments(results_dir):
    """Experiment folders under ``results_dir`` that contain seed CSVs."""
    results_dir = Path(results_dir)
    found = [p for p in results_dir.iterdir()
             if p.is_dir() and not p.name.startswith(("_", ".")) and p.name != "summary"
             and seed_files(p)]
    return sorted(found, key=lambda p: _natural_key(p.name))


# ---------------------------------------------------------------- metrics

def confusion_matrix(y_true, y_pred, k=K):
    """Rows are true classes, columns are predicted classes."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    return np.bincount(k * y_true + y_pred, minlength=k * k).reshape(k, k)


def _ratio(num, den):
    num, den = np.broadcast_arrays(np.asarray(num, dtype=np.float64),
                                   np.asarray(den, dtype=np.float64))
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _nanmean(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices stay NaN
        return np.nanmean(values, axis=axis)


def metrics_from_confusion(cm):
    """Metric arrays from (..., K, K) confusion matrices (vectorised for bootstrap)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    recall = _ratio(tp, support)
    f1 = _ratio(2 * tp, support + predicted)
    return {"n": cm.sum(axis=(-2, -1)),
            "accuracy": _ratio(tp.sum(axis=-1), cm.sum(axis=(-2, -1))),
            "macro_f1": _nanmean(f1, -1),
            "balanced_accuracy": _nanmean(recall, -1),
            "support": support, "precision": _ratio(tp, predicted),
            "recall": recall, "f1": f1}


def metric_value(metrics, name):
    """Pick ``macro_f1``-style overall names or ``S_recall``-style class names."""
    if name in ("n", "accuracy", "macro_f1", "balanced_accuracy"):
        return metrics[name]
    cls, _, kind = name.partition("_")
    if cls not in CLASS_NAMES or kind not in ("support", "precision", "recall", "f1"):
        raise KeyError(f"unknown metric {name!r}")
    return metrics[kind][..., CLASS_NAMES.index(cls)]


def metric_names():
    names = ["n", "accuracy", "macro_f1", "balanced_accuracy"]
    for cls in CLASS_NAMES:
        names += [f"{cls}_{kind}" for kind in ("support", "precision", "recall", "f1")]
    return names


def summarise_confusion(cm):
    """One flat row of scalar metrics for a single confusion matrix."""
    metrics = metrics_from_confusion(cm)
    return {name: float(metric_value(metrics, name)) for name in metric_names()}


def record_confusions(df, records=None):
    """Per-record confusion matrices: (records, array of shape (R, K, K))."""
    pid = df["pid"].to_numpy()
    records = np.unique(pid) if records is None else np.asarray(records)
    index = np.searchsorted(records, pid)
    if np.any(index >= len(records)) or not np.array_equal(records[index], pid):
        raise ValueError("predictions contain records outside the requested list")
    flat = (index * K + df["y_true"].to_numpy()) * K + df["y_pred"].to_numpy()
    cms = np.bincount(flat, minlength=len(records) * K * K).reshape(len(records), K, K)
    return records, cms


# ---------------------------------------------------------------- analyses

def per_patient_table(df):
    """One row per record: support, recall and false positives for each class.

    Recall is NaN when the record has no beats of that class; support is always
    shown so small denominators are visible.
    """
    records, cms = record_confusions(df)
    rows = []
    for record, cm in zip(records, cms):
        metrics = metrics_from_confusion(cm)
        row = {"pid": int(record), "n": int(cm.sum()), "accuracy": float(metrics["accuracy"])}
        for i, cls in enumerate(CLASS_NAMES):
            row[f"{cls}_support"] = int(cm[i].sum())
            row[f"{cls}_recall"] = float(metrics["recall"][i])
            row[f"{cls}_false_pos"] = int(cm[:, i].sum() - cm[i, i])
        rows.append(row)
    return pd.DataFrame(rows)


def per_patient_mean(frames):
    """Per-record table averaged over seeds (supports are identical across seeds)."""
    tables = [per_patient_table(df) for df in frames.values()]
    stacked = pd.concat(tables)
    mean = stacked.groupby("pid", sort=True).mean()
    for column in ["n"] + [f"{cls}_support" for cls in CLASS_NAMES]:
        mean[column] = mean[column].round().astype(np.int64)
    mean.insert(0, "seeds", len(tables))
    return mean.reset_index()


def top_confusions(frames, limit=20):
    """Largest (record, true, predicted) error cells, averaged over seeds."""
    totals = None
    for df in frames.values():
        records, cms = record_confusions(df)
        totals = cms.astype(np.float64) if totals is None else totals + cms
    totals /= len(frames)
    rows = []
    for r, record in enumerate(records):
        for i, true in enumerate(CLASS_NAMES):
            support = totals[r, i].sum()
            for j, pred in enumerate(CLASS_NAMES):
                if i != j and totals[r, i, j] > 0:
                    rows.append({"pid": int(record), "true": true, "pred": pred,
                                 "beats": float(totals[r, i, j]),
                                 "share_of_true": float(totals[r, i, j] / support)})
    table = pd.DataFrame(rows, columns=["pid", "true", "pred", "beats", "share_of_true"])
    return table.sort_values("beats", ascending=False).head(limit).reset_index(drop=True)


RATIO_BINS = (0.0, 0.8, 0.9, 1.1, np.inf)


def timing_breakdown(frames, bins=RATIO_BINS, focus=232):
    """Recall of each abnormal class by prematurity bin, averaged over seeds.

    Shows whether missed S beats are the ones without an early-timing cue, and
    whether that differs between the dominant record ``focus`` and the rest.
    Needs predictions aligned to the test NPZ (``rr_ratio`` column).
    """
    first = next(iter(frames.values()))
    if "rr_ratio" not in first.columns:
        return pd.DataFrame()
    labels = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}" for lo, hi in zip(bins[:-1], bins[1:])]
    rows = []
    for cls_index, cls in enumerate(CLASS_NAMES[1:], start=1):
        for group, select in (("all", None), (f"record_{focus}", True), ("other_records", False)):
            for label, lo, hi in zip(labels, bins[:-1], bins[1:]):
                recalls, support = [], 0
                for df in frames.values():
                    mask = (df["y_true"] == cls_index) & (df["rr_ratio"] >= lo) & (df["rr_ratio"] < hi)
                    if select is not None:
                        mask &= (df["pid"] == focus) if select else (df["pid"] != focus)
                    support = int(mask.sum())
                    if support:
                        recalls.append(float(np.mean(df.loc[mask, "y_pred"] == cls_index)))
                rows.append({"class": cls, "records": group, "ratio_bin": label, "support": support,
                             "recall": float(np.mean(recalls)) if recalls else np.nan})
    return pd.DataFrame(rows)


def experiment_confusions(experiment, records=None):
    """(records, array (seeds, R, K, K)) for one experiment."""
    first = next(iter(experiment["seeds"].values()))
    records = np.unique(first["pid"].to_numpy()) if records is None else records
    stacked = [record_confusions(df, records)[1] for df in experiment["seeds"].values()]
    return records, np.stack(stacked)


def metrics_by_seed(experiment, subsets=SUBSETS):
    """Long table: experiment x seed x subset with every scalar metric."""
    records, cms = experiment_confusions(experiment)
    rows = []
    for seed, seed_cms in zip(experiment["seeds"], cms):
        for subset, excluded in subsets.items():
            keep = ~np.isin(records, excluded)
            rows.append({"experiment": experiment["name"], "test": experiment["test"],
                         "seed": seed, "subset": subset,
                         **summarise_confusion(seed_cms[keep].sum(axis=0))})
    return pd.DataFrame(rows)


def mean_std(by_seed):
    """Mean and sample SD (ddof=1) over seeds, per experiment and subset."""
    numeric = [c for c in metric_names() if c in by_seed.columns]
    grouped = by_seed.groupby(["experiment", "test", "subset"], sort=False, dropna=False)
    mean = grouped[numeric].mean().add_suffix("_mean")
    std = grouped[numeric].std(ddof=1).add_suffix("_std")
    count = grouped.size().rename("seeds")
    return pd.concat([count, mean, std], axis=1).reset_index()


def patient_bootstrap(experiments, metrics=("macro_f1",), n_boot=2000, seed=0):
    """Seed-averaged metrics on record-level bootstrap resamples.

    ``experiments`` maps name -> (seeds, R, K, K) confusion arrays that share one
    record order. Every experiment sees the same resampled records, so columns
    can be differenced for paired comparisons. Returns a DataFrame with one row
    per resample and one column per ``<experiment>|<metric>``.
    """
    shapes = {cms.shape[1] for cms in experiments.values()}
    if len(shapes) != 1:
        raise ValueError("bootstrap experiments must share one test record list")
    rng = np.random.default_rng(seed)
    n_records = shapes.pop()
    draws = rng.integers(0, n_records, size=(n_boot, n_records))
    weights = np.stack([np.bincount(d, minlength=n_records) for d in draws]).astype(np.float64)
    columns = {}
    for name, cms in experiments.items():
        boot = np.einsum("br,srij->bsij", weights, cms.astype(np.float64))
        values = metrics_from_confusion(boot)
        for metric in metrics:
            columns[f"{name}|{metric}"] = _nanmean(metric_value(values, metric), 1)
    return pd.DataFrame(columns)


def interval(values, level=0.95):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(values, [tail, 100 - tail])
    return float(low), float(high)


def bootstrap_table(experiments, metrics, reference=None, n_boot=2000, seed=0):
    """95% patient-bootstrap intervals, plus paired differences to ``reference``."""
    boot = patient_bootstrap(experiments, metrics, n_boot=n_boot, seed=seed)
    rows = []
    for name, cms in experiments.items():
        point = metrics_from_confusion(cms.sum(axis=1))  # per seed, all records
        for metric in metrics:
            row = {"experiment": name, "metric": metric,
                   "estimate": float(_nanmean(metric_value(point, metric), 0))}
            row["ci_low"], row["ci_high"] = interval(boot[f"{name}|{metric}"])
            if reference is not None and reference in experiments and name != reference:
                diff = boot[f"{name}|{metric}"] - boot[f"{reference}|{metric}"]
                ref_point = metrics_from_confusion(experiments[reference].sum(axis=1))
                row["reference"] = reference
                row["diff"] = row["estimate"] - float(_nanmean(metric_value(ref_point, metric), 0))
                row["diff_low"], row["diff_high"] = interval(diff)
                row["p_improves"] = float(np.mean(diff[np.isfinite(diff)] > 0))
            rows.append(row)
    return pd.DataFrame(rows)


def trivial_baseline(y_true, label=0):
    """Metrics of predicting one class for every beat (default: always N)."""
    y_true = np.asarray(y_true)
    return summarise_confusion(confusion_matrix(y_true, np.full_like(y_true, label)))


# ---------------------------------------------------------------- post-hoc analyses
# Added 27 Oct, after the first test evaluation, to explain why S detection fails
# on new patients. They re-read the saved predictions and train nothing.

def auroc(scores, positive):
    """Area under the ROC curve from the rank-sum statistic (ties get mid-ranks).

    NaN when only one kind of beat is present.
    """
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = pd.Series(np.asarray(scores, dtype=np.float64)).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ranking_table(frames, cls="S", min_support=10):
    """How well ``p_<cls>`` ranks ``cls`` beats above the rest, pooled and per record.

    Pooled AUROC compares beats across patients, so it drops when each patient's
    scores are shifted by a different amount even if every patient's own beats
    are ordered correctly. Within-record AUROC (records with at least
    ``min_support`` beats of each kind) removes that shift. ``predicted_share``
    is the fraction of beats whose argmax is ``cls``. Means and SD over seeds.
    """
    c = CLASS_NAMES.index(cls)
    rows = []
    for seed, df in frames.items():
        parts = [("pooled", df)] + [(str(pid), part) for pid, part in df.groupby("pid")]
        for record, part in parts:
            positive = part["y_true"].to_numpy() == c
            if record != "pooled" and min(positive.sum(), (~positive).sum()) < min_support:
                continue
            rows.append({"record": record, "seed": seed, "support": int(positive.sum()),
                         "n": len(part), "auroc": auroc(part[f"p_{cls}"], positive),
                         "predicted_share": float(np.mean(part["y_pred"].to_numpy() == c))})
    long = pd.DataFrame(rows)
    table = long.groupby("record", sort=False).agg(
        support=("support", "first"), n=("n", "first"), auroc=("auroc", "mean"),
        auroc_sd=("auroc", "std"), predicted_share=("predicted_share", "mean"))
    return table.reset_index()


THRESHOLDS = np.unique(np.concatenate([np.logspace(-4, -1, 31), np.linspace(0.1, 0.9, 17)]))


def threshold_rule(probs, c, t):
    """Predict class ``c`` when its probability is >= t, else the most probable other class."""
    probs = np.asarray(probs, dtype=np.float64)
    others = probs.copy()
    others[:, c] = -np.inf
    return np.where(probs[:, c] >= t, c, others.argmax(axis=1))


def threshold_transfer(exp_dir, frames, cls="S", grid=THRESHOLDS, focus=232):
    """Choose a ``cls`` threshold on validation patients and score it on test patients.

    For each seed with a ``seed<k>_val.csv`` (validation predictions, written by
    training or by ``src.predict``), t maximises the ``cls`` F1 on validation
    data (smallest t on ties); the test predictions are only scored. Tests
    whether the class is missed because the argmax threshold is badly placed
    for new patients, and whether a threshold set on other patients fixes it.
    """
    c = CLASS_NAMES.index(cls)
    rows = []
    for seed, test in frames.items():
        val_path = Path(exp_dir) / f"{seed}_val.csv"
        if not val_path.is_file():
            continue
        val = load_predictions(val_path)
        val_probs = val[list(PROB_COLUMNS)].to_numpy()
        val_y = val["y_true"].to_numpy()
        f1s = [summarise_confusion(confusion_matrix(val_y, threshold_rule(val_probs, c, t)))[f"{cls}_f1"]
               for t in grid]
        best = int(np.nanargmax(f1s))
        y_true = test["y_true"].to_numpy()
        y_rule = threshold_rule(test[list(PROB_COLUMNS)].to_numpy(), c, grid[best])
        keep = test["pid"].to_numpy() != focus
        rule_all = summarise_confusion(confusion_matrix(y_true, y_rule))
        rule_rest = summarise_confusion(confusion_matrix(y_true[keep], y_rule[keep]))
        argmax_all = summarise_confusion(confusion_matrix(y_true, test["y_pred"].to_numpy()))
        rows.append({"seed": seed, "threshold": float(grid[best]), f"val_{cls}_f1": float(f1s[best]),
                     f"val_{cls}_auroc": auroc(val[f"p_{cls}"], val_y == c),
                     **{f"{cls}_{k}": rule_all[f"{cls}_{k}"] for k in ("recall", "precision", "f1")},
                     **{f"{cls}_{k}_without_{focus}": rule_rest[f"{cls}_{k}"]
                        for k in ("recall", "precision", "f1")},
                     "macro_f1": rule_all["macro_f1"], "macro_f1_argmax": argmax_all["macro_f1"]})
    return pd.DataFrame(rows)


def posthoc_markdown(rankings, transfers, cls="S", focus=232):
    """Summary section for the post-hoc ranking and threshold analyses."""
    parts = [f"\n## Post-hoc (added 27 Oct, after the first test evaluation): {cls} ranking\n",
             f"AUROC of p_{cls} for {cls} vs all other beats, mean over seeds. 'Within records' "
             f"is the median over test records with at least 10 {cls} and 10 other beats. "
             f"'Predicted {cls}' is the share of beats whose argmax is {cls}.\n"]
    rows = []
    for name, table in rankings.items():
        pooled = table[table["record"] == "pooled"].iloc[0]
        records = table[table["record"] != "pooled"]
        own = table[table["record"] == str(focus)]
        rows.append([name, _fmt(pooled["auroc"], pooled["auroc_sd"]),
                     f"{records['auroc'].median():.3f} ({len(records)} records)" if len(records) else "n/a",
                     _fmt(own["auroc"].iloc[0], own["auroc_sd"].iloc[0]) if len(own) else "n/a",
                     f"{pooled['predicted_share']:.3f}",
                     f"{own['predicted_share'].iloc[0]:.3f}" if len(own) else "n/a"])
    parts.append(markdown_table(["Experiment", "Pooled AUROC", "Within records (median)",
                                 f"Within {focus}", f"Predicted {cls} (all)",
                                 f"Predicted {cls} ({focus})"], rows))
    if transfers:
        parts += [f"\n## Post-hoc: {cls} threshold chosen on validation patients\n",
                  f"Predict {cls} when p_{cls} >= t; t maximises {cls} F1 on the validation "
                  "records. Mean ± SD over seeds.\n"]
        keys = ["threshold", f"val_{cls}_auroc", f"{cls}_recall", f"{cls}_precision", f"{cls}_f1",
                f"{cls}_f1_without_{focus}", "macro_f1", "macro_f1_argmax"]
        rows = [[name] + [_fmt(t[k].mean(), t[k].std(ddof=1) if len(t) > 1 else None, 4 if k == "threshold" else 3)
                          for k in keys] for name, t in transfers.items()]
        parts.append(markdown_table(["Experiment", "t", f"Val {cls} AUROC", f"{cls} Se", f"{cls} +P",
                                     f"{cls} F1", f"{cls} F1 w/o {focus}", "Macro-F1 (rule)",
                                     "Macro-F1 (argmax)"], rows))
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------- reporting

def _fmt(mean, std=None, digits=3):
    if mean is None or not np.isfinite(mean):
        return "n/a"
    if std is None or not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def markdown_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def summary_rows(summary, subset, metrics):
    rows = []
    for _, r in summary[summary["subset"] == subset].iterrows():
        test = "unverified" if pd.isna(r["test"]) else r["test"].replace(".npz", "")
        rows.append([r["experiment"], test, int(r["seeds"])]
                    + [_fmt(r[f"{m}_mean"], r[f"{m}_std"]) for m in metrics])
    return rows


LABELS = {"accuracy": "Acc", "macro_f1": "Macro-F1", "balanced_accuracy": "Bal. acc"}
for _cls in CLASS_NAMES:
    LABELS.update({f"{_cls}_recall": f"{_cls} Se", f"{_cls}_precision": f"{_cls} +P",
                   f"{_cls}_f1": f"{_cls} F1"})


def write_summary_markdown(path, summary, boot, sources, dummy):
    overall = ["accuracy", "macro_f1", "balanced_accuracy"] + [f"{c}_f1" for c in CLASS_NAMES]
    per_class = [f"{c}_{k}" for c in CLASS_NAMES[1:] for k in ("recall", "precision")]
    base = ["Experiment", "Test set", "Seeds"]
    parts = []
    if dummy:
        parts.append("> **DUMMY PREDICTIONS: pipeline test only. These are NOT experimental "
                     "results and must not appear in the report.**\n")
    parts += [
        "# Evaluation summary\n",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `src/evaluate.py` "
        f"from: {', '.join(sources)}. Values are mean ± SD over seeds (ddof=1). "
        "Se = recall (sensitivity), +P = precision (positive predictivity).\n",
        "## Main results (all test records)\n",
        markdown_table(base + [LABELS[m] for m in overall], summary_rows(summary, "all", overall)),
        "\n## Abnormal classes: sensitivity and positive predictivity\n",
        markdown_table(base + [LABELS[m] for m in per_class], summary_rows(summary, "all", per_class)),
        "\n## Sensitivity analysis: without record 232 (holds most test S beats)\n",
        markdown_table(base + [LABELS[m] for m in ("macro_f1", "S_recall", "S_precision", "S_f1")],
                       summary_rows(summary, "without_232", ["macro_f1", "S_recall", "S_precision", "S_f1"])),
        "\n## Sensitivity analysis: without record 213 (holds most test F beats)\n",
        markdown_table(base + [LABELS[m] for m in ("macro_f1", "F_recall", "F_precision", "F_f1")],
                       summary_rows(summary, "without_213", ["macro_f1", "F_recall", "F_precision", "F_f1"])),
    ]
    if boot is not None and len(boot):
        parts.append("\n## Patient-bootstrap 95% intervals (seed-averaged; records resampled)\n")
        rows = []
        for _, r in boot.iterrows():
            row = [r["experiment"], LABELS.get(r["metric"], r["metric"]), _fmt(r["estimate"]),
                   f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]"]
            if "diff" in boot.columns and pd.notna(r.get("diff")):
                row += [f"{r['diff']:+.3f}", f"[{r['diff_low']:+.3f}, {r['diff_high']:+.3f}]",
                        f"{r['p_improves']:.2f}"]
            else:
                row += ["", "", ""]
            rows.append(row)
        ref = boot["reference"].dropna().iloc[0] if "reference" in boot.columns and boot["reference"].notna().any() else "reference"
        parts.append(markdown_table(["Experiment", "Metric", "Estimate", "95% CI",
                                     f"Δ vs {ref}", "Δ 95% CI", "P(Δ>0)"], rows))
        parts.append("\nP(Δ>0) is the share of bootstrap resamples in which the experiment beats the "
                     "reference; it is a descriptive stability measure, not a p-value.")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def summarise(results_dir, processed_dir, output_dir, reference="E1", n_boot=2000,
              figures=True, boot_metrics=("macro_f1", "S_f1", "V_f1", "F_f1"),
              allow_unmatched=False):
    """Evaluate every experiment under ``results_dir`` and write tables/figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = [load_experiment(p, processed_dir, allow_unmatched)
                   for p in discover_experiments(results_dir)]
    if not experiments:
        raise FileNotFoundError(f"No results/<experiment>/seed*.csv found in {results_dir}")
    dummy = any(e["dummy"] for e in experiments)
    by_seed = pd.concat([metrics_by_seed(e) for e in experiments], ignore_index=True)
    summary = mean_std(by_seed)
    by_seed.to_csv(output_dir / "metrics_by_seed.csv", index=False)
    summary.to_csv(output_dir / "metrics_mean_std.csv", index=False)
    for e in experiments:
        per_patient_mean(e["seeds"]).to_csv(output_dir / f"per_patient_{e['name']}.csv", index=False)
        top_confusions(e["seeds"]).to_csv(output_dir / f"top_confusions_{e['name']}.csv", index=False)
        timing = timing_breakdown(e["seeds"])
        if not timing.empty:
            timing.to_csv(output_dir / f"timing_breakdown_{e['name']}.csv", index=False)

    # Bootstrap only among experiments evaluated on the same records and labels.
    groups = {}
    for e in experiments:
        first = next(iter(e["seeds"].values()))
        key = first["pid"].to_numpy().tobytes() + first["y_true"].to_numpy().tobytes()
        groups.setdefault(key, []).append(e)
    boot_tables = []
    for group in groups.values():
        cms = {e["name"]: experiment_confusions(e)[1] for e in group}
        ref = reference if reference in cms else None
        boot_tables.append(bootstrap_table(cms, list(boot_metrics), ref, n_boot=n_boot))
    boot = pd.concat(boot_tables, ignore_index=True)
    boot.to_csv(output_dir / "bootstrap.csv", index=False)
    write_summary_markdown(output_dir / "summary.md", summary, boot,
                           [e["name"] for e in experiments], dummy)

    rankings, transfers = {}, {}
    for path, e in zip(discover_experiments(results_dir), experiments):
        rankings[e["name"]] = ranking_table(e["seeds"])
        rankings[e["name"]].to_csv(output_dir / f"ranking_S_{e['name']}.csv", index=False)
        transfer = threshold_transfer(path, e["seeds"])
        if not transfer.empty:
            transfers[e["name"]] = transfer
            transfer.to_csv(output_dir / f"threshold_transfer_S_{e['name']}.csv", index=False)
    with open(output_dir / "summary.md", "a", encoding="utf-8") as stream:
        stream.write(posthoc_markdown(rankings, transfers))
    if figures:
        from . import figures as fig
        fig.experiment_figures(experiments, summary, boot, output_dir / "figures",
                               reference=reference, dummy=dummy)
        if not dummy:
            fig.plot_training_curves(results_dir, output_dir / "figures" / "training_curves")
    print(f"Evaluated {len(experiments)} experiment(s); wrote {output_dir}", flush=True)
    return summary


# ---------------------------------------------------------------- data checks

def template_similarity(processed_dir, record=232):
    """Pearson correlation between mean beat templates (standardised MLII).

    Compares the S beats of one test record with that record's own N beats and
    with the pooled training N and S templates (patient protocol).
    """
    train = _cached_split(str(Path(processed_dir) / "patient_train.npz"))
    test = _cached_split(str(Path(processed_dir) / "patient_test.npz"))

    def template(arrays, cls, pid=None):
        mask = arrays["y"] == CLASS_NAMES.index(cls)
        if pid is not None:
            mask &= arrays["pid"] == pid
        return arrays["X"][mask, 0].mean(axis=0, dtype=np.float64)

    s_rec, n_rec = template(test, "S", record), template(test, "N", record)
    s_train, n_train = template(train, "S"), template(train, "N")
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731
    return {"record": record,
            "S_vs_own_N": corr(s_rec, n_rec),
            "S_vs_train_S": corr(s_rec, s_train),
            "S_vs_train_N": corr(s_rec, n_train),
            "train_S_vs_train_N": corr(s_train, n_train),
            "r_peak_amplitude": {"train_N": float(n_train[128]), "train_S": float(s_train[128]),
                                 "record_N": float(n_rec[128]), "record_S": float(s_rec[128])}}


def rhythm_context(processed_dir, raw_dir, files=("patient_train.npz", "patient_val.npz",
                                                   "patient_test.npz")):
    """Rhythm annotation ('+' markers, e.g. '(N', '(SVTA', '(SBR') in force at each
    S beat, from the original MIT-BIH annotation files. Labels are used only to
    describe the data, never as model inputs.
    """
    import wfdb
    rows = []
    for name in files:
        arrays = _cached_split(str(Path(processed_dir) / name))
        s_beats = arrays["y"] == CLASS_NAMES.index("S")
        for record in np.unique(arrays["pid"][s_beats]):
            ann = wfdb.rdann(str(Path(raw_dir) / str(record)), "atr")
            changes = [(s, note.strip("\x00")) for s, sym, note
                       in zip(ann.sample, ann.symbol, ann.aux_note) if sym == "+"]
            starts = np.array([s for s, _ in changes], dtype=np.int64)
            mask = s_beats & (arrays["pid"] == record)
            index = np.searchsorted(starts, arrays["sample"][mask], side="right") - 1
            rhythm = np.array([changes[i][1] if i >= 0 else "unknown" for i in index])
            ratio = arrays["rr_raw"][mask, 3]
            for value in np.unique(rhythm):
                rows.append({"partition": name.replace(".npz", ""), "pid": int(record),
                             "rhythm": value.lstrip("("), "S_beats": int(np.sum(rhythm == value)),
                             "median_ratio": float(np.median(ratio[rhythm == value]))})
    return pd.DataFrame(rows)


DESCRIBED_FILES = ("patient_train.npz", "patient_strict_train.npz", "patient_val.npz",
                   "patient_test.npz", "random_train.npz", "random_test.npz")


def describe_data(processed_dir, output_dir, raw_dir=Path("data/raw/mitdb")):
    """Dataset-level facts for the report: class concentration, S-beat timing
    and the always-N baseline. These are properties of the labelled data, not
    model results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    facts = {}
    for name in DESCRIBED_FILES:
        arrays = _cached_split(str(Path(processed_dir) / name))
        y, pid, ratio = arrays["y"], arrays["pid"], arrays["rr_raw"][:, 3]
        rows = []
        for record in np.unique(pid):
            counts = np.bincount(y[pid == record], minlength=K)
            rows.append({"pid": int(record), **{c: int(n) for c, n in zip(CLASS_NAMES, counts)}})
        table = pd.DataFrame(rows)
        table.to_csv(output_dir / f"class_counts_by_record_{name.replace('.npz', '')}.csv", index=False)
        concentration = {}
        for cls in CLASS_NAMES[1:]:
            ordered = table.sort_values(cls, ascending=False)
            total = int(table[cls].sum())
            concentration[cls] = {
                "total": total, "top_record": int(ordered.iloc[0]["pid"]),
                "top_share": float(ordered.iloc[0][cls] / total),
                "top2_share": float(ordered.iloc[:2][cls].sum() / total),
                "records_with_class": int((table[cls] > 0).sum()),
                "records": int(len(table))}
        # pre-RR / mean of the last 10 RR; < 0.9 is a simple "premature" marker.
        timing = {cls: {"median_ratio": float(np.median(ratio[y == i])),
                        "share_ratio_below_0.9": float(np.mean(ratio[y == i] < 0.9))}
                  for i, cls in enumerate(CLASS_NAMES)}
        facts[name] = {"n": int(len(y)), "classes": dict(zip(CLASS_NAMES, map(int, np.bincount(y, minlength=K)))),
                       "concentration": concentration, "rr_timing": timing}
        if name in TEST_FILES:
            facts[name]["always_N"] = trivial_baseline(y)
        timing_rows = []
        for record in np.unique(pid):
            row = {"pid": int(record)}
            for i, cls in enumerate(CLASS_NAMES):
                mask = (pid == record) & (y == i)
                row[f"{cls}_n"] = int(mask.sum())
                row[f"{cls}_median_ratio"] = float(np.median(ratio[mask])) if mask.any() else np.nan
                row[f"{cls}_median_pre_rr_s"] = (float(np.median(arrays["rr_raw"][mask, 0]))
                                                 if mask.any() else np.nan)
            timing_rows.append(row)
        pd.DataFrame(timing_rows).to_csv(
            output_dir / f"rr_timing_by_record_{name.replace('.npz', '')}.csv", index=False)
    facts["record_232_templates"] = template_similarity(processed_dir)
    context = rhythm_context(processed_dir, raw_dir)
    context.to_csv(output_dir / "s_beat_rhythm_context.csv", index=False)
    totals = context.groupby(["partition", "rhythm"])["S_beats"].sum().unstack(fill_value=0)
    facts["s_beat_rhythm_share"] = (totals.div(totals.sum(axis=1), axis=0).round(4)
                                    .to_dict(orient="index"))
    path = output_dir / "data_facts.json"
    path.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)
    return facts


# ---------------------------------------------------------------- dummy data

def make_dummy_predictions(arrays, seed, skill=0.8):
    """Synthetic predictions for testing this pipeline only, never for reporting.

    The true class gets extra probability mass ``skill``; the rest is random.
    Rows carry a ``dummy`` column so every summary is stamped as a dummy run.
    """
    rng = np.random.default_rng(seed)
    y = arrays["y"]
    logits = rng.normal(size=(len(y), K))
    logits[np.arange(len(y)), y] += 4 * skill
    logits[:, 0] += 1.5  # mimic a bias towards the majority class
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    df = pd.DataFrame({"pid": arrays["pid"], "y_true": y, "y_pred": probs.argmax(axis=1)})
    for i, column in enumerate(PROB_COLUMNS):
        df[column] = probs[:, i]
    df["sample"] = arrays["sample"]
    df[DUMMY_COLUMN] = 1
    return df


def write_dummy_results(processed_dir, output_dir, seeds=(0, 1, 2)):
    """Create a fake results tree (E0 random, E1/E2 patient) to exercise the code."""
    output_dir = Path(output_dir)
    plan = {"E0_dummy": ("random_test.npz", 1.0), "E1_dummy": ("patient_test.npz", 0.6),
            "E2_dummy": ("patient_test.npz", 0.7)}
    for exp, (test_file, skill) in plan.items():
        arrays = _cached_split(str(Path(processed_dir) / test_file))
        (output_dir / exp).mkdir(parents=True, exist_ok=True)
        for k in seeds:
            seed = 1000 * len(exp) + 17 * k + int(10 * skill)
            make_dummy_predictions(arrays, seed, skill).to_csv(output_dir / exp / f"seed{k}.csv", index=False)
    print(f"Wrote DUMMY predictions to {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("summarise", help="Evaluate all results/<exp>/seed*.csv")
    run.add_argument("--results-dir", type=Path, default=Path("results"))
    run.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    run.add_argument("--output-dir", type=Path, default=Path("results/summary"))
    run.add_argument("--reference", default="E1", help="experiment for paired differences")
    run.add_argument("--n-boot", type=int, default=2000)
    run.add_argument("--no-figures", action="store_true")
    run.add_argument("--allow-unmatched", action="store_true",
                     help="warn instead of failing when a CSV matches no test NPZ (not for reported results)")
    data = sub.add_parser("describe-data", help="Class concentration, S timing, trivial baseline")
    data.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    data.add_argument("--output-dir", type=Path, default=Path("results/data_analysis"))
    data.add_argument("--raw-dir", type=Path, default=Path("data/raw/mitdb"))
    data.add_argument("--no-figures", action="store_true")
    dummy = sub.add_parser("make-dummy", help="Write fake predictions to test the pipeline")
    dummy.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    dummy.add_argument("--output-dir", type=Path, default=Path("results/_dummy"))
    args = parser.parse_args()
    if args.command == "summarise":
        summarise(args.results_dir, args.processed_dir, args.output_dir, args.reference,
                  args.n_boot, figures=not args.no_figures, allow_unmatched=args.allow_unmatched)
    elif args.command == "describe-data":
        describe_data(args.processed_dir, args.output_dir, args.raw_dir)
        if not args.no_figures:
            from . import figures as fig
            fig.data_figures(args.processed_dir, args.output_dir / "figures")
    else:
        write_dummy_results(args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
