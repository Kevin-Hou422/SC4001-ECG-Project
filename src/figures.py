"""Report figures for M3 (matplotlib, static PNG + PDF).

Style: one fixed colour per heartbeat class (N blue, S orange, V aqua, F yellow,
validated for colour-vision deficiency on adjacent pairs), a sequential blue
ramp for heatmaps, hairline solid grids and recessive axes. Sized for an A4
single-column report (text width about 6.3 in) at 8 pt, so the figures are
legible next to Arial 10 pt body text. Light aqua/yellow marks sit below 3:1
contrast on white, so every figure is paired with direct labels or a table.

Every figure built from predictions carrying the ``dummy`` column is stamped
"DUMMY DATA - NOT RESULTS".
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

from .data import CLASS_NAMES, DS2, load_split  # noqa: E402


CLASS_COLORS = {"N": "#2a78d6", "S": "#eb6834", "V": "#1baf7a", "F": "#eda100"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE_RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
CRITICAL = "#d03b3b"
TEXT_WIDTH = 6.3  # inches
FORMATS = ("png", "pdf")
FS = 360
RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
}


def _save(fig, path, dummy=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if dummy:
        fig.text(0.5, 0.5, "DUMMY DATA - NOT RESULTS", ha="center", va="center",
                 rotation=20, fontsize=22, color=CRITICAL, alpha=0.35, weight="bold")
    for fmt in FORMATS:
        fig.savefig(path.with_suffix(f".{fmt}"))
    plt.close(fig)


def _text_on(value, threshold=0.55):
    """White text on dark cells, ink on light cells."""
    return "white" if value >= threshold else INK


# ---------------------------------------------------------------- experiments

def plot_confusion(cm, title, path, dummy=False):
    """Row-normalised confusion matrix; cells show % of the true class and beats."""
    cm = np.asarray(cm, dtype=np.float64)
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(3.0, 2.75))
        ax.imshow(share, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, f"{100 * share[i, j]:.1f}%\n{cm[i, j]:,.0f}", ha="center",
                        va="center", fontsize=7, color=_text_on(share[i, j]))
        ax.set_xticks(range(len(cm)), CLASS_NAMES)
        ax.set_yticks(range(len(cm)), [f"{c}  (n={int(round(r)):,})" for c, r in zip(CLASS_NAMES, rows[:, 0])])
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _save(fig, path, dummy)


def plot_comparison(summary, path, metrics=("macro_f1", "S_f1", "V_f1", "F_f1"),
                    subset="all", dummy=False):
    """Small multiples: mean ± SD over seeds for each metric and experiment."""
    from .evaluate import LABELS
    table = summary[summary["subset"] == subset].reset_index(drop=True)
    names = [f"{r.experiment} (random split)" if r.test == "random_test.npz" else r.experiment
             for r in table.itertuples()]
    y = np.arange(len(table))[::-1]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(metrics), figsize=(TEXT_WIDTH, 0.35 * len(table) + 1.0),
                                 sharey=True, layout="constrained")
        for ax, metric in zip(np.atleast_1d(axes), metrics):
            mean = table[f"{metric}_mean"].to_numpy(dtype=float)
            std = table[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.errorbar(mean, y, xerr=std, fmt="o", color=CLASS_COLORS["N"], ms=5,
                        mec="white", mew=1, elinewidth=1.2, capsize=0)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_ylim(-0.6, len(table) - 0.4)
            ax.set_title(LABELS.get(metric, metric))
            ax.spines["left"].set_visible(False)
        np.atleast_1d(axes)[0].set_yticks(y, names)
        fig.supxlabel("Score (mean ± SD over seeds)", fontsize=8, color=INK_2)
        _save(fig, path, dummy)


def plot_per_patient(tables, cls, path, dummy=False):
    """Heatmap of per-record recall for one class: experiments x test records.

    Only records that contain the class are shown; columns are sorted by support
    and labelled with it, so a 100% recall on 1 beat is not mistaken for evidence.
    """
    first = next(iter(tables.values()))
    present = first[first[f"{cls}_support"] > 0].sort_values(f"{cls}_support", ascending=False)
    records = present["pid"].tolist()
    support = present[f"{cls}_support"].tolist()
    values = np.array([[t.set_index("pid").loc[r, f"{cls}_recall"] for r in records]
                       for t in tables.values()])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(min(TEXT_WIDTH, 0.42 * len(records) + 1.2),
                                        0.32 * len(tables) + 1.0))
        ax.imshow(values, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color=_text_on(values[i, j]))
        ax.set_xticks(range(len(records)), [f"{r}\n{n:,}" for r, n in zip(records, support)],
                      fontsize=6.5)
        ax.set_yticks(range(len(tables)), list(tables))
        ax.set_xlabel(f"Test record and number of {cls} beats (sorted by count)")
        ax.set_title(f"{cls} recall per test record (mean over seeds)")
        for spine in ax.spines.values():
            spine.set_visible(False)
        _save(fig, path, dummy)


def plot_bootstrap(boot, path, dummy=False):
    """Paired patient-bootstrap differences against the reference experiment."""
    from .evaluate import LABELS
    table = boot.dropna(subset=["diff"]) if "diff" in boot.columns else boot.iloc[0:0]
    if table.empty:
        return
    metrics = list(dict.fromkeys(table["metric"]))
    reference = table["reference"].iloc[0]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(metrics), sharey=True, layout="constrained",
                                 figsize=(TEXT_WIDTH, 0.35 * table["experiment"].nunique() + 1.0))
        for ax, metric in zip(np.atleast_1d(axes), metrics):
            part = table[table["metric"] == metric].reset_index(drop=True)
            diff = part["diff"].to_numpy(dtype=float)
            low = part["diff_low"].to_numpy(dtype=float)
            high = part["diff_high"].to_numpy(dtype=float)
            y = np.arange(len(part))[::-1]
            ax.axvline(0, color=AXIS, lw=0.8)
            ax.grid(axis="x")
            ax.set_axisbelow(True)
            ax.errorbar(diff, y, xerr=[diff - low, high - diff], fmt="o",
                        color=CLASS_COLORS["N"], ms=5, mec="white", mew=1,
                        elinewidth=1.2, capsize=0)
            ax.set_title(f"Δ {LABELS.get(metric, metric)}")
            ax.spines["left"].set_visible(False)
            ax.set_yticks(y, part["experiment"])
            ax.set_ylim(-0.6, len(part) - 0.4)
            ax.xaxis.set_major_locator(MaxNLocator(3))
        fig.supxlabel(f"Difference from {reference} with 95% patient-bootstrap interval",
                      fontsize=8, color=INK_2)
        _save(fig, path, dummy)


def plot_training_curves(results_dir, path, highlight=("E0", "E1")):
    """Validation loss and macro-F1 per epoch from M2's ``seed<k>_history.csv``.

    The ``highlight`` experiments get colour, the rest are thin grey lines; the
    dot marks the epoch early stopping kept (best validation macro-F1). The loss
    panel shows only ``highlight``: weighted and focal losses (E2, E3) are on
    other scales. Returns False when no history files exist.
    """
    import pandas as pd
    histories = sorted(Path(results_dir).glob("E*/seed*_history.csv"))
    if not histories:
        return False
    colors = dict(zip(highlight, (CLASS_COLORS["N"], CLASS_COLORS["S"])))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.3), layout="constrained")
        for path_csv in histories:
            exp = path_csv.parent.name
            h = pd.read_csv(path_csv)
            best = int(h["val_macro_f1"].idxmax())
            style = (dict(color=colors[exp], lw=1.4, alpha=0.9, zorder=3) if exp in colors
                     else dict(color=AXIS, lw=0.9, zorder=2))
            for ax, column in zip(axes, ("val_loss", "val_macro_f1")):
                if column == "val_loss" and exp not in colors:
                    continue
                ax.plot(h["epoch"], h[column], **style)
                ax.plot(h["epoch"][best], h[column][best], "o", ms=3.5, mec="white", mew=0.6,
                        color=style["color"], zorder=style["zorder"] + 1)
        for ax, title in zip(axes, ("Validation loss (cross-entropy)", "Validation macro-F1")):
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        labels = [f"{exp} ({'random' if exp == 'E0' else 'patient'} split)" for exp in highlight]
        handles = [plt.Line2D([], [], color=colors[e], lw=1.4) for e in highlight]
        handles.append(plt.Line2D([], [], color=AXIS, lw=0.9))
        labels.append("E2–E5 (patient split)")
        axes[1].legend(handles, labels, loc="lower right")
        _save(fig, path)
    return True


def experiment_figures(experiments, summary, boot, out_dir, reference="E1", dummy=False):
    from .evaluate import experiment_confusions, per_patient_mean
    out_dir = Path(out_dir)
    for e in experiments:
        _, cms = experiment_confusions(e)
        mean_cm = cms.sum(axis=1).mean(axis=0)
        plot_confusion(mean_cm, f"{e['name']}: mean over {cms.shape[0]} seed(s)",
                       out_dir / f"confusion_{e['name']}", dummy)
    plot_comparison(summary, out_dir / "comparison_all", dummy=dummy)
    plot_comparison(summary, out_dir / "comparison_without_232",
                    metrics=("macro_f1", "S_recall", "S_precision", "S_f1"),
                    subset="without_232", dummy=dummy)
    patient = {e["name"]: per_patient_mean(e["seeds"]) for e in experiments
               if set(next(iter(e["seeds"].values()))["pid"].unique()) == set(DS2)}
    if patient:
        for cls in CLASS_NAMES[1:]:
            plot_per_patient(patient, cls, out_dir / f"per_patient_{cls}", dummy)
    plot_bootstrap(boot, out_dir / "bootstrap_vs_reference", dummy)


# ---------------------------------------------------------------- dataset

def _concentration(y, pid, cls_index):
    records, counts = np.unique(pid[y == cls_index], return_counts=True)
    order = np.argsort(-counts, kind="stable")
    return records[order], np.cumsum(counts[order]) / counts.sum()


def plot_concentration(splits, path):
    """Cumulative share of each abnormal class held by the top-k records."""
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(splits), figsize=(TEXT_WIDTH, 2.3), sharey=True)
        for ax, (title, arrays) in zip(np.atleast_1d(axes), splits.items()):
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            for i, cls in enumerate(CLASS_NAMES[1:], start=1):
                records, share = _concentration(arrays["y"], arrays["pid"], i)
                k = np.arange(1, len(share) + 1)
                ax.plot(k, 100 * share, color=CLASS_COLORS[cls], lw=1.4, marker="o", ms=4,
                        mec="white", mew=0.8,
                        label=f"{cls}: top record {records[0]} holds {100 * share[0]:.0f}%")
            ax.set_title(title)
            ax.set_xlabel("Number of records, largest contributor first")
            ax.set_ylim(0, 102)
            ax.set_xlim(0.5, None)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc="lower right", handlelength=1.6)
        np.atleast_1d(axes)[0].set_ylabel("Cumulative share of class beats (%)")
        _save(fig, path)


def plot_rr_prematurity(splits, path, min_beats=20):
    """Pre-RR / local mean RR of S beats per record against the N-beat range."""
    groups = []
    for title, arrays in splits.items():
        s = arrays["y"] == 1
        records, counts = np.unique(arrays["pid"][s], return_counts=True)
        keep = [(r, n) for r, n in sorted(zip(records, counts), key=lambda t: -t[1]) if n >= min_beats]
        groups.append((title, [(r, n, arrays["rr_raw"][s & (arrays["pid"] == r), 3]) for r, n in keep]))
    all_n = np.concatenate([a["rr_raw"][a["y"] == 0, 3] for a in splits.values()])
    low, high = np.percentile(all_n, [10, 90])
    labels, data, positions, pos, section = [], [], [], 0, []
    for title, items in groups:
        section.append((pos, title))
        for record, n, values in items:
            labels.append(f"{record} (n={n:,})")
            data.append(values)
            positions.append(pos)
            pos += 1
        pos += 0.8
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.62, 0.22 * pos + 0.8))
        ax.axvspan(low, high, color=CLASS_COLORS["N"], alpha=0.10, lw=0)
        ax.text((low + high) / 2, -1.0, "N beats\n10-90th pct", fontsize=7, color=INK_2,
                ha="center", va="center")
        ax.axvline(1.0, color=AXIS, lw=0.8)
        ax.boxplot(data, positions=positions, vert=False, widths=0.55, whis=(10, 90),
                   showfliers=False, patch_artist=True,
                   boxprops={"facecolor": CLASS_COLORS["S"], "edgecolor": "white", "lw": 1},
                   medianprops={"color": "white", "lw": 1.5},
                   whiskerprops={"color": CLASS_COLORS["S"], "lw": 1.2},
                   capprops={"lw": 0})
        ax.set_yticks(positions, labels)
        for start, title in section:
            ax.text(-0.02, start - 0.75, title, transform=ax.get_yaxis_transform(),
                    ha="right", fontsize=7.5, weight="bold", color=INK)
        ax.set_ylim(pos - 0.5, -1.7)
        ax.set_xlim(0.3, 1.3)
        ax.set_xlabel("Previous RR / mean of last 10 RR (S beats; <1 = premature)")
        ax.set_title("Timing of S beats varies by patient")
        ax.spines["left"].set_visible(False)
        _save(fig, path)


def plot_templates(train, test, path, record=232):
    """Median waveform with interquartile band for N and S beats."""
    t_ms = (np.arange(256) - 128) / FS * 1000
    panels = {"Training records (pooled)": train,
              f"Test record {record}": {k: v[test["pid"] == record] for k, v in test.items()}}
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.1), sharey=True)
        for ax, (title, arrays) in zip(axes, panels.items()):
            ax.grid(axis="y")
            ax.set_axisbelow(True)
            for i, cls in ((0, "N"), (1, "S")):
                beats = arrays["X"][arrays["y"] == i, 0]
                q25, median, q75 = np.percentile(beats, [25, 50, 75], axis=0)
                ax.fill_between(t_ms, q25, q75, color=CLASS_COLORS[cls], alpha=0.12, lw=0)
                ax.plot(t_ms, median, color=CLASS_COLORS[cls], lw=1.3,
                        label=f"{cls} median (n={len(beats):,})")
            ax.set_title(title)
            ax.set_xlabel("Time from R peak (ms)")
            ax.legend(loc="upper right")
        axes[0].set_ylabel("Amplitude (standardised)")
        _save(fig, path)


def data_figures(processed_dir, out_dir, protocol="patient_strict"):
    """Figures describing the labelled data (not model results).

    ``protocol`` picks the training records: ``patient_strict`` (main, E1-E5)
    leaves out record 201; validation and test records are the same in both.
    """
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    train = load_split(processed_dir / f"{protocol}_train.npz")
    val = load_split(processed_dir / f"{protocol}_val.npz")
    test = load_split(processed_dir / f"{protocol}_test.npz")
    n_train, n_test = len(np.unique(train["pid"])), len(np.unique(test["pid"]))
    plot_concentration({f"Training records ({n_train})": train,
                        f"Test records ({n_test}, DS2)": test}, out_dir / "class_concentration")
    plot_rr_prematurity({"Training": train, "Validation": val, "Test": test},
                        out_dir / "rr_prematurity_S")
    plot_templates(train, test, out_dir / "templates_232")
    print(f"Wrote data figures to {out_dir}", flush=True)
