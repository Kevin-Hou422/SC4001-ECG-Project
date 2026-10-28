"""M3: export test predictions from M2's frozen checkpoints, on CPU or GPU.

Same output as M2's ``python -m src.train --mode predict-test`` (it reuses
M2's experiment table, ``predict_rows`` and CSV columns), with two differences:

- It runs on CPU when no GPU is available. Inference is fast on CPU.
- M2's check needs the training NPZ to be byte-identical to the one used in
  training. NPZ bytes depend on the numpy/zlib build, so a rebuild of the same
  data on another machine fails it. Here the check is behavioural instead:
  every checkpoint must reproduce its stored best validation macro-F1 on the
  local validation NPZ. A match on ~9.6k validation beats is only plausible
  if the arrays and the train-fitted normalisation are the same.

Nothing is trained or tuned: the test set is read once, after the check.
Run from the repository root:
  python -m src.predict --experiments all --seeds 42 43 44
Writes results/<exp>/seed<k>.csv (test), results/<exp>/seed<k>_val.csv
(validation, as training also writes) and results/export_check.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import train as m2
from .data import CLASS_NAMES, ECGDataset, sha256
from .models import ECGCNN

CHECK_COLUMNS = ["experiment", "seed", "protocol", "loss", "augmentation", "use_rr",
                 "best_epoch", "stored_val_macro_f1", "local_val_macro_f1", "abs_diff",
                 "same_train_bytes", "test_file", "test_beats",
                 *[f"{part}_recall_{c}" for part in ("train", "val") for c in CLASS_NAMES]]


def recalls(rows, part):
    """Per-class recall of the selected checkpoint on training or validation beats.

    Shows whether a class missed on new patients was learned on the training
    patients at all (diagnostic only; uses no test data).
    """
    y_true = np.array([r["y_true"] for r in rows])
    y_pred = np.array([r["y_pred"] for r in rows])
    return {f"{part}_recall_{c}": float(np.mean(y_pred[y_true == i] == i)) if (y_true == i).any()
            else float("nan") for i, c in enumerate(CLASS_NAMES)}


def load_checkpoint(exp: str, seed: int, args):
    """Load one checkpoint and check that it is the experiment it claims to be."""
    path = args.checkpoints_dir / exp / f"seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = m2.EXPERIMENTS[exp]
    if ckpt["experiment"] != exp or ckpt["seed"] != seed:
        raise ValueError(f"{path}: experiment/seed mismatch")
    for key, expected in (("loss", config["loss"]), ("augmentation", config["augment"]),
                          ("use_rr", config["use_rr"])):
        if ckpt[key] != expected:
            raise ValueError(f"{path}: {key}={ckpt[key]!r}, expected {expected!r}")
    expected_split = "random" if config["split"] == "random" else ("patient", "patient_strict")
    if ckpt["protocol"] not in np.atleast_1d(expected_split):
        raise ValueError(f"{path}: protocol {ckpt['protocol']} does not fit {exp}")
    return ckpt


def loader(path: Path, args):
    data = ECGDataset(path, augment=False)
    return data, DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)


def export_one(exp: str, seed: int, args):
    ckpt = load_checkpoint(exp, seed, args)
    protocol, use_rr = ckpt["protocol"], ckpt["use_rr"]
    model = ECGCNN(use_rr=use_rr)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device_resolved)

    # Behavioural data check on validation data, before the test set is touched.
    val, val_loader = loader(m2.npz_path(args, protocol, "val"), args)
    rows = m2.predict_rows(model, val_loader, args.device_resolved, use_rr)
    m2.save_csv(args.results_dir / exp / f"seed{seed}_val.csv", rows, m2.PREDICTION_COLUMNS)
    local_f1 = m2.macro_f1(np.array([r["y_true"] for r in rows]),
                           np.array([r["y_pred"] for r in rows]))
    stored_f1 = float(ckpt["best_val_macro_f1"])
    diff = abs(local_f1 - stored_f1)
    same_bytes = sha256(m2.npz_path(args, protocol, "train")) == ckpt["train_sha256"]
    if diff > args.tolerance:
        raise ValueError(f"{exp} seed{seed}: validation macro-F1 {local_f1:.6f} differs "
                         f"from the stored {stored_f1:.6f}; local data may not match "
                         "the data used for training")
    fit = recalls(rows, "val")
    _, train_loader = loader(m2.npz_path(args, protocol, "train"), args)
    fit.update(recalls(m2.predict_rows(model, train_loader, args.device_resolved, use_rr), "train"))

    test, test_loader = loader(m2.npz_path(args, protocol, "test"), args)
    if test.protocol != protocol or test.partition != "test":
        raise ValueError(f"{test.path}: invalid test archive metadata")
    rows = m2.predict_rows(model, test_loader, args.device_resolved, use_rr)
    output = args.results_dir / exp / f"seed{seed}.csv"
    m2.save_csv(output, rows, m2.PREDICTION_COLUMNS)
    print(f"{exp} seed{seed} [{protocol}]: val macro-F1 {local_f1:.4f} "
          f"(stored {stored_f1:.4f}) -> {output} ({len(rows)} beats)", flush=True)
    return {"experiment": exp, "seed": seed, "protocol": protocol, "loss": ckpt["loss"],
            "augmentation": ckpt["augmentation"], "use_rr": use_rr,
            "best_epoch": ckpt["best_epoch"], "stored_val_macro_f1": stored_f1,
            "local_val_macro_f1": local_f1, "abs_diff": diff,
            "same_train_bytes": same_bytes, "test_file": test.path.name,
            "test_beats": len(rows), **fit}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        choices=[*m2.EXPERIMENTS, "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help="allowed |local - stored| validation macro-F1 (GPU vs CPU)")
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device_resolved = torch.device(args.device)
    exps = list(m2.EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    checks = [export_one(exp, seed, args) for exp in exps for seed in args.seeds]
    m2.save_csv(args.results_dir / "export_check.csv", checks, CHECK_COLUMNS)
    print(f"Checks -> {args.results_dir / 'export_check.csv'}")


if __name__ == "__main__":
    main()
