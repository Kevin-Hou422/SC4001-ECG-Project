"""
Run from the repository root:
  python -m src.prepare
  python -m src.train --experiments all --seeds 42 43 44

AFTER model selection is frozen,

Outputs per experiment/seed:
  checkpoints/E1/seed42.pt         best validation-macro-F1 checkpoint
  results/E1/seed42_history.csv    training and validation metrics by epoch
  results/E1/seed42_val.csv        validation probabilities (for diagnostics)
  results/E1/seed42_config.json    configuration and validation performance
  results/E1/seed42.csv            TEST predictions, only in predict-test mode

This script reads src.data.ECGDataset, uses src.models.ECGCNN, and does not
recreate M1's splitting, normalisation, or augmentation logic.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import CLASS_NAMES, ECGDataset, sha256
from .models import ECGCNN

EXPERIMENTS = {
    "E0": dict(split="random", loss="ce", augment=False, use_rr=False),
    "E1": dict(split="patient", loss="ce", augment=False, use_rr=False),
    "E2": dict(split="patient", loss="weighted_ce", augment=False, use_rr=False),
    "E3": dict(split="patient", loss="focal", augment=False, use_rr=False),
    "E4": dict(split="patient", loss="ce", augment=True, use_rr=False),
    "E5": dict(split="patient", loss="ce", augment=False, use_rr=True),
}
PREDICTION_COLUMNS = ["pid", "y_true", "y_pred", "p_N", "p_S", "p_V", "p_F", "sample", "subject_id"]


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN behaviour; some device operations can still vary.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(choice: str):
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but PyTorch was not compiled with CUDA support "
            "or no GPU is available."
        )
    return torch.device(choice)


def make_loss(name: str, train_labels: np.ndarray, device: torch.device, gamma: float):
    if name == "ce":
        return nn.CrossEntropyLoss(), None
    if name == "focal":
        return FocalLoss(gamma), None
    if name == "weighted_ce":
        counts = np.bincount(train_labels.astype(np.int64), minlength=len(CLASS_NAMES))
        if (counts == 0).any():
            raise ValueError("Cannot compute inverse-frequency weights: absent training class")
        # Inverse-frequency weights normalised to mean=1, from TRAIN only.
        weights = (1.0 / counts.astype(np.float64))
        weights /= weights.mean()
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=weight_tensor), weights.tolist()
    raise ValueError(f"Unknown loss: {name}")


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """Macro F1 over the fixed four classes, including zero-prediction classes."""
    scores = []
    for c in range(len(CLASS_NAMES)):
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(scores))


def move_inputs(batch, device: torch.device, use_rr: bool):
    x = batch["X"].to(device=device, dtype=torch.float32)
    rr = batch["rr"].to(device=device, dtype=torch.float32) if use_rr else None
    y = batch["y"].to(device=device, dtype=torch.long)
    return x, rr, y


def run_epoch(model, loader, loss_fn, optimizer, device, use_rr):
    """One training epoch or validation epoch (optimizer=None)."""
    training = optimizer is not None
    model.train(training)
    loss_total, n_total = 0.0, 0
    labels, predictions = [], []
    with torch.set_grad_enabled(training):
        for batch in loader:
            x, rr, y = move_inputs(batch, device, use_rr)
            logits = model(x, rr)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            n = y.size(0)
            loss_total += float(loss.detach().item()) * n
            n_total += n
            labels.append(y.detach().cpu().numpy())
            predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
    if n_total == 0:
        raise ValueError("Encountered empty dataset")
    y_true, y_pred = np.concatenate(labels), np.concatenate(predictions)
    return {"loss": loss_total / n_total,
            "accuracy": float(np.mean(y_true == y_pred)),
            "macro_f1": macro_f1(y_true, y_pred)}


@torch.inference_mode()
def predict_rows(model, loader, device, use_rr):
    """Emit rows in the same order as the input NPZ (shuffle=False required)."""
    model.eval()
    rows = []
    for batch in loader:
        x, rr, y = move_inputs(batch, device, use_rr)
        probabilities = torch.softmax(model(x, rr), dim=1).cpu().numpy()
        labels = y.cpu().numpy()
        pids = batch["pid"].numpy()
        samples = batch["sample"].numpy()
        subjects = batch["subject_id"].numpy()
        for i in range(len(labels)):
            row = {
                "pid": int(pids[i]), "y_true": int(labels[i]),
                "y_pred": int(probabilities[i].argmax()),
                "sample": int(samples[i]), "subject_id": int(subjects[i]),
            }
            row.update({f"p_{c}": float(probabilities[i, j]) for j, c in enumerate(CLASS_NAMES)})
            rows.append(row)
    return rows


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def npz_path(args, protocol: str, partition: str):
    path = args.data_dir / f"{protocol}_{partition}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run 'python -m src.prepare' from the repo root first."
        )
    return path


def get_protocol(exp: str, args):
    return "random" if exp == "E0" else args.patient_protocol


def get_loaders(args, protocol: str, seed: int, augment: bool, *, test_only=False):
    """DataLoader(num_workers=0) so M1's per-epoch augmentation is reliable."""
    common = dict(batch_size=args.batch_size, num_workers=0,
                  pin_memory=(args.device_resolved.type == "cuda"))
    if test_only:
        test = ECGDataset(npz_path(args, protocol, "test"), augment=False)
        return test, DataLoader(test, shuffle=False, **common)

    train = ECGDataset(npz_path(args, protocol, "train"), augment=augment, seed=seed)
    val = ECGDataset(npz_path(args, protocol, "val"), augment=False)
    if train.protocol != protocol or val.protocol != protocol:
        raise ValueError("Processed files do not match selected protocol")
    if train.partition != "train" or val.partition != "val":
        raise ValueError("Processed files do not match selected partitions")
    if protocol == "patient_strict":
        shared = set(train.arrays["subject_id"]) & set(val.arrays["subject_id"])
        if shared:
            raise ValueError(f"Strict patient split has overlapping subjects: {shared}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(train, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val, shuffle=False, **common)
    return train, train_loader, val_loader


def train_one(exp: str, seed: int, args):
    config = EXPERIMENTS[exp]
    protocol = get_protocol(exp, args)
    set_seed(seed)
    train_data, train_loader, val_loader = get_loaders(
        args, protocol, seed, augment=config["augment"])
    model = ECGCNN(use_rr=config["use_rr"]).to(args.device_resolved)
    loss_fn, class_weights = make_loss(
        config["loss"], train_data.arrays["y"], args.device_resolved, args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = args.checkpoints_dir / exp / f"seed{seed}.pt"
    output_dir = args.results_dir / exp
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "experiment": exp, "seed": seed, "protocol": protocol,
        "use_rr": config["use_rr"], "loss": config["loss"],
        "augmentation": config["augment"], "class_weights": class_weights,
        "train_file": str(npz_path(args, protocol, "train")),
        "train_sha256": sha256(npz_path(args, protocol, "train")),
        "val_file": str(npz_path(args, protocol, "val")),
        "val_sha256": sha256(npz_path(args, protocol, "val")),
        "epochs_limit": args.epochs, "patience": args.patience,
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "focal_gamma": args.focal_gamma,
        "device": str(args.device_resolved),
        "classes": list(CLASS_NAMES),
    }

    best_f1, best_epoch, stale = -1.0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_data.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, loss_fn, optimizer,
                                  args.device_resolved, config["use_rr"])
        val_metrics = run_epoch(model, val_loader, loss_fn, None,
                                args.device_resolved, config["use_rr"])
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        })
        score = val_metrics["macro_f1"]
        if score > best_f1 + 1e-8:
            best_f1, best_epoch, stale = score, epoch, 0
            # Store CPU tensors for portable Windows/Colab/CPU/Mac loading.
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": state, **identity,
                        "best_epoch": best_epoch, "best_val_macro_f1": best_f1},
                       checkpoint_path)
        else:
            stale += 1
        print(f"{exp} seed={seed} epoch {epoch:02d}: "
              f"train_loss={train_metrics['loss']:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"val_macroF1={score:.4f} (best={best_f1:.4f})", flush=True)
        if stale >= args.patience:
            print(f"Early stopping after {epoch} epochs; best epoch={best_epoch}", flush=True)
            break

    save_csv(output_dir / f"seed{seed}_history.csv", history, list(history[0]))
    # Load best model, not final epoch, for the exported validation predictions.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    model.to(args.device_resolved)
    val_predictions = predict_rows(model, val_loader, args.device_resolved, config["use_rr"])
    save_csv(output_dir / f"seed{seed}_val.csv", val_predictions, PREDICTION_COLUMNS)
    save_json(output_dir / f"seed{seed}_config.json", {
        **identity, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "checkpoint": str(checkpoint_path),
        "prediction_file": str(output_dir / f"seed{seed}_val.csv"),
    })
    print(f"Saved best model -> {checkpoint_path}")
    print(f"Saved validation predictions -> {output_dir / f'seed{seed}_val.csv'}")
    return {"experiment": exp, "seed": seed, "protocol": protocol,
            "best_epoch": best_epoch, "best_val_macro_f1": best_f1}


def export_test(exp: str, seed: int, args):
    """Final-only, frozen-model inference; does NOT retrain or tune on test."""
    protocol = get_protocol(exp, args)
    ckpt_path = args.checkpoints_dir / exp / f"seed{seed}.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing best model: {ckpt_path}. Run training first.")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if checkpoint["experiment"] != exp or checkpoint["seed"] != seed:
        raise ValueError("Checkpoint experiment/seed mismatch")
    if checkpoint["protocol"] != protocol:
        raise ValueError("Checkpoint patient protocol differs from requested protocol")
    expected_rr = EXPERIMENTS[exp]["use_rr"]
    if checkpoint["use_rr"] != expected_rr:
        raise ValueError("Checkpoint architecture (use_rr) mismatch")
    current_train = npz_path(args, protocol, "train")
    if sha256(current_train) != checkpoint["train_sha256"]:
        raise ValueError("Checkpoint training archive differs from current training archive")
    test, test_loader = get_loaders(args, protocol, seed, augment=False, test_only=True)
    if test.protocol != protocol or test.partition != "test":
        raise ValueError("Invalid test archive metadata")
    model = ECGCNN(use_rr=expected_rr)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device_resolved)
    rows = predict_rows(model, test_loader, args.device_resolved, expected_rr)
    output = args.results_dir / exp / f"seed{seed}.csv"
    save_csv(output, rows, PREDICTION_COLUMNS)
    print(f"Final test predictions: {output} ({len(rows)} beats)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", choices=["train", "predict-test"], default="train")
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=[*EXPERIMENTS, "all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--patient-protocol", choices=["patient", "patient_strict"],
                        default="patient_strict", help="E1-E5 patient split; E0 always uses random")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument( "--device",default="cuda",choices=["cuda"],help="Use NVIDIA GPU (CUDA required)")
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if "all" in args.experiments and len(args.experiments) != 1:
        parser.error("Use --experiments all OR list individual experiments")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Seeds must be distinct non-negative integers")
    if min(args.epochs, args.patience, args.batch_size, args.cpu_threads) < 1:
        parser.error("epochs, patience, batch-size and cpu-threads must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.focal_gamma < 0:
        parser.error("lr must be positive; weight-decay and focal-gamma must be nonnegative")
    torch.set_num_threads(args.cpu_threads)
    args.device_resolved = resolve_device(args.device)
    exps = list(EXPERIMENTS) if args.experiments == ["all"] else args.experiments
    print(f"Device={args.device_resolved} | mode={args.mode} | "
          f"experiments={exps} | seeds={args.seeds} | patient protocol={args.patient_protocol}", flush=True)
    summary = []
    for exp in exps:
        for seed in args.seeds:
            if args.mode == "train":
                summary.append(train_one(exp, seed, args))
            else:
                export_test(exp, seed, args)
    if summary:
        path = args.results_dir / "training_summary.csv"
        save_csv(path, summary, list(summary[0]))
        print(f"Training summary -> {path}")
        for exp in exps:
            vals = [x["best_val_macro_f1"] for x in summary if x["experiment"] == exp]
            print(f"{exp}: best validation macro-F1 = {np.mean(vals):.4f} ± "
                  f"{np.std(vals, ddof=1) if len(vals)>1 else 0.:.4f} across {len(vals)} seed(s)")


if __name__ == "__main__":
    main()
