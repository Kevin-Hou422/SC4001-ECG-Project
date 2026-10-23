"""E6: train-only convolutional autoencoder pretraining and strict encoder export.

No validation/test arrays are loaded or used for model selection. Epochs are a
fixed budget. Reconstruction losses are training diagnostics, not classification
or unseen-patient performance. M2 owns the downstream E1/E6 comparison.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import (CLASS_NAMES, DS1, DS2, load_split, sha256, subject_id, write_json)


ARCHITECTURE = {
    "name": "m1_conv_encoder_v1", "input_shape": [1, 256],
    "channels": [16, 32, 64], "kernels": [7, 5, 5], "strides": [2, 2, 2],
    "normalization": "GroupNorm(4,16), GroupNorm(4,32), GroupNorm(8,64)",
    "activation": "ReLU", "output_shape": [64, 32], "embedding_dim": 64,
}


class ECGEncoder(nn.Module):
    """Transfer contract: (B,1,256) -> (B,64,32); embedding() -> (B,64)."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(1, 16, 7, stride=2, padding=3), nn.GroupNorm(4, 16), nn.ReLU(),
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.GroupNorm(8, 64), nn.ReLU(),
        )

    def forward(self, x):
        return self.blocks(x)

    def embedding(self, x):
        return self(x).mean(dim=-1)


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ECGEncoder()
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, 4, stride=2, padding=1),
            nn.GroupNorm(4, 32), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, stride=2, padding=1),
            nn.GroupNorm(4, 16), nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass(frozen=True)
class PretrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    threads: int = 2
    device: str = "cpu"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.threads) < 1 or self.seed < 0:
            raise ValueError("epochs, batch size and threads must be positive; seed non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimiser settings")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")


def read_training_source(path, manifest_path):
    """Reject val/test/random inputs even if renamed; bind arrays to manifest."""
    path, manifest_path = Path(path), Path(manifest_path)
    with np.load(path, allow_pickle=False) as archive:
        partition = str(archive["partition"])
        protocol = str(archive["protocol"])
    if partition != "train" or protocol not in ("patient", "patient_strict"):
        raise ValueError("E6 accepts only patient or patient_strict TRAINING archives")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["protocols"][protocol]
    expected = entry["files"]["train"]
    digest = sha256(path)
    if digest != expected["sha256"]:
        raise ValueError("Training archive does not match its audited manifest")
    arrays = load_split(path)
    records = set(map(int, arrays["pid"]))
    val_records = set(manifest["config"]["val_records"])
    allowed = set(DS1) - val_records
    if protocol == "patient_strict":
        allowed.discard(201)
    if records != allowed or records & (set(DS2) | val_records):
        raise ValueError("Pretraining contains an incorrect record, validation or DS2 data")
    if protocol == "patient_strict":
        test_subjects = {subject_id(r) for r in DS2}
        if set(map(int, arrays["subject_id"])) & test_subjects:
            raise ValueError("Strict pretraining overlaps test subjects")
    identity = {"protocol": protocol, "train_sha256": digest,
                "manifest_sha256": sha256(manifest_path), "n_beats": len(arrays["y"]),
                "records": sorted(records), "normalization": entry["normalization"],
                "pipeline_sha256": manifest["pipeline_sha256"],
                "class_names": list(CLASS_NAMES), "uses_labels_for_optimization": False,
                "uses_validation_or_test_for_optimization": False}
    return arrays["X"], identity


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_atomic(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".part")
    torch.save(value, temporary)
    temporary.replace(path)


def reconstruction_mse(model, loader, device):
    model.eval()
    error, elements = 0.0, 0
    with torch.inference_mode():
        for (x,) in loader:
            x = x.to(device)
            residual = model(x) - x
            error += float(residual.square().sum().item())
            elements += x.numel()
    return error / elements


def pretrain(train_path, manifest_path, output_dir, config=None, resume=False):
    config = config or PretrainConfig()
    # Validate source before creating outputs or constructing the model.
    x, identity = read_training_source(train_path, manifest_path)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError("Output already exists; use --resume or a new directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    dataset = TensorDataset(torch.from_numpy(x))
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                              generator=generator, num_workers=0)
    diagnostic_loader = DataLoader(dataset, batch_size=config.batch_size,
                                   shuffle=False, num_workers=0)
    settings = asdict(config)
    settings["device"] = device
    signature = {"data": identity, "settings": {k: v for k, v in settings.items() if k != "epochs"},
                 "architecture": ARCHITECTURE, "source_sha256": sha256(__file__)}
    history, start_epoch, elapsed_before = [], 0, 0.0
    checkpoint_path = output_dir / "last.pt"
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError("Cannot resume without last.pt")
        previous = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if previous["signature"] != signature:
            raise ValueError("Resume data, settings, architecture or source differ")
        if previous["epoch"] > config.epochs:
            raise ValueError("Target epochs is smaller than the saved epoch")
        model.load_state_dict(previous["model_state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer_state_dict"])
        generator.set_state(previous["loader_rng_state"])
        torch.set_rng_state(previous["torch_rng_state"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(previous["cuda_rng_states"])
        history = previous["history"]
        start_epoch = previous["epoch"]
        elapsed_before = previous["elapsed_seconds"]
        initial_mse = previous["initial_train_mse"]
    else:
        initial_mse = reconstruction_mse(model, diagnostic_loader, device)
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs):
        model.train()
        error, elements = 0.0, 0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = nn.functional.mse_loss(prediction, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reconstruction loss")
            loss.backward()
            optimizer.step()
            error += float(loss.detach()) * batch.numel()
            elements += batch.numel()
        history.append({"epoch": epoch + 1, "train_online_mse": error / elements})
        elapsed = elapsed_before + time.monotonic() - started
        # Save RNG and optimiser state after every epoch for Colab interruptions.
        save_atomic({"signature": signature, "epoch": epoch + 1,
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "loader_rng_state": generator.get_state(),
                     "torch_rng_state": torch.get_rng_state(),
                     "cuda_rng_states": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                     "initial_train_mse": initial_mse, "history": history,
                     "elapsed_seconds": elapsed}, checkpoint_path)
        print(f"{identity['protocol']} seed={config.seed} epoch={epoch+1}/{config.epochs} "
              f"online_train_mse={error / elements:.6f}", flush=True)
    final_mse = reconstruction_mse(model, diagnostic_loader, device)
    encoder_state = {key: value.detach().cpu().clone()
                     for key, value in model.encoder.state_dict().items()}
    export = {"format_version": 1, "architecture": ARCHITECTURE,
              "encoder_state_dict": encoder_state, "seed": config.seed,
              "epochs": config.epochs, "data": identity,
              "encoder_sha256": state_digest(encoder_state)}
    save_atomic(export, output_dir / "encoder.pt")
    summary = {"status": "completed", "config": settings, "data": identity,
               "architecture": ARCHITECTURE,
               "initial_train_mse": initial_mse, "final_train_mse": final_mse,
               "history": history, "selection": "final epoch of fixed budget; no validation/test selection",
               "diagnostic": "training reconstruction only; not classification/generalisation performance",
               "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
               "autoencoder_parameters": sum(p.numel() for p in model.parameters()),
               "encoder_sha256": export["encoder_sha256"],
               "export_file_sha256": sha256(output_dir / "encoder.pt"),
               "source_sha256": signature["source_sha256"],
               "elapsed_seconds": elapsed_before + time.monotonic() - started,
               "software": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": np.__version__, "platform": platform.platform()}}
    write_json(output_dir / "summary.json", summary)
    return summary


def load_pretrained_encoder(checkpoint_path, encoder=None, expected_train_sha256=None):
    """Strict transfer; callers must provide the training archive hash for E6."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported encoder format or architecture")
    if expected_train_sha256 is not None and checkpoint["data"]["train_sha256"] != expected_train_sha256:
        raise ValueError("Encoder was pretrained on a different training archive")
    if state_digest(checkpoint["encoder_state_dict"]) != checkpoint["encoder_sha256"]:
        raise ValueError("Encoder state checksum mismatch")
    encoder = ECGEncoder() if encoder is None else encoder
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
    return encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--protocol", choices=("patient", "patient_strict"), default="patient")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/pretraining"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    for seed in args.seeds:
        config = PretrainConfig(seed, args.epochs, args.batch_size, args.learning_rate,
                                args.weight_decay, args.threads, args.device)
        result = pretrain(args.processed_dir / f"{args.protocol}_train.npz",
                          args.processed_dir / "manifest.json",
                          args.output_dir / args.protocol / f"seed{seed}", config, args.resume)
        print(f"Completed seed {seed}: train reconstruction MSE "
              f"{result['initial_train_mse']:.6f} -> {result['final_train_mse']:.6f}", flush=True)


if __name__ == "__main__":
    main()
