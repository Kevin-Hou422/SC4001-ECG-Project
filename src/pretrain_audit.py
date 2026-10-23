"""Independent checks for the published E6 encoder exports (no classifier fitting)."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import sha256, write_json
from .pretrain import (ARCHITECTURE, load_pretrained_encoder, read_training_source,
                       state_digest)


def audit_export(folder, train_path, manifest_path, expected_seed=None, expected_epochs=None):
    folder = Path(folder)
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "completed":
        raise ValueError("Pretraining run is not completed")
    if expected_seed is not None and summary["config"]["seed"] != expected_seed:
        raise ValueError("Wrong pretraining seed")
    epochs = summary["config"]["epochs"]
    if expected_epochs is not None and epochs != expected_epochs:
        raise ValueError("Wrong fixed epoch budget")
    if [row["epoch"] for row in summary["history"]] != list(range(1, epochs + 1)):
        raise ValueError("Incomplete or misordered training history")
    losses = [summary["initial_train_mse"], summary["final_train_mse"]]
    losses += [row["train_online_mse"] for row in summary["history"]]
    if not np.isfinite(losses).all() or np.any(np.array(losses) < 0):
        raise ValueError("Invalid reconstruction diagnostics")
    x, identity = read_training_source(train_path, manifest_path)
    stored = summary["data"]
    for key in ("protocol", "n_beats", "records", "pipeline_sha256",
                "uses_labels_for_optimization", "uses_validation_or_test_for_optimization"):
        if stored[key] != identity[key]:
            raise ValueError(f"Training data semantics mismatch: {key}")
    for key in ("x_mean", "x_std", "rr_mean", "rr_std"):
        if not np.allclose(stored["normalization"][key], identity["normalization"][key],
                           rtol=1e-8, atol=1e-10):
            raise ValueError(f"Training normalization mismatch: {key}")
    export_path = folder / "encoder.pt"
    if sha256(export_path) != summary["export_file_sha256"]:
        raise ValueError("Encoder file checksum mismatch")
    checkpoint = torch.load(export_path, map_location="cpu", weights_only=True)
    if checkpoint["data"] != stored or checkpoint["architecture"] != ARCHITECTURE:
        raise ValueError("Export metadata mismatch")
    if checkpoint["seed"] != summary["config"]["seed"] or checkpoint["epochs"] != epochs:
        raise ValueError("Export seed or epoch metadata mismatch")
    encoder = load_pretrained_encoder(export_path)
    if state_digest(encoder.state_dict()) != summary["encoder_sha256"]:
        raise ValueError("Encoder tensor checksum mismatch")
    encoder.eval()
    with torch.inference_mode():
        output = encoder(torch.from_numpy(x[:4]))
    if output.shape != (4, 64, 32) or not torch.isfinite(output).all():
        raise ValueError("Encoder forward interface failed")
    # Rebuilding NPZs under another package/platform may change container bytes or
    # last-bit reductions. Match protocol, source pipeline and fitted statistics
    # above; report the exact file-hash comparison rather than hiding a mismatch.
    return {"status": "passed", "protocol": stored["protocol"],
            "seed": summary["config"]["seed"], "epochs": epochs,
            "n_train_beats": stored["n_beats"],
            "exact_training_npz_sha256_match": stored["train_sha256"] == identity["train_sha256"],
            "training_semantics_and_normalization_match": True,
            "source_matches_current_pretrain": summary["source_sha256"] == sha256(Path(__file__).with_name("pretrain.py")),
            "encoder_sha256": summary["encoder_sha256"],
            "classification_evaluated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/pretraining"))
    parser.add_argument("--protocols", nargs="+", default=["patient", "patient_strict"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("docs/pretraining_validation.json"))
    args = parser.parse_args()
    rows = []
    for protocol in args.protocols:
        for seed in args.seeds:
            result = audit_export(args.artifact_dir / protocol / f"seed{seed}",
                                  args.processed_dir / f"{protocol}_train.npz",
                                  args.processed_dir / "manifest.json", seed, args.epochs)
            if not result["source_matches_current_pretrain"]:
                raise ValueError("Export came from a different pretraining source")
            rows.append(result)
            print(f"Audited {protocol} seed {seed}: passed", flush=True)
    write_json(args.output, {"status": "passed", "runs": rows,
                             "classification_evaluated": False})


if __name__ == "__main__":
    main()
