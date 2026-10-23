# E6 autoencoder pretraining and M2 transfer contract

M1 provides executable unsupervised pretraining and six audited encoder exports:
two patient protocols, each using seeds 42, 43 and 44. Each run uses **all** of that
protocol's training beats for a fixed ten epochs. It never reads validation/test
arrays, consumes no class labels in the optimiser, and does not select an epoch
using validation or test performance. Training reconstruction MSE is a diagnostic;
it does not establish an S-class improvement or an unseen-patient result.

## Architecture and optimisation

- Encoder: Conv1d 1->16 (kernel 7, stride 2, padding 3), then 16->32 and 32->64
  (kernel 5, stride 2, padding 2), each with GroupNorm and ReLU.
- GroupNorm groups: 4, 4 and 8. No running normalisation statistics are fitted on
  validation/test data. Encoder output is `(batch, 64, 32)`; temporal averaging
  supplies a 64-dimensional embedding.
- Decoder: transposed convolutions 64->32->16->1 with kernel 4, stride 2,
  padding 1; GroupNorm/ReLU after the first two; final output is linear.
- Objective: mean squared reconstruction error on standardised waveforms.
- AdamW, learning rate 0.001, weight decay 0.0001, batch size 256, ten epochs,
  two CPU threads for the recorded runs, deterministic algorithms enabled.
- No augmentation, resampling or class balancing is added during pretraining.
  The majority class therefore dominates the reconstruction objective; supervised
  minority-class benefit remains a question for E6's downstream comparison.

The architecture is a project implementation. No particular published
autoencoder's architecture or numerical results are claimed as reproduced.

## Reproduce

Install a suitable PyTorch build, or use Colab's existing build:

```bash
python -m pip install -r requirements-pretrain.txt
python -m src.prepare
python -m src.pretrain --protocol patient --seeds 42 43 44 --epochs 10
python -m src.pretrain --protocol patient_strict --seeds 42 43 44 --epochs 10
python -m src.pretrain_audit
```

Existing output directories are never silently overwritten. Use a new
`--output-dir` for an independent rerun, or add `--resume` to continue from
`last.pt`. The local run's `last.pt` stores model, optimiser, epoch, loader RNG
and framework RNG state after every epoch. Resume checks the data, settings,
architecture and source. A unit test confirms that a one-epoch interruption plus
resume to epoch two exactly matches uninterrupted two-epoch training on CPU.
Different software/hardware need not give bitwise-identical learned weights.

`encoder.pt` and `summary.json` are published; mutable optimiser checkpoints
`last.pt` remain untracked. In Colab, choose an output directory under mounted
Google Drive **before starting training** to persist epoch checkpoints. User
authentication/Drive consent is handled interactively by Google, not by this code.

## Transfer into M2's classifier

The repository currently has no M2 `src/models.py`. This module defines a clear
encoder interface without replacing M2's classifier/training files. For E1 versus
E6 to isolate pretraining, M2 must use the **same encoder and classifier structure**
in both runs, with the only difference being random versus pretrained encoder
initialisation. If M2 chooses another encoder, align the autoencoder with that
architecture before claiming the comparison isolates pretraining. Do not load
partial keys with `strict=False` or compare different architectures as E1/E6.

```python
from pathlib import Path
import torch
from torch import nn
from src.data import sha256
from src.pretrain import ECGEncoder, load_pretrained_encoder

protocol, seed = 'patient', 42
train_path = Path(f'data/processed/{protocol}_train.npz')
torch.manual_seed(seed)

# E1: ECGEncoder() initialised from scratch.
# E6: initialise the identical encoder with the matching seed/protocol export.
encoder = load_pretrained_encoder(
    f'artifacts/pretraining/{protocol}/seed{seed}/encoder.pt',
    expected_train_sha256=sha256(train_path),
)
classifier = nn.Sequential(encoder, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 4))
# M2 trains the full classifier using TRAIN labels and chooses stopping on VAL.
# Repeat for three seeds and evaluate DS2 only after the protocol is frozen.
```

The archive-hash argument enforces exact training-file provenance. NPZ files
rebuilt by a different environment can differ in container bytes or last-bit
reductions; `src.pretrain_audit` reports the exact-hash match separately and checks
protocol, complete training record list, preprocessing source and fitted
normalisation (rtol 1e-8, atol 1e-10). A mismatch must be inspected; for exact-file
reproduction use the same prepared arrays, or rerun pretraining on the newly
audited arrays. Never mix `patient` and `patient_strict` weights/data.

## Outputs and what may be reported

- `artifacts/pretraining/<protocol>/seed<seed>/encoder.pt`: CPU state dictionary,
  architecture, seed/epoch and source-data identity; load with `weights_only=True`.
- `summary.json`: full ten-epoch history, initial/final full-training reconstruction
  MSE, settings, parameter counts, file/tensor checksums and actual software.
- `docs/pretraining_validation.json`: independent audit of all six exports.
- `docs/report/M1_E6_RESULTS.md`: generated diagnostics and a report-ready methods
  paragraph, with explicit limits on interpretation.

These are pretrained features, not trained heartbeat classifiers. M2/M3 still own
the supervised E6 run, classification metrics and any claim of generalisation.
