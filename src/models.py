"""1D-CNN heartbeat classifiers for the SC4001 ECG experiments.

E0-E4: ECGCNN(use_rr=False)     -> logits (batch, 4)
E5:    ECGCNN(use_rr=True)      -> logits (batch, 4)
E6:    ECGCNN(use_rr=False) with the same encoder initialised from M1's export.

E1 and E6 can use identical architectures; only initial weights differ.



Return raw logits for nn.CrossEntropyLoss / focal loss.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import CLASS_NAMES, RR_NAMES, sha256
from .pretrain import ECGEncoder, load_pretrained_encoder


class ECGCNN(nn.Module):
    """Small 1D CNN with optional late fusion of four RR-interval features.
    Architecture:
      ECG waveform (B, 1, 256)
        -> M1's Conv1d encoder (B, 64, 32)
        -> global average pooling (B, 64)
        -> [optional: concatenate standardised RR features (B, 4)]
        -> linear classifier -> four class logits (N, S, V, F).

    Set ``use_rr=False`` for E0-E4, including weighted/focal losses and
    augmentation: these are *training* changes, not architecture changes.
    Set ``use_rr=True`` only for the E5 RR experiment.
    """

    def __init__(self, use_rr: bool = False):
        super().__init__()
        self.use_rr = bool(use_rr)
        self.encoder = ECGEncoder()  
        self.pool = nn.AdaptiveAvgPool1d(1)
        input_features = 64 + (len(RR_NAMES) if self.use_rr else 0)
        self.classifier = nn.Linear(input_features, len(CLASS_NAMES))

    def forward(self, x: torch.Tensor, rr: torch.Tensor | None = None) -> torch.Tensor:
        """Return logits of shape (batch, 4); do not apply softmax here."""
        if x.ndim != 3 or x.shape[1:] != (1, 256):
            raise ValueError(f"X must have shape (batch, 1, 256); got {tuple(x.shape)}")

        features = self.pool(self.encoder(x)).flatten(start_dim=1)  # (B, 64)

        if self.use_rr:
            if rr is None:
                raise ValueError("This model uses RR features: pass batch['rr']")
            if rr.ndim != 2 or rr.shape != (x.shape[0], len(RR_NAMES)):
                raise ValueError(
                    f"rr must have shape ({x.shape[0]}, {len(RR_NAMES)}); "
                    f"got {tuple(rr.shape)}"
                )
            features = torch.cat((features, rr), dim=1)  # (B, 68)

        return self.classifier(features)

    def load_pretrained(self, checkpoint_path: str | Path,
                        train_npz_path: str | Path) -> None:
        """E6 only: load M1's encoder and verify the exact training archive.

        This checks both the exported architecture/state checksum and the
        It raises on a mismatch rather than silently loading partial weights.
        The classification head is *not* pretrained and remains random.
        """
        load_pretrained_encoder(
            checkpoint_path,
            encoder=self.encoder,
            expected_train_sha256=sha256(train_npz_path),
        )
