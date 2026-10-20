"""Training-only waveform augmentation in training-standardised units.

RR features and labels are unchanged: these transforms simulate measurement
noise/gain/baseline/alignment changes, not changes in the underlying beat times.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    probability: float = 0.5  # independent probability for each transform
    noise_std: float = 0.02
    scale_min: float = 0.9
    scale_max: float = 1.1
    wander_amplitude: float = 0.05
    wander_hz_min: float = 0.1
    wander_hz_max: float = 0.5
    max_shift: int = 8
    fs: float = 360.0

    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if self.noise_std < 0 or self.wander_amplitude < 0:
            raise ValueError("noise and wander amplitude must be non-negative")
        if not 0 < self.scale_min <= self.scale_max:
            raise ValueError("scaling range must be positive and ordered")
        if not 0 < self.wander_hz_min <= self.wander_hz_max < self.fs / 2:
            raise ValueError("invalid baseline wander frequency range")
        if not isinstance(self.max_shift, int) or self.max_shift < 0:
            raise ValueError("max_shift must be a non-negative integer")


def augment_beat(x, rng, config=None):
    """Return a fresh float32 (1, 256) array; never mutate cached data.

    Shift uses edge padding, never circular wraparound. Parameters are fixed
    defaults for initial experiments, not a claim of physiological optimality.
    """
    c = config or AugmentConfig()
    out = np.array(x, dtype=np.float32, copy=True)
    if out.shape != (1, 256) or not np.isfinite(out).all():
        raise ValueError("expected finite beat with shape (1, 256)")
    if rng.random() < c.probability:
        out *= rng.uniform(c.scale_min, c.scale_max)
    if rng.random() < c.probability:
        out += rng.normal(0, c.noise_std, size=out.shape).astype(np.float32)
    if rng.random() < c.probability:
        t = np.arange(out.shape[-1]) / c.fs
        wave = c.wander_amplitude * np.sin(
            2 * np.pi * rng.uniform(c.wander_hz_min, c.wander_hz_max) * t
            + rng.uniform(0, 2 * np.pi)
        )
        out += wave.astype(np.float32)
    if rng.random() < c.probability and c.max_shift:
        shift = int(rng.integers(-c.max_shift, c.max_shift + 1))
        if abs(shift) >= out.shape[-1]:
            raise ValueError("shift must be shorter than the beat")
        if shift > 0:
            out = np.pad(out, ((0, 0), (shift, 0)), mode="edge")[:, :256]
        elif shift < 0:
            out = np.pad(out, ((0, 0), (0, -shift)), mode="edge")[:, -shift:]
    return np.ascontiguousarray(out, dtype=np.float32)

