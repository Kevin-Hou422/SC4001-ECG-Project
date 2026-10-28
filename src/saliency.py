"""Saliency for trained beat classifiers (M3 stretch goal).

Integrated gradients (Sundararajan et al., 2017) attribute the target-class
logit to each of the 256 input samples by averaging gradients along a straight
path from a baseline to the beat. With the default baseline of zeros (the
training-mean amplitude after standardisation), attributions sum approximately
to logit(beat) - logit(baseline), which ``completeness_gap`` checks.

The code is model-agnostic: pass ``forward``, a function mapping a float tensor
of shape (B, 1, 256) to logits (B, 4). For a model that also takes RR features,
close over the matching RR batch, e.g. ``lambda x: model(x, rr_batch)``.
Put the model in ``eval()`` mode first so dropout/batch-norm are fixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import CLASS_NAMES


def _as_tensor(x):
    import torch
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def integrated_gradients(forward, x, target, baseline=None, steps=64):
    """Attributions with the same shape as ``x`` (B, 1, 256), as a NumPy array."""
    import torch
    x = _as_tensor(x)
    baseline = torch.zeros_like(x) if baseline is None else _as_tensor(baseline).expand_as(x)
    target = torch.as_tensor(np.asarray(target), dtype=torch.long).reshape(-1, 1)
    total = torch.zeros_like(x)
    for alpha in (torch.arange(steps, dtype=torch.float32) + 0.5) / steps:  # midpoint rule
        point = (baseline + alpha * (x - baseline)).requires_grad_(True)
        score = forward(point).gather(1, target).sum()
        (grad,) = torch.autograd.grad(score, point)
        total += grad
    return ((x - baseline) * total / steps).detach().cpu().numpy()


def gradient_x_input(forward, x, target):
    """Cheaper single-gradient saliency, for a quick comparison with IG."""
    import torch
    x = _as_tensor(x).requires_grad_(True)
    target = torch.as_tensor(np.asarray(target), dtype=torch.long).reshape(-1, 1)
    (grad,) = torch.autograd.grad(forward(x).gather(1, target).sum(), x)
    return (grad * x).detach().cpu().numpy()


def completeness_gap(forward, x, target, attributions, baseline=None):
    """Per beat: sum(attributions) - (logit(x) - logit(baseline)); near 0 if steps suffice."""
    import torch
    x = _as_tensor(x)
    baseline = torch.zeros_like(x) if baseline is None else _as_tensor(baseline).expand_as(x)
    target = torch.as_tensor(np.asarray(target), dtype=torch.long).reshape(-1, 1)
    with torch.no_grad():
        delta = forward(x).gather(1, target) - forward(baseline).gather(1, target)
    return attributions.reshape(len(x), -1).sum(axis=1) - delta.squeeze(1).cpu().numpy()


def attribute_in_batches(forward_for, x, target, batch_size=256, **kwargs):
    """Run IG over many beats. ``forward_for(slice)`` returns the forward
    function for rows ``slice`` (so RR features can be matched to the batch)."""
    parts = []
    for start in range(0, len(x), batch_size):
        rows = slice(start, start + batch_size)
        parts.append(integrated_gradients(forward_for(rows), x[rows], target[rows], **kwargs))
    return np.concatenate(parts)


def plot_class_saliency(x, attributions, y, path, title="Integrated gradients by class", fs=360):
    """Small multiples: median beat (top) and mean |attribution| (bottom) per class."""
    import matplotlib.pyplot as plt
    from .figures import CLASS_COLORS, INK_2, RC, TEXT_WIDTH, _save
    t_ms = (np.arange(x.shape[-1]) - x.shape[-1] // 2) / fs * 1000
    classes = [i for i in range(len(CLASS_NAMES)) if np.any(y == i)]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, len(classes), figsize=(TEXT_WIDTH, 2.8), sharex=True,
                                 sharey="row", layout="constrained")
        axes = np.asarray(axes).reshape(2, len(classes))
        for col, i in enumerate(classes):
            cls = CLASS_NAMES[i]
            beats, attr = x[y == i, 0], np.abs(attributions[y == i, 0])
            axes[0, col].plot(t_ms, np.median(beats, axis=0), color=CLASS_COLORS[cls], lw=1.3)
            axes[0, col].set_title(f"{cls} (n={len(beats):,})")
            axes[1, col].fill_between(t_ms, 0, attr.mean(axis=0), color=CLASS_COLORS[cls],
                                      alpha=0.35, lw=0)
            axes[1, col].plot(t_ms, attr.mean(axis=0), color=CLASS_COLORS[cls], lw=1.0)
            for ax in axes[:, col]:
                ax.grid(axis="y")
                ax.set_axisbelow(True)
        axes[0, 0].set_ylabel("Median beat")
        axes[1, 0].set_ylabel("Mean |attribution|")
        fig.supxlabel("Time from R peak (ms)", fontsize=8, color=INK_2)
        fig.suptitle(title, fontsize=8.5, weight="bold", x=0.01, ha="left")
        _save(fig, Path(path))
