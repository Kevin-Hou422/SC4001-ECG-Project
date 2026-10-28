"""Checks of the integrated-gradients code on small synthetic models."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:  # saliency needs PyTorch; data/evaluation code does not
    torch = None

from src.saliency import (attribute_in_batches, completeness_gap, gradient_x_input,
                          integrated_gradients, plot_class_saliency)


@unittest.skipIf(torch is None, "PyTorch not installed")
class SaliencyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = np.random.default_rng(0).normal(size=(6, 1, 256)).astype(np.float32)
        self.target = np.array([0, 1, 2, 3, 1, 0])

    def test_linear_model_has_exact_attributions(self):
        linear = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(256, 4))
        attr = integrated_gradients(linear, self.x, self.target, steps=8)
        weights = linear[1].weight.detach().numpy()
        expected = self.x[:, 0] * weights[self.target]
        np.testing.assert_allclose(attr[:, 0], expected, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(gradient_x_input(linear, self.x, self.target)[:, 0],
                                   expected, rtol=1e-4, atol=1e-6)

    def test_completeness_on_nonlinear_cnn(self):
        cnn = torch.nn.Sequential(torch.nn.Conv1d(1, 4, 7, padding=3), torch.nn.Tanh(),
                                  torch.nn.AdaptiveAvgPool1d(8), torch.nn.Flatten(),
                                  torch.nn.Linear(32, 4)).eval()
        attr = integrated_gradients(cnn, self.x, self.target, steps=128)
        gap = completeness_gap(cnn, self.x, self.target, attr)
        self.assertLess(np.abs(gap).max(), 1e-2)

    def test_batches_match_single_call_with_extra_inputs(self):
        rr = torch.randn(6, 4)
        head = torch.nn.Linear(256 + 4, 4)

        def forward_for(rows):
            return lambda x: head(torch.cat([x.flatten(1), rr[rows]], dim=1))

        whole = integrated_gradients(forward_for(slice(0, 6)), self.x, self.target, steps=4)
        batched = attribute_in_batches(forward_for, self.x, self.target, batch_size=4, steps=4)
        np.testing.assert_allclose(whole, batched, rtol=1e-5, atol=1e-6)

    def test_plot_writes_files(self):
        y = np.array([0, 1, 2, 3, 1, 0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "saliency"
            plot_class_saliency(self.x, self.x * 0.1, y, path)
            self.assertTrue(path.with_suffix(".png").is_file())


if __name__ == "__main__":
    unittest.main()
