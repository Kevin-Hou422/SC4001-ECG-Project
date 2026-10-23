"""E6 safeguards and exact epoch-boundary resume on small synthetic signals."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.data import build, sha256
from test_data import fixture

if torch is not None:
    from src.pretrain import (ConvAutoencoder, ECGEncoder, PretrainConfig,
                              load_pretrained_encoder, pretrain, read_training_source)
    from src.pretrain_audit import audit_export


@unittest.skipIf(torch is None, "optional PyTorch not installed")
class PretrainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.processed = self.root / "processed"
        with patch("src.data.collect_records", return_value=(fixture(), [], {})):
            with contextlib.redirect_stdout(io.StringIO()):
                build("unused", self.processed)
        self.train = self.processed / "patient_strict_train.npz"
        self.manifest = self.processed / "manifest.json"

    def tearDown(self):
        self.temp.cleanup()

    def run_pretrain(self, name, epochs=1, resume=False, seed=42):
        with contextlib.redirect_stdout(io.StringIO()):
            return pretrain(self.train, self.manifest, self.root / name,
                            PretrainConfig(seed=seed, epochs=epochs, batch_size=128, threads=1), resume)

    def test_decoder_preserves_waveform_length(self):
        model = ConvAutoencoder()
        x = torch.randn(2, 1, 256)
        self.assertEqual(model.encoder(x).shape, (2, 64, 32))
        self.assertEqual(model.encoder.embedding(x).shape, (2, 64))
        self.assertEqual(model(x).shape, x.shape)

    def test_only_correct_training_source_is_accepted(self):
        x, identity = read_training_source(self.train, self.manifest)
        self.assertEqual(len(x), identity["n_beats"])
        self.assertNotIn(201, identity["records"])
        for name in ("patient_val", "patient_test", "patient_strict_val", "random_train"):
            with self.assertRaisesRegex(ValueError, "only.*TRAINING"):
                read_training_source(self.processed / f"{name}.npz", self.manifest)

    def test_modified_training_archive_rejected(self):
        with self.train.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "audited manifest"):
            read_training_source(self.train, self.manifest)

    def test_encoder_export_loads_exactly_and_rejects_wrong_data(self):
        summary = self.run_pretrain("run")
        path = self.root / "run/encoder.pt"
        encoder = load_pretrained_encoder(path, expected_train_sha256=sha256(self.train))
        state = torch.load(self.root / "run/last.pt", weights_only=True)
        model = ConvAutoencoder()
        model.load_state_dict(state["model_state_dict"])
        for key, tensor in model.encoder.state_dict().items():
            torch.testing.assert_close(tensor, encoder.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(summary["selection"], "final epoch of fixed budget; no validation/test selection")
        with self.assertRaisesRegex(ValueError, "different training archive"):
            load_pretrained_encoder(path, expected_train_sha256="wrong")

    def test_resume_matches_uninterrupted_training(self):
        full = self.run_pretrain("full", epochs=2)
        self.run_pretrain("resumed", epochs=1)
        resumed = self.run_pretrain("resumed", epochs=2, resume=True)
        self.assertEqual(full["encoder_sha256"], resumed["encoder_sha256"])
        self.assertEqual(full["history"], resumed["history"])
        self.assertEqual(full["final_train_mse"], resumed["final_train_mse"])

    def test_resume_rejects_changed_seed(self):
        self.run_pretrain("run")
        with self.assertRaisesRegex(ValueError, "Resume data"):
            self.run_pretrain("run", epochs=2, resume=True, seed=43)

    def test_existing_run_is_not_overwritten(self):
        self.run_pretrain("run")
        with self.assertRaises(FileExistsError):
            self.run_pretrain("run")

    def test_target_model_must_match_encoder_architecture(self):
        self.run_pretrain("run")
        with self.assertRaises(RuntimeError):
            load_pretrained_encoder(self.root / "run/encoder.pt", encoder=torch.nn.Linear(2, 2))

    def test_independent_export_audit(self):
        self.run_pretrain("run")
        result = audit_export(self.root / "run", self.train, self.manifest, 42, 1)
        self.assertEqual(result["status"], "passed")
        self.assertTrue(result["exact_training_npz_sha256_match"])
        self.assertTrue(result["source_matches_current_pretrain"])
        self.assertFalse(result["classification_evaluated"])

    def test_tampered_encoder_is_rejected(self):
        self.run_pretrain("run")
        with (self.root / "run/encoder.pt").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            audit_export(self.root / "run", self.train, self.manifest)


if __name__ == "__main__":
    unittest.main()
