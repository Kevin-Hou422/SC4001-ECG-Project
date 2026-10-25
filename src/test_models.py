"""Unit tests for M2's 1D-CNN and encoder compatibility."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from src.data import sha256
from src.models import ECGCNN
from src.pretrain import ARCHITECTURE, state_digest


class ECGCNNTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(4, 1, 256)
        self.rr = torch.randn(4, 4)
        self.labels = torch.tensor([0, 1, 2, 3])

    def test_baseline_shape_and_backprop(self):
        model = ECGCNN()
        logits = model(self.x)
        self.assertEqual(tuple(logits.shape), (4, 4))
        nn.CrossEntropyLoss()(logits, self.labels).backward()
        self.assertIsNotNone(model.encoder.blocks[0].weight.grad)

    def test_baseline_ignores_rr(self):
        model = ECGCNN(use_rr=False).eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(self.x), model(self.x, self.rr)))

    def test_rr_shape_and_backprop(self):
        model = ECGCNN(use_rr=True)
        self.assertEqual(model.classifier.in_features, 68)
        logits = model(self.x, self.rr)
        self.assertEqual(tuple(logits.shape), (4, 4))
        nn.CrossEntropyLoss()(logits, self.labels).backward()
        self.assertIsNotNone(model.classifier.weight.grad)
        with self.assertRaises(ValueError):
            model(self.x)
        with self.assertRaises(ValueError):
            model(self.x, self.rr[:, :3])

    def test_bad_waveform_shape(self):
        with self.assertRaises(ValueError):
            ECGCNN()(self.x.squeeze(1))

    def test_pretrained_encoder_matches_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            train_file = Path(tmp) / 'patient_train.npz'
            train_file.write_bytes(b'placeholder training archive for checksum test')
            source_model = ECGCNN()
            state = {name: tensor.detach().clone()
                     for name, tensor in source_model.encoder.state_dict().items()}
            export = {
                'format_version': 1,
                'architecture': ARCHITECTURE,
                'encoder_state_dict': state,
                'seed': 42,
                'epochs': 10,
                'data': {'train_sha256': sha256(train_file)},
                'encoder_sha256': state_digest(state),
            }
            checkpoint = Path(tmp) / 'encoder.pt'
            torch.save(export, checkpoint)
            torch.manual_seed(43)
            new_model = ECGCNN()
            self.assertFalse(torch.equal(source_model.encoder.blocks[0].weight,
                                         new_model.encoder.blocks[0].weight))
            new_model.load_pretrained(checkpoint, train_file)
            for name, tensor in source_model.encoder.state_dict().items():
                self.assertTrue(torch.equal(tensor, new_model.encoder.state_dict()[name]))
            self.assertTrue(torch.allclose(new_model(self.x), new_model(self.x)))
            different_file = Path(tmp) / 'wrong_train.npz'
            different_file.write_bytes(b'not the same')
            with self.assertRaises(ValueError):
                new_model.load_pretrained(checkpoint, different_file)


if __name__ == '__main__':
    unittest.main()
