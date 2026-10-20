"""Protocol checks on synthetic signals; these are not experiment results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.augmentation import AugmentConfig, augment_beat
from src.data import (ARRAY_KEYS, CLASS_NAMES, DEFAULT_VAL, DS1, DS2, RECORDS,
                      ECGDataset, audit, build, fit_normalizer, load_split,
                      make_splits, overlap_summary, segment_record, subject_id,
                      subset)


def fixture():
    """Each record has all four classes; distinct beat identities."""
    rng = np.random.default_rng(123)
    pid = np.repeat(np.array(RECORDS, dtype=np.int64), 12)
    n = len(pid)
    return {"X": rng.normal(size=(n, 1, 256)).astype(np.float32),
            "y": np.tile(np.arange(4, dtype=np.int64), n // 4),
            "pid": pid,
            "subject_id": np.array([subject_id(r) for r in pid], dtype=np.int64),
            "sample": np.tile(np.arange(12, dtype=np.int64) * 300 + 256, len(RECORDS)),
            "rr_raw": rng.uniform(0.3, 1.5, size=(n, 4)).astype(np.float32)}


class SegmentationTests(unittest.TestCase):
    def test_center_alignment_mapping_and_ignore_rhythm_markers(self):
        signal = np.arange(2000, dtype=np.float32)
        samples = [200, 300, 400, 600, 800, 1000, 1200, 1400]
        symbols = ["N", "+", "A", "V", "F", "Q", "L", "N"]
        data, excluded = segment_record(signal, samples, symbols, 100)
        np.testing.assert_array_equal(data["y"], [1, 2, 3, 0])
        np.testing.assert_array_equal(data["sample"], [400, 600, 800, 1200])
        self.assertEqual(data["X"][0, 0, 128], 400)
        # '+' at 300 is not a beat. Q at 1000 still supplies RR context.
        self.assertAlmostEqual(float(data["rr_raw"][0, 0]), 200 / 360)
        self.assertAlmostEqual(float(data["rr_raw"][-1, 0]), 200 / 360)
        self.assertEqual(excluded["q_target"], 1)
        self.assertEqual(excluded["non_target_annotation"], 1)
        self.assertEqual(excluded["missing_rr_context"], 2)

    def test_incomplete_windows_are_dropped(self):
        data, excluded = segment_record(np.ones(1000), [20, 100, 400, 900, 980], ["N"] * 5, 100)
        np.testing.assert_array_equal(data["sample"], [400])
        self.assertEqual(excluded["window_boundary"], 4)

    def test_nonfinite_window_is_dropped(self):
        signal = np.ones(2000)
        signal[600] = np.nan
        data, excluded = segment_record(signal, [200, 600, 1000, 1400], ["N"] * 4, 100)
        np.testing.assert_array_equal(data["sample"], [1000])
        self.assertEqual(excluded["nonfinite_signal"], 1)

    def test_timing_ignores_class_labels_and_is_record_local(self):
        signal = np.ones(2500)
        samples = [200, 500, 900, 1300, 1800]
        a, _ = segment_record(signal, samples, ["N"] * 5, 201)
        b, _ = segment_record(signal, samples, ["V", "A", "F", "L", "N"], 202)
        np.testing.assert_array_equal(a["rr_raw"], b["rr_raw"])
        np.testing.assert_array_equal(b["subject_id"], [201] * 3)
        self.assertAlmostEqual(float(a["rr_raw"][1, 2]), 350 / 360)
        self.assertAlmostEqual(float(a["rr_raw"][1, 3]), 400 / 350, places=6)

    def test_bad_sample_order_and_sampling_rate_fail(self):
        with self.assertRaises(ValueError):
            segment_record(np.ones(1000), [500, 200], ["N", "N"], 100)
        with self.assertRaises(ValueError):
            segment_record(np.ones(1000), [200, 500], ["N", "N"], 100, fs=250)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.data = fixture()

    def test_strict_subject_disjointness_and_conventional_exception(self):
        splits = make_splits(self.data)
        stats = fit_normalizer(self.data, splits["patient"][0])
        conventional = [subset(self.data, idx, stats) for idx in splits["patient"]]
        strict = [subset(self.data, idx, stats) for idx in splits["patient_strict"]]
        self.assertEqual(overlap_summary(conventional)["train_test"]["shared_subjects"], [201])
        for info in overlap_summary(strict).values():
            self.assertEqual(info["shared_subjects"], [])
            self.assertEqual(info["same_beats"], 0)
        self.assertEqual(set(strict[2]["pid"]), set(DS2))
        self.assertNotIn(201, strict[0]["pid"])

    def test_validation_records_are_fixed_and_separate(self):
        for protocol in ("patient", "patient_strict"):
            train, val, test = make_splits(self.data)[protocol]
            self.assertEqual(set(self.data["pid"][val]), set(DEFAULT_VAL))
            self.assertFalse(set(self.data["subject_id"][val]) & set(self.data["subject_id"][train]))
            self.assertFalse(set(self.data["subject_id"][val]) & set(self.data["subject_id"][test]))

    def test_random_partitions_cover_pool_once_and_are_reproducible(self):
        a = make_splits(self.data, seed=42)["random"]
        b = make_splits(self.data, seed=42)["random"]
        c = make_splits(self.data, seed=43)["random"]
        np.testing.assert_array_equal(np.sort(np.concatenate(a)), np.arange(len(self.data["y"])))
        self.assertEqual(len(set(np.concatenate(a))), len(self.data["y"]))
        for aa, bb in zip(a, b):
            np.testing.assert_array_equal(aa, bb)
        self.assertFalse(np.array_equal(a[0], c[0]))
        for idx in a:
            self.assertEqual(set(self.data["y"][idx]), set(range(4)))

    def test_test_outliers_do_not_affect_training_fits(self):
        train, val, test = make_splits(self.data)["patient_strict"]
        original = fit_normalizer(self.data, train)
        self.data["X"][test] = 1e6
        self.data["rr_raw"][val] = 1e6
        self.assertEqual(original, fit_normalizer(self.data, train))

    def test_invalid_validation_choices_rejected(self):
        for ids in ((100, 108, 118, 223), (108, 108, 118, 223), (108, 118, 201, 223)):
            with self.assertRaises(ValueError):
                make_splits(self.data, val_records=ids)


class AugmentationTests(unittest.TestCase):
    def test_deterministic_finite_and_no_input_mutation(self):
        x = np.linspace(-1, 1, 256, dtype=np.float32)[None, :]
        old = x.copy()
        a = augment_beat(x, np.random.default_rng(42))
        b = augment_beat(x, np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(x, old)
        self.assertEqual(a.shape, x.shape)
        self.assertTrue(np.isfinite(a).all())

    def test_disabled_augmentation_returns_copy(self):
        x = np.ones((1, 256), dtype=np.float32)
        a = augment_beat(x, np.random.default_rng(42), AugmentConfig(probability=0))
        np.testing.assert_array_equal(a, x)
        self.assertFalse(np.shares_memory(x, a))

    def test_shift_is_not_circular(self):
        x = np.arange(256, dtype=np.float32)[None, :]
        c = AugmentConfig(probability=1, noise_std=0, scale_min=1, scale_max=1,
                          wander_amplitude=0, max_shift=8)
        a = augment_beat(x, np.random.default_rng(2), c)
        # Constant difference in interior, with monotonic edge padding.
        self.assertTrue(np.all(np.diff(a[0]) >= 0))
        self.assertTrue(np.allclose(np.diff(a[0, 16:-16]), 1))

    def test_invalid_config_fails(self):
        with self.assertRaises(ValueError):
            AugmentConfig(probability=1.5)
        with self.assertRaises(ValueError):
            AugmentConfig(scale_min=-1)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "processed"
        with patch("src.data.collect_records", return_value=(fixture(), [], {})):
            build("unused", self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_and_independent_audit(self):
        result = audit(self.root)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(list(self.root.glob("*.npz"))), 9)
        arrays = load_split(self.root / "patient_strict_train.npz")
        self.assertEqual(set(arrays), set(ARRAY_KEYS))
        self.assertEqual(arrays["X"].shape[1:], (1, 256))

    def test_tampered_artifact_rejected(self):
        with (self.root / "patient_test.npz").open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "File changed"):
            audit(self.root)

    def test_existing_output_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            build("unused", self.root)

    def test_augmentation_refused_for_val_and_test(self):
        for name in ("patient_val", "random_test", "patient_strict_test"):
            with self.assertRaisesRegex(ValueError, "only on training"):
                ECGDataset(self.root / f"{name}.npz", augment=True)

    def test_torch_dataset_contract_if_available(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional PyTorch not installed in M1 build environment")
        from torch.utils.data import DataLoader
        dataset = ECGDataset(self.root / "patient_train.npz", augment=True,
                             augment_config=AugmentConfig(probability=1))
        first = dataset[0]
        torch.testing.assert_close(first["X"], dataset[0]["X"])
        dataset.set_epoch(1)
        self.assertFalse(torch.equal(first["X"], dataset[0]["X"]))
        self.assertEqual(first["y"].dtype, torch.long)
        batch = next(iter(DataLoader(dataset, batch_size=4)))
        self.assertEqual(batch["X"].shape, (4, 1, 256))
        self.assertEqual(batch["rr"].shape, (4, 4))
        frozen = ECGDataset(self.root / "patient_test.npz")
        original = frozen.arrays["X"][0].copy()
        frozen[0]["X"].zero_()
        np.testing.assert_array_equal(frozen.arrays["X"][0], original)


if __name__ == "__main__":
    unittest.main()
