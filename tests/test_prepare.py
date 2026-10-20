"""Regression tests for the exact metadata-only state published on GitHub."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.data import build
from src.prepare import ensure_prepared
from test_data import fixture


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.reference = self.root / "reference"
        with patch("src.data.collect_records", return_value=(fixture(), [], {})):
            with contextlib.redirect_stdout(io.StringIO()):
                build("unused", self.reference)
        self.target = self.root / "clone" / "processed"
        self.target.mkdir(parents=True)
        for name in ("manifest.json", "audit.json", "class_statistics.csv"):
            shutil.copy2(self.reference / name, self.target / name)

    def tearDown(self):
        self.temp.cleanup()

    def run_prepare(self):
        with patch("src.prepare.verify_raw", return_value={}):
            with patch("src.data.collect_records", return_value=(fixture(), [], {})):
                with contextlib.redirect_stdout(io.StringIO()):
                    return ensure_prepared("unused", self.target)

    def test_fresh_clone_metadata_without_npz_rebuilds(self):
        self.assertEqual(len(list(self.target.glob("*.npz"))), 0)
        result = self.run_prepare()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(len(list(self.target.glob("*.npz"))), 9)

    def test_complete_cache_is_audited_without_rebuilding(self):
        self.run_prepare()
        with patch("src.prepare.verify_raw", return_value={}):
            with patch("src.prepare.build", side_effect=AssertionError("must not rebuild")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(ensure_prepared("unused", self.target)["status"], "passed")

    def test_partial_cache_is_preserved_and_rejected(self):
        source = self.reference / "patient_train.npz"
        shutil.copy2(source, self.target / source.name)
        original = (self.target / source.name).read_bytes()
        with self.assertRaisesRegex(ValueError, "Incomplete processed cache"):
            self.run_prepare()
        self.assertEqual(original, (self.target / source.name).read_bytes())

    def test_wrong_configuration_is_rejected(self):
        path = self.target / "manifest.json"
        metadata = json.loads(path.read_text())
        metadata["config"]["seed"] = 999
        path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "different source/configuration"):
            self.run_prepare()

    def test_failed_build_does_not_publish_partial_arrays(self):
        with patch("src.prepare.verify_raw", return_value={}):
            with patch("src.prepare.build", side_effect=RuntimeError("simulated failure")):
                with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                    ensure_prepared("unused", self.target)
        self.assertFalse(list(self.target.glob("*.npz")))
        self.assertFalse(list(self.target.parent.glob(".m1_build_*")))

    def test_empty_directory_builds_successfully(self):
        self.target = self.root / "empty"
        self.assertEqual(self.run_prepare()["status"], "passed")


if __name__ == "__main__":
    unittest.main()
