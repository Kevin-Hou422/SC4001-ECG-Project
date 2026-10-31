"""Checks of the M3 evaluation code on hand-made data; not experiment results."""
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

from src.data import CLASS_NAMES, DS2, subject_id
from src.evaluate import (PROB_COLUMNS, attach_beat_identity, auroc, bootstrap_table,
                          confusion_matrix, find_test_split, load_experiment,
                          load_predictions, make_dummy_predictions, metrics_by_seed,
                          metrics_from_confusion, patient_bootstrap, per_patient_table,
                          ranking_table, record_confusions, summarise, summarise_confusion,
                          threshold_rule, threshold_transfer, timing_breakdown,
                          trivial_baseline)


def predictions(pid, y_true, y_pred):
    probs = np.full((len(y_pred), 4), 0.1)
    probs[np.arange(len(y_pred)), y_pred] = 0.7
    df = pd.DataFrame({"pid": pid, "y_true": y_true, "y_pred": y_pred})
    for i, column in enumerate(PROB_COLUMNS):
        df[column] = probs[:, i]
    return df


def make_test_split(records=(100, 232, 213), per_record=12, seed=0):
    """A minimal test split that passes src.data.validate_arrays."""
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.asarray(records, dtype=np.int64), per_record)
    n = len(pid)
    return {"X": rng.normal(size=(n, 1, 256)).astype(np.float32),
            "y": np.tile(np.arange(4, dtype=np.int64), n // 4),
            "pid": pid,
            "subject_id": np.array([subject_id(r) for r in pid], dtype=np.int64),
            "sample": np.tile(np.arange(per_record, dtype=np.int64) * 300 + 256, len(records)),
            "rr": rng.normal(size=(n, 4)).astype(np.float32),
            "rr_raw": rng.uniform(0.3, 1.5, size=(n, 4)).astype(np.float32)}


class MetricTests(unittest.TestCase):
    def test_confusion_orientation(self):
        cm = confusion_matrix([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 3, 0])
        self.assertEqual(cm[0, 1], 1)  # true N predicted S
        self.assertEqual(cm[3, 0], 1)  # true F predicted N
        self.assertEqual(cm.sum(), 6)

    def test_hand_computed_metrics(self):
        # N: TP 2, FP 1 (an F), FN 1 (-> S). S: TP 1, FP 1. V: TP 1. F: TP 1, FN 1.
        y_true = [0, 0, 0, 1, 2, 3, 3]
        y_pred = [0, 0, 1, 1, 2, 3, 0]
        m = summarise_confusion(confusion_matrix(y_true, y_pred))
        self.assertAlmostEqual(m["accuracy"], 5 / 7)
        self.assertAlmostEqual(m["N_precision"], 2 / 3)
        self.assertAlmostEqual(m["N_recall"], 2 / 3)
        self.assertAlmostEqual(m["S_precision"], 1 / 2)
        self.assertAlmostEqual(m["S_recall"], 1.0)
        self.assertAlmostEqual(m["F_recall"], 1 / 2)
        f1 = [2 * 2 / (3 + 3), 2 * 1 / (1 + 2), 1.0, 2 * 1 / (2 + 1)]
        self.assertAlmostEqual(m["macro_f1"], np.mean(f1))
        self.assertAlmostEqual(m["balanced_accuracy"], np.mean([2 / 3, 1, 1, 1 / 2]))

    def test_undefined_values_are_nan_not_zero(self):
        # S is present but never predicted; F is absent and never predicted.
        m = summarise_confusion(confusion_matrix([0, 0, 1, 2], [0, 0, 0, 2]))
        self.assertTrue(np.isnan(m["S_precision"]))
        self.assertEqual(m["S_recall"], 0.0)
        self.assertEqual(m["S_f1"], 0.0)
        self.assertTrue(np.isnan(m["F_recall"]))
        self.assertTrue(np.isnan(m["F_f1"]))
        n_f1 = 2 * 2 / (3 + 2)
        self.assertAlmostEqual(m["macro_f1"], np.mean([n_f1, 0.0, 1.0]))  # F excluded

    def test_absent_class_with_false_positives_scores_zero(self):
        m = summarise_confusion(confusion_matrix([0, 0, 2], [0, 3, 2]))
        self.assertTrue(np.isnan(m["F_recall"]))
        self.assertEqual(m["F_precision"], 0.0)
        self.assertEqual(m["F_f1"], 0.0)

    def test_matches_scikit_learn(self):
        try:
            from sklearn.metrics import f1_score, precision_recall_fscore_support
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 4, 500)
        y_pred = np.where(rng.random(500) < 0.7, y_true, rng.integers(0, 4, 500))
        m = summarise_confusion(confusion_matrix(y_true, y_pred))
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(4))
        for i, cls in enumerate(CLASS_NAMES):
            self.assertAlmostEqual(m[f"{cls}_precision"], p[i])
            self.assertAlmostEqual(m[f"{cls}_recall"], r[i])
            self.assertAlmostEqual(m[f"{cls}_f1"], f[i])
            self.assertEqual(m[f"{cls}_support"], s[i])
        self.assertAlmostEqual(m["macro_f1"], f1_score(y_true, y_pred, average="macro"))

    def test_vectorised_metrics_match_loop(self):
        rng = np.random.default_rng(2)
        cms = rng.integers(0, 20, size=(5, 3, 4, 4))
        batch = metrics_from_confusion(cms)
        for i in range(5):
            for j in range(3):
                single = metrics_from_confusion(cms[i, j])
                np.testing.assert_allclose(batch["f1"][i, j], single["f1"])
                self.assertAlmostEqual(float(batch["macro_f1"][i, j]), float(single["macro_f1"]))

    def test_always_normal_baseline(self):
        y = np.array([0] * 90 + [1] * 5 + [2] * 4 + [3])
        m = trivial_baseline(y)
        self.assertAlmostEqual(m["accuracy"], 0.9)
        self.assertAlmostEqual(m["macro_f1"], (2 * 90 / 190) / 4)


class PerRecordTests(unittest.TestCase):
    def setUp(self):
        self.df = predictions(pid=[100, 100, 100, 232, 232, 232],
                              y_true=[0, 0, 2, 1, 1, 0],
                              y_pred=[0, 1, 2, 0, 1, 0])

    def test_record_confusions_sum_to_overall(self):
        records, cms = record_confusions(self.df)
        np.testing.assert_array_equal(records, [100, 232])
        overall = confusion_matrix(self.df["y_true"], self.df["y_pred"])
        np.testing.assert_array_equal(cms.sum(axis=0), overall)

    def test_per_patient_table_keeps_support_and_nan(self):
        table = per_patient_table(self.df).set_index("pid")
        self.assertEqual(table.loc[100, "S_support"], 0)
        self.assertTrue(np.isnan(table.loc[100, "S_recall"]))
        self.assertEqual(table.loc[100, "S_false_pos"], 1)
        self.assertAlmostEqual(table.loc[232, "S_recall"], 0.5)

    def test_timing_breakdown_bins_by_prematurity(self):
        df = self.df.assign(rr_ratio=[1.0, 1.0, 0.7, 0.7, 1.0, 1.0])
        table = timing_breakdown({"seed0": df, "seed1": df}).set_index(["class", "records", "ratio_bin"])
        # Record 232 S beats: the premature one (0.7) is missed, the on-time one (1.0) found.
        self.assertEqual(table.loc[("S", "record_232", "0-0.8"), "recall"], 0.0)
        self.assertEqual(table.loc[("S", "record_232", "0.9-1.1"), "recall"], 1.0)
        self.assertEqual(table.loc[("S", "all", "0-0.8"), "support"], 1)
        self.assertTrue(np.isnan(table.loc[("S", "other_records", "0-0.8"), "recall"]))
        self.assertTrue(timing_breakdown({"seed0": self.df}).empty)  # not aligned to an NPZ

    def test_subsets_exclude_whole_records(self):
        experiment = {"name": "E1", "test": None, "seeds": {"seed0": self.df}}
        table = metrics_by_seed(experiment).set_index("subset")
        self.assertEqual(table.loc["all", "n"], 6)
        self.assertEqual(table.loc["without_232", "n"], 3)
        self.assertTrue(np.isnan(table.loc["without_232", "S_recall"]))


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.cms = rng.integers(0, 30, size=(3, 10, 4, 4))  # seeds, records

    def test_deterministic_and_paired_zero_difference(self):
        a = patient_bootstrap({"E1": self.cms, "E2": self.cms}, ["macro_f1"], n_boot=200, seed=5)
        b = patient_bootstrap({"E1": self.cms, "E2": self.cms}, ["macro_f1"], n_boot=200, seed=5)
        pd.testing.assert_frame_equal(a, b)
        np.testing.assert_array_equal(a["E1|macro_f1"], a["E2|macro_f1"])
        table = bootstrap_table({"E1": self.cms, "E2": self.cms}, ["macro_f1"], "E1", n_boot=200)
        row = table[table["experiment"] == "E2"].iloc[0]
        self.assertEqual(row["diff"], 0.0)
        self.assertEqual(row["diff_low"], 0.0)

    def test_interval_brackets_estimate_for_homogeneous_records(self):
        same = np.repeat(self.cms[:, :1], 10, axis=1)  # every record identical
        table = bootstrap_table({"E1": same}, ["macro_f1", "S_recall"], n_boot=100)
        for _, row in table.iterrows():
            self.assertAlmostEqual(row["ci_low"], row["estimate"])
            self.assertAlmostEqual(row["ci_high"], row["estimate"])

    def test_mismatched_record_lists_rejected(self):
        with self.assertRaises(ValueError):
            patient_bootstrap({"A": self.cms, "B": self.cms[:, :5]})


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, df, name="seed0.csv"):
        path = self.dir / name
        df.to_csv(path, index=False)
        return path

    def test_valid_file_round_trip_and_integral_floats(self):
        df = predictions([100, 232], [0, 1], [0, 1])
        df["y_pred"] = df["y_pred"].astype(float)
        loaded = load_predictions(self.write(df))
        self.assertEqual(loaded["y_pred"].dtype, np.int64)

    def test_contract_violations_rejected(self):
        good = predictions([100, 232], [0, 1], [0, 1])
        bad_cases = [good.drop(columns="p_S"), good.assign(y_true=[0, 4]),
                     good.assign(p_N=[0.9, 0.9]), good.assign(y_pred=[0.5, 1])]
        for i, bad in enumerate(bad_cases):
            with self.subTest(i=i), self.assertRaises(ValueError):
                load_predictions(self.write(bad, f"bad{i}.csv"))

    def test_argmax_disagreement_warns(self):
        df = predictions([100, 232], [0, 1], [0, 1])
        df.loc[0, "y_pred"] = 2
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_predictions(self.write(df))
        self.assertTrue(any("argmax" in str(w.message) for w in caught))

    def test_alignment_with_test_npz(self):
        arrays = make_test_split()
        np.savez(self.dir / "patient_test.npz", **arrays)
        df = predictions(arrays["pid"], arrays["y"], arrays["y"])
        name, found = find_test_split(df, self.dir)
        self.assertEqual(name, "patient_test.npz")
        aligned = attach_beat_identity(df, found)
        np.testing.assert_array_equal(aligned["sample"], arrays["sample"])
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        self.assertEqual(find_test_split(shuffled, self.dir), (None, None))
        with self.assertRaises(ValueError):
            attach_beat_identity(df.assign(sample=arrays["sample"] + 1), found)
        with self.assertRaises(ValueError):
            attach_beat_identity(df.assign(subject_id=arrays["subject_id"] + 1), found)

    def test_unmatched_predictions_fail_unless_allowed(self):
        arrays = make_test_split()
        np.savez(self.dir / "patient_test.npz", **arrays)
        exp = self.dir / "E1"
        exp.mkdir()
        predictions(arrays["pid"], arrays["y"], arrays["y"]).sample(frac=1, random_state=0) \
            .to_csv(exp / "seed0.csv", index=False)
        with self.assertRaises(ValueError):
            load_experiment(exp, self.dir)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loaded = load_experiment(exp, self.dir, allow_unmatched=True)
        self.assertIsNone(loaded["test"])
        self.assertTrue(any("no matching test NPZ" in str(w.message) for w in caught))

    def test_seeds_must_share_rows(self):
        exp = self.dir / "E1"
        exp.mkdir()
        predictions([100, 232], [0, 1], [0, 1]).to_csv(exp / "seed0.csv", index=False)
        predictions([232, 100], [1, 0], [1, 0]).to_csv(exp / "seed1.csv", index=False)
        with self.assertRaises(ValueError):
            load_experiment(exp)

    def test_training_side_files_are_not_seeds(self):
        # train.py writes seed<k>_val.csv and seed<k>_history.csv next to seed<k>.csv.
        exp = self.dir / "E1"
        exp.mkdir()
        predictions([100, 232], [0, 1], [0, 1]).to_csv(exp / "seed0.csv", index=False)
        predictions([118], [1], [1]).to_csv(exp / "seed0_val.csv", index=False)
        pd.DataFrame({"epoch": [1]}).to_csv(exp / "seed0_history.csv", index=False)
        self.assertEqual(list(load_experiment(exp)["seeds"]), ["seed0"])


def scored(pid, y_true, p_s):
    """Predictions whose only informative column is p_S (the rest is mostly p_N)."""
    p_s = np.asarray(p_s, dtype=np.float64)
    probs = np.column_stack([1 - p_s - 0.02, p_s, np.full_like(p_s, 0.01), np.full_like(p_s, 0.01)])
    df = pd.DataFrame({"pid": pid, "y_true": y_true, "y_pred": probs.argmax(axis=1)})
    for i, column in enumerate(PROB_COLUMNS):
        df[column] = probs[:, i]
    return df


class PostHocTests(unittest.TestCase):
    def test_auroc_matches_scikit_learn_with_ties(self):
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            self.skipTest("scikit-learn not installed")
        rng = np.random.default_rng(0)
        scores = rng.integers(0, 5, size=200) / 4  # many ties
        positive = rng.random(200) < 0.3
        self.assertAlmostEqual(auroc(scores, positive), roc_auc_score(positive, scores))
        self.assertTrue(np.isnan(auroc([0.1, 0.2], [True, True])))

    def test_within_record_ranking_removes_patient_shift(self):
        # Each record orders its S beats perfectly, but record 232 sits higher overall.
        y = np.array([1] * 10 + [0] * 10)
        a = scored(np.full(20, 100), y, np.r_[np.full(10, 0.30), np.full(10, 0.20)])
        b = scored(np.full(20, 232), y, np.r_[np.full(10, 0.45), np.full(10, 0.40)])
        table = ranking_table({"seed0": pd.concat([a, b], ignore_index=True)}).set_index("record")
        self.assertAlmostEqual(table.loc["100", "auroc"], 1.0)
        self.assertAlmostEqual(table.loc["232", "auroc"], 1.0)
        self.assertLess(table.loc["pooled", "auroc"], 1.0)
        self.assertEqual(table.loc["pooled", "predicted_share"], 0.0)

    def test_threshold_rule(self):
        probs = np.array([[0.6, 0.3, 0.1, 0.0], [0.2, 0.1, 0.7, 0.0], [0.9, 0.05, 0.0, 0.05]])
        np.testing.assert_array_equal(threshold_rule(probs, 1, 0.25), [1, 2, 0])
        np.testing.assert_array_equal(threshold_rule(probs, 1, 0.01), [1, 1, 1])

    def test_threshold_is_chosen_on_validation_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            exp = Path(tmp)
            # Validation: S beats have p_S 0.05, N beats 0.001 -> any t in (0.001, 0.05] is best.
            y_val = np.array([1] * 5 + [0] * 20)
            scored(np.full(25, 118), y_val, np.r_[np.full(5, 0.05), np.full(20, 0.001)]) \
                .to_csv(exp / "seed0_val.csv", index=False)
            # Test: the same rule finds the S beats of record 232 but also flags record 100.
            y_test = np.array([1] * 4 + [0] * 4 + [0] * 4)
            test = scored(np.r_[np.full(8, 232), np.full(4, 100)], y_test,
                          np.r_[np.full(4, 0.06), np.full(4, 0.0005), np.full(4, 0.06)])
            table = threshold_transfer(exp, {"seed0": test, "seed1": test})
        self.assertEqual(list(table["seed"]), ["seed0"])  # seed1 has no validation file
        result = table.iloc[0]
        self.assertGreater(result["threshold"], 0.001)
        self.assertLessEqual(result["threshold"], 0.05)
        self.assertAlmostEqual(result["val_S_f1"], 1.0)
        self.assertAlmostEqual(result["S_recall"], 1.0)
        self.assertAlmostEqual(result["S_precision"], 0.5)
        self.assertTrue(np.isnan(result["S_recall_without_232"]))


class EndToEndTests(unittest.TestCase):
    def test_summarise_on_dummy_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            processed, results = tmp / "processed", tmp / "results"
            processed.mkdir()
            arrays = make_test_split(records=DS2, per_record=8)
            np.savez(processed / "patient_test.npz", **arrays)
            for exp, skill in (("E1", 0.5), ("E2", 0.8)):
                (results / exp).mkdir(parents=True)
                for k in range(3):
                    make_dummy_predictions(arrays, seed=10 * k + int(10 * skill), skill=skill) \
                        .to_csv(results / exp / f"seed{k}.csv", index=False)
            summary = summarise(results, processed, tmp / "summary", reference="E1",
                                n_boot=50, figures=False)
            self.assertEqual(set(summary["experiment"]), {"E1", "E2"})
            self.assertTrue((summary["seeds"] == 3).all())
            text = (tmp / "summary" / "summary.md").read_text(encoding="utf-8")
            self.assertIn("DUMMY PREDICTIONS", text)
            self.assertIn("Δ vs E1", text)
            per_patient = pd.read_csv(tmp / "summary" / "per_patient_E2.csv")
            self.assertEqual(sorted(per_patient["pid"]), sorted(DS2))


if __name__ == "__main__":
    unittest.main()
