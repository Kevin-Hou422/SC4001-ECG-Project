"""MIT-BIH acquisition, heartbeat segmentation, splits, RR features and loaders.

Run from ecg_project: python -m src.data download / build / audit.
No classifier is trained here; M2 owns model training and M3 owns evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
from urllib.request import urlopen
import warnings

import numpy as np

from .augmentation import AugmentConfig, augment_beat


SOURCE = "https://physionet.org/files/mitdb/1.0.0/"
DATA_LICENSE = "https://opendatacommons.org/licenses/by/1-0/"
ATTRIBUTION = "MIT-BIH Arrhythmia Database v1.0.0, George Moody and Roger Mark, PhysioNet, doi:10.13026/C2F305"
DS1 = (101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230)
DS2 = (100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234)
RECORDS = tuple(sorted(DS1 + DS2))
PACED = (102, 104, 107, 217)
DEFAULT_VAL = (108, 118, 203, 223)
CLASS_NAMES = ("N", "S", "V", "F")
CLASS_SYMBOLS = {"N": ("N", "L", "R", "e", "j"),
                 "S": ("A", "a", "J", "S"),
                 "V": ("V", "E"), "F": ("F",)}
SYMBOL_TO_CLASS = {s: i for i, name in enumerate(CLASS_NAMES)
                   for s in CLASS_SYMBOLS[name]}
Q_SYMBOLS = {"/", "f", "Q"}
BEAT_SYMBOLS = set(SYMBOL_TO_CLASS) | Q_SYMBOLS
RR_NAMES = ("pre_rr_seconds", "post_rr_seconds", "past10_mean_rr_seconds",
            "pre_over_past10_mean")
ARRAY_KEYS = ("X", "y", "pid", "subject_id", "sample", "rr", "rr_raw")


def subject_id(record_id):
    """201 and 202 are two recordings of the same subject."""
    return 201 if int(record_id) == 202 else int(record_id)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def download(raw_dir, workers=4):
    """Download only the 44 needed records and verify published SHA256 hashes.

    Downloads are retried, atomically renamed and safely resumable on rerun.
    Partial files are not accepted as records. TLS verification stays enabled.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE + "SHA256SUMS.txt", timeout=60) as response:
        sums_text = response.read().decode("utf-8")
    expected = {}
    for line in sums_text.splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip().lstrip("*")] = digest
    names = [f"{r}.{ext}" for r in RECORDS for ext in ("hea", "dat", "atr")]
    if any(name not in expected for name in names):
        raise ValueError("Source checksum list does not cover all required files")

    def fetch(name):
        target = raw_dir / name
        if target.exists() and sha256(target) == expected[name]:
            return name, "cached"
        partial = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):
            try:
                with urlopen(SOURCE + name, timeout=60) as source, partial.open("wb") as out:
                    while block := source.read(1024 * 1024):
                        out.write(block)
                if sha256(partial) != expected[name]:
                    raise ValueError(f"SHA256 mismatch: {name}")
                partial.replace(target)
                return name, "downloaded"
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(fetch, name) for name in names]
        for i, job in enumerate(as_completed(jobs), 1):
            name, status = job.result()
            print(f"[{i}/{len(names)}] {name}: {status}", flush=True)
    (raw_dir / "SHA256SUMS.txt").write_text(sums_text, encoding="utf-8")
    write_json(raw_dir / "download_manifest.json", {
        "source": SOURCE, "files": {name: expected[name] for name in names}})


def segment_record(signal, samples, symbols, record_id, fs=360):
    """256 samples [R-128, R+128), physical MLII; no filter or beatwise fitting.

    RR uses consecutive *beat* annotations including Q, never rhythm markers.
    First/last beats without pre/post context and incomplete windows are dropped.
    post-RR requires future context: this is offline heartbeat classification.
    """
    signal = np.asarray(signal)
    samples = np.asarray(samples, dtype=np.int64)
    if signal.ndim != 1 or len(samples) != len(symbols) or fs != 360:
        raise ValueError("expected MLII vector, matched annotations, fs=360")
    if np.any(np.diff(samples) < 0):
        raise ValueError("annotations must be ordered")
    beat_indices = [i for i, symbol in enumerate(symbols) if symbol in BEAT_SYMBOLS]
    peaks = samples[beat_indices]
    beat_labels = [symbols[i] for i in beat_indices]
    if np.any(np.diff(peaks) <= 0):
        raise ValueError("beat locations must be strictly increasing")
    xs, ys, positions, features = [], [], [], []
    excluded = Counter()
    for symbol in symbols:
        if symbol in Q_SYMBOLS:
            excluded["q_target"] += 1
        elif symbol not in BEAT_SYMBOLS:
            excluded["non_target_annotation"] += 1
    for i, (peak, symbol) in enumerate(zip(peaks, beat_labels)):
        if symbol not in SYMBOL_TO_CLASS:
            continue
        if peak < 128 or peak + 128 > len(signal):
            excluded["window_boundary"] += 1
            continue
        if i == 0 or i == len(peaks) - 1:
            excluded["missing_rr_context"] += 1
            continue
        x = signal[peak - 128:peak + 128]
        if not np.isfinite(x).all():
            excluded["nonfinite_signal"] += 1
            continue
        pre = (peak - peaks[i - 1]) / fs
        post = (peaks[i + 1] - peak) / fs
        local = np.diff(peaks[max(0, i - 10):i + 1]).mean() / fs
        xs.append(x.astype(np.float32)[None, :])
        ys.append(SYMBOL_TO_CLASS[symbol])
        positions.append(peak)
        features.append((pre, post, local, pre / local))
    n = len(ys)
    arrays = {
        "X": np.stack(xs) if n else np.empty((0, 1, 256), dtype=np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "pid": np.full(n, int(record_id), dtype=np.int64),
        "subject_id": np.full(n, subject_id(record_id), dtype=np.int64),
        "sample": np.asarray(positions, dtype=np.int64),
        "rr_raw": np.asarray(features, dtype=np.float32).reshape(-1, 4),
    }
    return arrays, dict(excluded)


def collect_records(raw_dir):
    import wfdb
    raw_dir = Path(raw_dir)
    hashes = verify_raw(raw_dir)
    pieces, record_audit = [], []
    for rid in RECORDS:
        record = wfdb.rdrecord(str(raw_dir / str(rid)), physical=True)
        annotation = wfdb.rdann(str(raw_dir / str(rid)), "atr")
        if record.sig_name.count("MLII") != 1:
            raise ValueError(f"{rid}: exactly one MLII channel required: {record.sig_name}")
        channel = record.sig_name.index("MLII")  # record 114 has swapped leads
        arrays, exclusions = segment_record(record.p_signal[:, channel],
                                             annotation.sample, annotation.symbol,
                                             rid, record.fs)
        pieces.append(arrays)
        record_audit.append({"pid": rid, "subject_id": subject_id(rid),
                             "lead_index": channel, "lead_name": "MLII",
                             "fs": record.fs, "n": len(arrays["y"]),
                             "classes": class_counts(arrays["y"]),
                             "annotation_symbols": dict(Counter(annotation.symbol)),
                             "excluded": exclusions})
        print(f"Segmented {rid}: {len(arrays['y'])} beats, MLII channel {channel}", flush=True)
    data = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
    return data, record_audit, hashes


def verify_raw(raw_dir):
    """Verify cloned raw files offline against the publisher's checksum list."""
    raw_dir = Path(raw_dir)
    checksum_file = raw_dir / "SHA256SUMS.txt"
    if not checksum_file.is_file():
        raise FileNotFoundError("Missing SHA256SUMS.txt; run the download command first")
    expected = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip().lstrip("*")] = digest
    hashes = {}
    for rid in RECORDS:
        for ext in ("hea", "dat", "atr"):
            name = f"{rid}.{ext}"
            digest = sha256(raw_dir / name)
            if digest != expected.get(name):
                raise ValueError(f"Source checksum mismatch: {name}")
            hashes[name] = digest
    return hashes


def pipeline_hashes():
    return {name: sha256(Path(__file__).with_name(name))
            for name in ("data.py", "augmentation.py")}


def class_counts(y):
    return {name: int(np.count_nonzero(y == i)) for i, name in enumerate(CLASS_NAMES)}


def stratified_partition(indices, y, fraction, seed):
    """Split off a per-class fraction; retain both sides for non-singletons."""
    rng = np.random.default_rng(seed)
    left, right = [], []
    for label in range(4):
        group = np.array(indices[y[indices] == label], copy=True)
        rng.shuffle(group)
        n = int(round(len(group) * fraction))
        if len(group) > 1:
            n = max(1, min(n, len(group) - 1))
        else:
            n = 0
        right.extend(group[:n])
        left.extend(group[n:])
    left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    rng.shuffle(left)
    rng.shuffle(right)
    return left, right


def make_splits(data, seed=42, val_records=DEFAULT_VAL):
    if len(val_records) != 4 or len(set(val_records)) != 4:
        raise ValueError("choose exactly four distinct validation records")
    if not set(val_records) <= set(DS1) - {201}:
        raise ValueError("validation must be in DS1 and cannot contain shared subject 201")
    indices = np.arange(len(data["y"]), dtype=np.int64)
    development, random_test = stratified_partition(indices, data["y"], 0.2, seed)
    random_train, random_val = stratified_partition(development, data["y"], 0.2, seed + 1)
    patient_val = indices[np.isin(data["pid"], val_records)]
    patient_train = indices[np.isin(data["pid"], sorted(set(DS1) - set(val_records)))]
    patient_test = indices[np.isin(data["pid"], DS2)]
    strict_train = patient_train[data["pid"][patient_train] != 201]
    return {"random": (random_train, random_val, random_test),
            "patient": (patient_train, patient_val, patient_test),
            "patient_strict": (strict_train, patient_val, patient_test)}


def fit_normalizer(data, train):
    """Scalar waveform and per-feature RR statistics, TRAINING subset only."""
    x = data["X"][train]
    rr = data["rr_raw"][train]
    return {"x_mean": float(x.mean(dtype=np.float64)),
            "x_std": max(float(x.std(dtype=np.float64)), 1e-8),
            "rr_mean": rr.mean(axis=0, dtype=np.float64).tolist(),
            "rr_std": np.maximum(rr.std(axis=0, dtype=np.float64), 1e-8).tolist()}


def subset(data, indices, stats):
    result = {key: value[indices].copy() for key, value in data.items()}
    result["X"] = ((result["X"] - stats["x_mean"]) / stats["x_std"]).astype(np.float32)
    result["rr"] = ((result["rr_raw"] - np.asarray(stats["rr_mean"]))
                    / np.asarray(stats["rr_std"])).astype(np.float32)
    return result


def validate_arrays(data):
    if not set(ARRAY_KEYS) <= set(data):
        raise ValueError(f"missing fields: {set(ARRAY_KEYS) - set(data)}")
    n = len(data["y"])
    shapes = {"X": (n, 1, 256), "rr": (n, 4), "rr_raw": (n, 4)}
    for key in ARRAY_KEYS:
        shape = shapes.get(key, (n,))
        if data[key].shape != shape or not np.isfinite(data[key]).all():
            raise ValueError(f"invalid shape or non-finite values: {key}")
    if n == 0 or not np.isin(data["y"], np.arange(4)).all():
        raise ValueError("empty partition or invalid labels")
    for key in ("X", "rr", "rr_raw"):
        if data[key].dtype != np.float32:
            raise ValueError(f"{key} must be float32")
    for key in ("y", "pid", "subject_id", "sample"):
        if data[key].dtype != np.int64:
            raise ValueError(f"{key} must be int64")
    if not np.array_equal(data["subject_id"], [subject_id(r) for r in data["pid"]]):
        raise ValueError("subject identity mismatch")
    if not np.isin(data["pid"], RECORDS).all() or np.any(data["sample"] < 128):
        raise ValueError("unexpected record or sample location")
    if not (data["rr_raw"] > 0).all():
        raise ValueError("RR intervals and ratios must be positive")
    if len(set(zip(data["pid"].tolist(), data["sample"].tolist()))) != n:
        raise ValueError("duplicate beat identities within partition")


def load_split(path):
    """Load numeric arrays without pickle; preserve saved row order for M3."""
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ARRAY_KEYS}
    validate_arrays(arrays)
    return arrays


def split_summary(data):
    return {"n": len(data["y"]), "classes": class_counts(data["y"]),
            "records": sorted(set(map(int, data["pid"]))),
            "subjects": sorted(set(map(int, data["subject_id"])))}


def overlap_summary(parts):
    report = {}
    for i, j in ((0, 1), (0, 2), (1, 2)):
        a, b = parts[i], parts[j]
        beat_a = set(zip(a["pid"].tolist(), a["sample"].tolist()))
        beat_b = set(zip(b["pid"].tolist(), b["sample"].tolist()))
        report[f"{('train', 'val', 'test')[i]}_{('train', 'val', 'test')[j]}"] = {
            "same_beats": len(beat_a & beat_b),
            "shared_records": sorted(set(map(int, a["pid"])) & set(map(int, b["pid"]))),
            "shared_subjects": sorted(set(map(int, a["subject_id"])) & set(map(int, b["subject_id"])))}
    return report


def build(raw_dir, output_dir, seed=42, val_records=DEFAULT_VAL):
    output_dir = Path(output_dir)
    # Never mix a new protocol with existing processed files or stale metadata.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")
    data, record_audit, hashes = collect_records(raw_dir)
    partitions = make_splits(data, seed, val_records)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "source": SOURCE,
                "data_license": DATA_LICENSE, "attribution": ATTRIBUTION,
                "pipeline_sha256": pipeline_hashes(),
                "source_sha256": hashes,
                "software": {p: version(p) for p in ("numpy", "wfdb", "scipy", "pandas")},
                "python": platform.python_version(),
                "config": {"seed": seed, "val_records": list(val_records),
                           "fs": 360, "lead": "MLII", "window": [-128, 128],
                           "filter": "none", "classes": CLASS_NAMES,
                           "class_symbols": CLASS_SYMBOLS,
                           "excluded_paced_records": PACED,
                           "rr_names": RR_NAMES, "rr_context": "offline, uses next beat",
                           "augmentation": asdict(AugmentConfig())},
                "record_audit": record_audit, "protocols": {}}
    for protocol, indices in partitions.items():
        stats = fit_normalizer(data, indices[0])
        parts = [subset(data, idx, stats) for idx in indices]
        overlaps = overlap_summary(parts)
        if any(p["same_beats"] for p in overlaps.values()):
            raise ValueError("The same beat appears in different partitions")
        if protocol == "patient_strict" and any(p["shared_subjects"] for p in overlaps.values()):
            raise ValueError("Strict protocol has overlapping subjects")
        entry = {"normalization": stats, "overlap": overlaps, "files": {}}
        for name, part in zip(("train", "val", "test"), parts):
            validate_arrays(part)
            if not all(class_counts(part["y"]).values()):
                raise ValueError(f"{protocol}/{name} must contain all four classes")
            target = output_dir / f"{protocol}_{name}.npz"
            np.savez_compressed(target, **part,
                                protocol=np.array(protocol), partition=np.array(name),
                                class_names=np.array(CLASS_NAMES), rr_names=np.array(RR_NAMES),
                                data_source=np.array(SOURCE), data_license=np.array(DATA_LICENSE),
                                attribution=np.array(ATTRIBUTION))
            entry["files"][name] = {**split_summary(part), "path": target.name,
                                    "sha256": sha256(target)}
            print(f"Saved {target.name}: {class_counts(part['y'])}", flush=True)
        s_mask = (parts[2]["y"] == 1)
        s_232 = int(np.count_nonzero(s_mask & (parts[2]["pid"] == 232)))
        entry["test_s_record232"] = {"n": s_232, "total_s": int(s_mask.sum()),
                                       "fraction": s_232 / int(s_mask.sum())}
        if protocol == "patient":
            entry["caveat"] = "Conventional DS1/DS2 shares subject 201 via records 201/202. Not strictly subject independent."
        elif protocol == "patient_strict":
            entry["caveat"] = "Record 201 removed from training; DS2 unchanged. Counts differ from conventional DS1."
        else:
            entry["caveat"] = "Intentional beat-random protocol; patients and possibly waveform windows/context overlap."
        manifest["protocols"][protocol] = entry
    write_json(output_dir / "manifest.json", manifest)
    rows = []
    for protocol, entry in manifest["protocols"].items():
        for name, info in entry["files"].items():
            rows.append({"protocol": protocol, "partition": name, "n": info["n"],
                         "records": len(info["records"]), "subjects": len(info["subjects"]),
                         **info["classes"]})
    with (output_dir / "class_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return audit(output_dir)


def audit(output_dir):
    """Independently reopen files, check hashes, labels, identities and fitting."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    report = {"status": "passed", "protocols": {}}
    for protocol, entry in manifest["protocols"].items():
        parts = []
        for partition in ("train", "val", "test"):
            info = entry["files"][partition]
            path = output_dir / info["path"]
            if sha256(path) != info["sha256"]:
                raise ValueError(f"File changed: {path}")
            part = load_split(path)
            if split_summary(part) != {key: info[key] for key in ("n", "classes", "records", "subjects")}:
                raise ValueError(f"Summary mismatch: {path}")
            parts.append(part)
        overlaps = overlap_summary(parts)
        if overlaps != entry["overlap"] or any(v["same_beats"] for v in overlaps.values()):
            raise ValueError(f"Invalid beat overlap: {protocol}")
        if protocol == "patient_strict" and any(v["shared_subjects"] for v in overlaps.values()):
            raise ValueError("Strict subject leakage")
        if protocol != "random":
            if set(parts[1]["pid"]) != set(manifest["config"]["val_records"]):
                raise ValueError("Wrong validation records")
            expected_train = set(DS1) - set(manifest["config"]["val_records"])
            if protocol == "patient_strict":
                expected_train.discard(201)
            if set(parts[0]["pid"]) != expected_train or set(parts[2]["pid"]) != set(DS2):
                raise ValueError("Wrong training or test records")
        train = parts[0]
        if abs(float(train["X"].mean(dtype=np.float64))) > 1e-5 or abs(float(train["X"].std(dtype=np.float64)) - 1) > 1e-5:
            raise ValueError("Waveform training normalisation mismatch")
        if not np.allclose(train["rr"].mean(axis=0, dtype=np.float64), 0, atol=1e-5):
            raise ValueError("RR training normalisation mismatch")
        stats = entry["normalization"]
        for part in parts:
            expected_rr = (part["rr_raw"] - stats["rr_mean"]) / stats["rr_std"]
            if not np.allclose(part["rr"], expected_rr, atol=1e-5):
                raise ValueError("RR transform mismatch")
        report["protocols"][protocol] = {"overlap": overlaps,
                                          "partitions": [split_summary(p) for p in parts]}
    write_json(output_dir / "audit.json", report)
    print("Audit passed: checksums, shapes, class statistics, splits and normalisation.", flush=True)
    return report


class ECGDataset:
    """PyTorch-compatible map dataset, with reproducible per-epoch augmentation.

    No torch dependency is needed to build/audit NPZs. Import torch only when
    retrieving tensors. Use DataLoader(persistent_workers=False) if augmenting,
    call set_epoch(epoch) before each training iterator. Evaluation must not augment.
    """
    def __init__(self, path, augment=False, seed=42, augment_config=None):
        self.path = Path(path)
        self.arrays = load_split(path)
        with np.load(path, allow_pickle=False) as archive:
            self.protocol = str(archive["protocol"])
            self.partition = str(archive["partition"])
        if augment and self.partition != "train":
            raise ValueError("Augmentation is allowed only on training data")
        self.augment = augment
        self.seed, self.epoch = int(seed), 0
        self.augment_config = augment_config or AugmentConfig()

    def __len__(self):
        return len(self.arrays["y"])

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __getitem__(self, index):
        import torch
        index = int(index)
        x = self.arrays["X"][index]
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            x = augment_beat(x, rng, self.augment_config)
        else:
            x = x.copy()
        return {"X": torch.from_numpy(x),
                "y": torch.tensor(int(self.arrays["y"][index]), dtype=torch.long),
                "rr": torch.from_numpy(self.arrays["rr"][index].copy()),
                "pid": int(self.arrays["pid"][index]),
                "subject_id": int(self.arrays["subject_id"][index]),
                "sample": int(self.arrays["sample"][index])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="Download 44 records with SHA256 validation")
    dl.add_argument("--raw-dir", type=Path, default=Path("data/raw/mitdb"))
    dl.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    prep = sub.add_parser("build", help="Build random, conventional and strict patient splits")
    prep.add_argument("--raw-dir", type=Path, default=Path("data/raw/mitdb"))
    prep.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--val-records", type=int, nargs=4, default=DEFAULT_VAL)
    check = sub.add_parser("audit", help="Independently validate generated artifacts")
    check.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    if args.command == "download":
        download(args.raw_dir, args.workers)
    elif args.command == "build":
        warnings.warn("Conventional DS1/DS2 shares subject 201/202; use patient_strict for strict unseen-subject claims.")
        build(args.raw_dir, args.output_dir, args.seed, tuple(args.val_records))
    else:
        audit(args.output_dir)


if __name__ == "__main__":
    main()
