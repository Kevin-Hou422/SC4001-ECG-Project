"""Prepare or reuse data safely, including a fresh Git clone with metadata only."""
import argparse
import json
from pathlib import Path
import tempfile

from .data import (DEFAULT_VAL, audit, build, pipeline_hashes, verify_raw)


REFERENCE_FILES = {"manifest.json", "audit.json", "class_statistics.csv"}


def ensure_prepared(raw_dir, output_dir, seed=42, val_records=DEFAULT_VAL):
    """Build missing arrays; audit complete arrays; refuse partial/stale caches.

    Git tracks reference metadata but ignores large NPZs. A metadata-only clone
    must rebuild, not attempt to load nonexistent arrays. Build in a temporary
    sibling directory before publishing any arrays to the requested location.
    """
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    hashes = verify_raw(raw_dir)
    manifest_path = output_dir / "manifest.json"
    prior = None
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (prior.get("pipeline_sha256") != pipeline_hashes()
                or prior["config"]["seed"] != seed
                or prior["config"]["val_records"] != list(val_records)
                or prior.get("source_sha256") != hashes):
            raise ValueError("Existing metadata use different source/configuration. Choose a new empty output directory.")
        files = [output_dir / info["path"] for protocol in prior["protocols"].values()
                 for info in protocol["files"].values()]
        present = [path.is_file() for path in files]
        if all(present):
            return audit(output_dir)
        if any(present):
            raise ValueError("Incomplete processed cache. Choose a new empty output directory; no arrays were overwritten.")
    existing = {path.name for path in output_dir.iterdir()} if output_dir.exists() else set()
    if existing - REFERENCE_FILES or (existing and prior is None):
        raise ValueError("Output directory contains unrecognised/incomplete files. Choose a new empty directory.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory owns only this newly created sibling, never user data.
    with tempfile.TemporaryDirectory(prefix=".m1_build_", dir=output_dir.parent) as staging:
        built = Path(staging) / "processed"
        build(raw_dir, built, seed, tuple(val_records))
        output_dir.mkdir(parents=True, exist_ok=True)
        # Target has no existing NPZ files. Metadata replacement is deliberate:
        # byte checksums may depend on package versions, counts/protocol do not.
        for path in sorted(built.iterdir(), key=lambda p: p.name == "manifest.json"):
            path.replace(output_dir / path.name)
    return audit(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mitdb"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_prepared(args.raw_dir, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
