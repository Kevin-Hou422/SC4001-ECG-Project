"""Generate the Colab entry point without requiring notebook authoring libraries."""
import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(True)}


cells = [
markdown('''
# SC4001 ECG Project — M1 Data Setup

Run top to bottom in **Google Colab**. This notebook clones the group repository
with its 44 original MIT-BIH records, verifies them, generates model inputs and
checks the M2 interface. **CPU is sufficient** for this data preparation notebook.

Source: [MIT-BIH v1.0.0, Moody & Mark / PhysioNet](https://doi.org/10.13026/C2F305).
Data terms: [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/).
No classification performance is claimed by running setup.
'''),
markdown('''
## 1. Clone the group repository

An existing checkout is reused without overwriting changes. For updated code,
use a fresh runtime or review and pull changes yourself.
'''),
code('''
from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/Kevin-Hou422/SC4001-ECG-Project.git"
BRANCH = "main"
ROOT = Path("/content/SC4001-ECG-Project")

if not ROOT.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(ROOT)], check=True)
elif not (ROOT / ".git").is_dir():
    raise RuntimeError("Existing ROOT is not a Git checkout; choose a fresh path.")
else:
    remote = subprocess.check_output(["git", "-C", str(ROOT), "remote", "get-url", "origin"], text=True).strip()
    if remote != REPO_URL:
        raise RuntimeError(f"Unexpected existing repository: {remote}")

os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Repository:", ROOT)
print("Commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
'''),
markdown('''
## 2. Install data dependencies

PyTorch normally comes with Colab. This step installs WFDB and the declared data
dependencies; it does not replace your GPU-specific PyTorch build.
'''),
code('''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
import importlib.metadata as metadata
print({name: metadata.version(name) for name in ["numpy", "wfdb", "scipy", "pandas"]})
'''),
markdown('''
## 3. Choose data locations and the experiment protocol

`patient` preserves the planned DS1/DS2 benchmark. **It shares subject 201 through
records 201/202**. Choose `patient_strict` for a strictly unseen-subject experiment
(record 201 is removed from training, DS2 stays unchanged). All three protocols
are generated, but use a consistent prefix for train/validation/test.

Data-split seed 42 stays fixed across M2's three model-training seeds.
'''),
code('''
SEED = 42
PROTOCOL = "patient"  # "random", "patient", or "patient_strict"
RAW_DIR = ROOT / "data/raw/mitdb"
PROCESSED_DIR = ROOT / "data/processed"
assert PROTOCOL in {"random", "patient", "patient_strict"}
'''),
markdown('''
## 4. Verify the included original dataset

No second download is needed. All 132 raw files are checked against the publisher's
SHA256 list. `pid` means recording ID; `subject_id` merges records 201 and 202.
'''),
code('''
from src.data import verify_raw, RECORDS
hashes = verify_raw(RAW_DIR)
raw_bytes = sum((RAW_DIR / name).stat().st_size for name in hashes)
print(f"Verified {len(RECORDS)} records / {len(hashes)} files / {raw_bytes / 1e6:.2f} MB")
'''),
markdown('''
## 5. Run protocol tests

Tests use synthetic signals and do not count as real experimental results.
They cover labels, RR timing, subject separation, train-only normalisation,
augmentation safeguards and the PyTorch batch interface when available.
'''),
code('''
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
'''),
markdown('''
## 6. Build and independently audit model inputs

The first run segments MLII and generates nine NPZ files, even though the cloned
repository already contains reference metadata. Subsequent runs audit complete
data rather than silently overwriting it. For changed processing code,
choose a new empty `PROCESSED_DIR`. Normalisation is fitted separately on each
protocol's training subset; no fitted DS2 statistics enter preprocessing.
'''),
code('''
import json
from src.prepare import ensure_prepared

manifest_path = PROCESSED_DIR / "manifest.json"
ensure_prepared(RAW_DIR, PROCESSED_DIR, seed=SEED)

manifest = json.loads(manifest_path.read_text())
print("Active protocol:", PROTOCOL)
print(manifest["protocols"][PROTOCOL]["caveat"])
'''),
markdown('''
## 7. Inspect actual counts and training waveforms

Counts differ from the preliminary plan because incomplete windows and beats
without pre/post RR context are excluded. Next-RR uses future context, so this is
**offline annotated-beat classification**, not a real-time beat detector.
'''),
code('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data import load_split, CLASS_NAMES

display(pd.read_csv(PROCESSED_DIR / "class_statistics.csv"))
print("DS2 S-class concentration:", manifest["protocols"]["patient"]["test_s_record232"])

arrays = load_split(PROCESSED_DIR / f"{PROTOCOL}_train.npz")
fig, axes = plt.subplots(2, 2, figsize=(11, 5), constrained_layout=True)
time_ms = (np.arange(256) - 128) / 360 * 1000
for label, ax in enumerate(axes.flat):
    idx = np.flatnonzero(arrays["y"] == label)[0]
    ax.plot(time_ms, arrays["X"][idx, 0], linewidth=1)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set(title=f"{CLASS_NAMES[label]}: record {arrays['pid'][idx]}",
           xlabel="Time from annotation centre (ms)", ylabel="Standardised MLII")
plt.show()
'''),
markdown('''
## 8. Check the M2 PyTorch interface

For E4 set `augment=True` **only for training**. Call `set_epoch(epoch)` before
each epoch and keep `persistent_workers=False`. RR is already standardised;
only E5's model should consume it. These cells only inspect a batch, not train.
'''),
code('''
import torch
from torch.utils.data import DataLoader
from src.data import ECGDataset

train_dataset = ECGDataset(PROCESSED_DIR / f"{PROTOCOL}_train.npz", augment=False, seed=42)
val_dataset = ECGDataset(PROCESSED_DIR / f"{PROTOCOL}_val.npz")
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(42))
batch = next(iter(train_loader))
print({key: (tuple(value.shape), str(value.dtype)) for key, value in batch.items()})
assert batch["X"].shape == (128, 1, 256)
assert batch["rr"].shape == (128, 4)
assert batch["y"].dtype == torch.long
print("PyTorch:", torch.__version__, "GPU available:", torch.cuda.is_available())
'''),
markdown('''
## 9. Optional: back up processed data to your Google Drive

Set `BACKUP_TO_DRIVE=True` to run this step. Google will ask for your permission
to mount Drive. The archive name includes the data manifest hash. Share the archive
with teammates using Drive's own sharing controls. No sharing permissions are
changed by this notebook. Runtime-local files otherwise disappear on recycle.
'''),
code('''
BACKUP_TO_DRIVE = False
if BACKUP_TO_DRIVE:
    from google.colab import drive
    import shutil
    from src.data import sha256
    drive.mount("/content/drive")
    backup_dir = Path("/content/drive/MyDrive/SC4001_ECG")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stem = "processed_" + sha256(manifest_path)[:12]
    archive = backup_dir / f"{stem}.zip"
    if not archive.exists():
        local_archive = shutil.make_archive(str(ROOT / stem), "zip", root_dir=PROCESSED_DIR)
        shutil.copy2(local_archive, archive)
    print("Backup:", archive)
else:
    print("Drive backup skipped. Data remain in this Colab runtime.")
'''),
markdown('''
## Handoff and next steps

- M2: implement the 1D CNN/training loop and run E0-E5 with three model seeds.
- M3: evaluate the fixed test set, with per-class and per-patient metrics, plus
  S-class results with and without record 232. No selection/tuning on DS2.
- M1 stretch: autoencoder pretraining uses training patients only, excluding
  validation and DS2; it is not included in this setup.
- Before writing results, confirm which patient protocol the team will use.
- See `README.md` and `docs/M1_HANDOFF.md` for the full interface and caveats.

Use **File → Save a copy in Drive** to keep your own notebook with outputs.
Review outputs before sharing; this repository's notebook starts without outputs.
'''),
]

notebook = {"cells": cells, "metadata": {
    "colab": {"name": "01_M1_Data_Setup.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"m1-cell-{index:02d}"
root = Path(__file__).resolve().parents[1]
target = root / "notebooks/01_M1_Data_Setup.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(target)
