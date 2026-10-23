"""Generate a standalone Colab notebook for M1's E6 exports and reproduction."""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(True)}


cells = [md('''
# SC4001 M1 — E6 Autoencoder Pretraining

This standalone notebook checks M1's six pretrained encoder exports, prepares
matching data and demonstrates M2's encoder interface. Optionally rerun the fixed
ten-epoch pretraining. CPU is supported; GPU is optional. All losses shown here
are **training reconstruction diagnostics, not classification performance**.

The published runs used seeds 42/43/44 for both `patient` and `patient_strict`.
No validation/test waveform is used in pretraining or epoch selection.
'''), md('''
## 1. Clone source, data and encoder exports
An existing checkout is reused without resetting user changes.
'''), code('''
from pathlib import Path
import os, sys, subprocess, json
REPO_URL = "https://github.com/Kevin-Hou422/SC4001-ECG-Project.git"
ROOT = Path("/content/SC4001-ECG-Project")
if not ROOT.exists():
    subprocess.run(["git", "clone", "--branch", "main", REPO_URL, str(ROOT)], check=True)
elif not (ROOT / ".git").is_dir():
    raise RuntimeError("Existing ROOT is not a Git checkout; choose another location.")
else:
    remote = subprocess.check_output(["git", "-C", str(ROOT), "remote", "get-url", "origin"], text=True).strip()
    if remote != REPO_URL:
        raise RuntimeError(f"Unexpected checkout: {remote}")
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
print("Source commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
'''), md('''
## 2. Install dependencies and choose protocol
Use one protocol consistently. The conventional `patient` protocol shares the
201/202 subject; `patient_strict` excludes training record 201. The pretraining
seed matches the export seed, while the data-split seed remains 42.
'''), code('''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-pretrain.txt"], check=True)
PROTOCOL = "patient"  # alternatively: patient_strict
SEEDS = [42, 43, 44]
EPOCHS = 10
PROCESSED = ROOT / "data/processed"
from src.prepare import ensure_prepared
ensure_prepared(ROOT / "data/raw/mitdb", PROCESSED, seed=42)
assert PROTOCOL in {"patient", "patient_strict"}
'''), md('''
## 3. Audit the published exports
Checks cohort, preprocessing, fitted statistics, file/tensor hashes and forward
shape. An exact NPZ byte-hash match is reported separately from semantic data
compatibility, since rebuilt containers/last-bit reductions can differ by platform.
'''), code('''
from src.pretrain_audit import audit_export
audits = []
for seed in SEEDS:
    result = audit_export(ROOT / f"artifacts/pretraining/{PROTOCOL}/seed{seed}",
                          PROCESSED / f"{PROTOCOL}_train.npz",
                          PROCESSED / "manifest.json", seed, EPOCHS)
    audits.append(result)
    print(seed, result)
'''), md('''
## 4. Optional rerun and interruption recovery
The default reuses the audited published exports. Set `RUN_PRETRAINING=True` to
run all three seeds on all training beats, with a fixed ten-epoch budget. For
interruption recovery, retain the output directory and set `RESUME=True`.

To persist **each epoch** in Google Drive, manually mount Drive first and set
`OUTPUT_DIR` to a folder there. The notebook does not mount Drive or change sharing
permissions automatically. Use a new output directory for independent reruns.
'''), code('''
RUN_PRETRAINING = False
RESUME = False
OUTPUT_DIR = ROOT / "local_pretraining"
DEVICE = "cpu"  # optionally cuda after choosing a GPU runtime
if RUN_PRETRAINING:
    from src.pretrain import PretrainConfig, pretrain
    for seed in SEEDS:
        summary = pretrain(PROCESSED / f"{PROTOCOL}_train.npz", PROCESSED / "manifest.json",
                           OUTPUT_DIR / PROTOCOL / f"seed{seed}",
                           PretrainConfig(seed=seed, epochs=EPOCHS, device=DEVICE), resume=RESUME)
        print(seed, summary["initial_train_mse"], summary["final_train_mse"])
else:
    print("Published exports reused; pretraining rerun skipped.")
'''), md('''
## 5. Demonstrate the transfer interface
For a valid supervised E1/E6 comparison, M2 must use the identical encoder and
classification head in both conditions, changing only encoder initialisation.
This cell does not train or evaluate a heartbeat classifier.
'''), code('''
import torch
from src.data import load_split, sha256
from src.pretrain import load_pretrained_encoder
selected_seed = SEEDS[0]
folder = (OUTPUT_DIR if RUN_PRETRAINING else ROOT / "artifacts/pretraining") / PROTOCOL / f"seed{selected_seed}"
result = audit_export(folder, PROCESSED / f"{PROTOCOL}_train.npz", PROCESSED / "manifest.json", selected_seed, EPOCHS)
exact_hash = sha256(PROCESSED / f"{PROTOCOL}_train.npz") if result["exact_training_npz_sha256_match"] else None
if exact_hash is None:
    print("NPZ bytes differ; audited cohort/source/normalisation match. Rerun pretraining for exact-file provenance.")
encoder = load_pretrained_encoder(folder / "encoder.pt", expected_train_sha256=exact_hash)
encoder.eval()
waveforms = load_split(PROCESSED / f"{PROTOCOL}_train.npz")["X"][:4]
with torch.inference_mode():
    feature_maps = encoder(torch.from_numpy(waveforms))
    embeddings = encoder.embedding(torch.from_numpy(waveforms))
print("Feature maps:", tuple(feature_maps.shape), "embeddings:", tuple(embeddings.shape))
assert feature_maps.shape == (4, 64, 32)
assert embeddings.shape == (4, 64)
'''), md('''
## 6. Inspect training diagnostics and save a run record
Reconstruction error alone does not prove a minority-class or generalisation
improvement. M2/M3 must still perform the supervised E6 experiment.
'''), code('''
import matplotlib.pyplot as plt
import platform
fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
for seed in SEEDS:
    folder = (OUTPUT_DIR if RUN_PRETRAINING else ROOT / "artifacts/pretraining") / PROTOCOL / f"seed{seed}"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    ax.plot([row["epoch"] for row in summary["history"]],
            [row["train_online_mse"] for row in summary["history"]], label=f"seed {seed}")
ax.set(xlabel="Epoch", ylabel="Online training reconstruction MSE", yscale="log", title=PROTOCOL)
ax.legend()
plt.show()
run_record = {"protocol": PROTOCOL, "seeds": SEEDS, "pretraining_rerun": RUN_PRETRAINING,
              "python": platform.python_version(), "platform": platform.platform(),
              "torch": str(torch.__version__), "export_audits": audits,
              "classification_evaluated": False}
Path("m1_e6_notebook_run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
print("Saved m1_e6_notebook_run.json. Save a notebook copy with outputs in Drive for the team record.")
'''), md('''
## Handoff
M1 supplies pretraining source, reproducible settings, six encoder exports, audit
records and report paragraphs. See `docs/E6_PRETRAINING.md` for exact transfer and
resume instructions. A human teammate's code review and Google runtime evidence
must be recorded after they actually occur.
''')]
for index, cell in enumerate(cells):
    cell["id"] = f"e6-cell-{index:02d}"
notebook = {"cells": cells, "metadata": {
    "colab": {"name": "02_M1_Autoencoder_Pretraining.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5}
target = Path(__file__).resolve().parents[1] / "notebooks/02_M1_Autoencoder_Pretraining.ipynb"
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(target)
