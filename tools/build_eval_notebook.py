"""Generate the M3 evaluation notebook (same plain-JSON approach as build_notebook.py)."""
import json
from pathlib import Path


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(True)}


cells = [
markdown('''
# SC4001 ECG Project: M3 Evaluation

Run top to bottom in **Google Colab** (CPU is enough). This notebook:

1. rebuilds and audits the processed data,
2. describes the labelled data (class concentration per record, S-beat timing),
3. predicts with the committed checkpoints (or loads existing predictions),
4. evaluates every `results/<experiment>/seed<k>.csv`: per-class metrics,
   mean ± SD over seeds, per-patient tables, results without record 232 (S)
   and 213 (F), patient-bootstrap intervals, and the post-hoc S ranking and
   threshold analyses.

The prediction CSV contract is `pid,y_true,y_pred,p_N,p_S,p_V,p_F` in test-NPZ
row order. Dummy predictions are only for testing the pipeline and every output
made from them is stamped as dummy.
'''),
markdown('''
## 1. Clone the group repository
'''),
code('''
from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/Kevin-Hou422/SC4001-ECG-Project.git"
BRANCH = "main"  # use "m3-evaluation" until that branch is merged
ROOT = Path("/content/SC4001-ECG-Project")

if not ROOT.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, str(ROOT)], check=True)
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print("Commit:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
'''),
code('''
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
'''),
markdown('''
## 2. Prepare and audit the NPZ files

M1's `ensure_prepared` builds the arrays on a fresh clone (which has metadata
but no NPZs) and audits them; an existing complete cache is audited and reused.
It refreshes the committed metadata files: do not commit those changes.
'''),
code('''
import shutil
from src.prepare import ensure_prepared

PROCESSED_DIR = ROOT / "data/processed"
ensure_prepared(ROOT / "data/raw/mitdb", PROCESSED_DIR)
'''),
code('''
subprocess.run([sys.executable, "-m", "unittest", "tests.test_evaluate", "tests.test_saliency"], check=True)
'''),
markdown('''
## 3. Describe the labelled data (not model results)
'''),
code('''
import json
from IPython.display import Image, Markdown, display
from src.evaluate import describe_data
from src.figures import data_figures

DATA_OUT = ROOT / "results/data_analysis"
facts = describe_data(PROCESSED_DIR, DATA_OUT, ROOT / "data/raw/mitdb")
data_figures(PROCESSED_DIR, DATA_OUT / "figures")
for name in ("patient_train.npz", "patient_test.npz"):
    print(name, json.dumps(facts[name]["concentration"], indent=1))
for figure in ("class_concentration", "rr_prematurity_S", "templates_232"):
    display(Image(str(DATA_OUT / "figures" / f"{figure}.png"), width=800))
'''),
markdown('''
## 4. Bring in the prediction CSVs

Choose one source:
- `"checkpoints"`: predict with the committed E0–E5 checkpoints (`src.predict`,
  about 2 minutes on CPU). Each checkpoint must first reproduce its stored
  validation macro-F1 on this runtime's data.
- `"runtime"`: predictions already written to `results/` in this runtime by training.
- `"drive"`: copy `results/` from a shared Google Drive folder (asks for permission).
- `"dummy"`: fake predictions to try the pipeline. **Never report these.**
'''),
code('''
SOURCE = "checkpoints"  # "checkpoints", "runtime", "drive" or "dummy"
DRIVE_RESULTS = "/content/drive/MyDrive/SC4001_ECG/results"
RESULTS_DIR = ROOT / "results"

if SOURCE == "checkpoints":
    subprocess.run([sys.executable, "-m", "src.predict", "--experiments", "all",
                    "--seeds", "42", "43", "44"], check=True)
elif SOURCE == "drive":
    from google.colab import drive
    drive.mount("/content/drive")
    shutil.copytree(DRIVE_RESULTS, RESULTS_DIR, dirs_exist_ok=True)
elif SOURCE == "dummy":
    from src.evaluate import write_dummy_results
    RESULTS_DIR = ROOT / "results/_dummy"
    write_dummy_results(PROCESSED_DIR, RESULTS_DIR)
print(sorted(p.name for p in RESULTS_DIR.iterdir() if p.is_dir()))
'''),
markdown('''
## 5. Evaluate all experiments

`REFERENCE` is the experiment that the fixes are compared with (paired patient
bootstrap). Outputs: `metrics_by_seed.csv`, `metrics_mean_std.csv`,
`per_patient_<exp>.csv`, `top_confusions_<exp>.csv`, `bootstrap.csv`,
`summary.md` and `figures/`.
'''),
code('''
from src.evaluate import summarise

REFERENCE = "E1" if SOURCE != "dummy" else "E1_dummy"
SUMMARY_DIR = RESULTS_DIR / "summary" if SOURCE == "dummy" else ROOT / "results/summary"
summary = summarise(RESULTS_DIR, PROCESSED_DIR, SUMMARY_DIR, reference=REFERENCE)
display(Markdown((SUMMARY_DIR / "summary.md").read_text()))
'''),
code('''
for png in sorted((SUMMARY_DIR / "figures").glob("*.png")):
    print(png.name)
    display(Image(str(png), width=700))
'''),
markdown('''
## 6. Optional: save the summary to Google Drive
'''),
code('''
SAVE_TO_DRIVE = False
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    target = Path("/content/drive/MyDrive/SC4001_ECG") / SUMMARY_DIR.relative_to(ROOT)
    shutil.copytree(SUMMARY_DIR, target, dirs_exist_ok=True)
    print("Saved to", target)
'''),
]

notebook = {"cells": cells, "metadata": {
    "colab": {"name": "03_M3_Evaluation.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"}},
    "nbformat": 4, "nbformat_minor": 5}
for index, cell in enumerate(cells):
    cell["id"] = f"m3-cell-{index:02d}"
root = Path(__file__).resolve().parents[1]
target = root / "notebooks/03_M3_Evaluation.ipynb"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(target)
