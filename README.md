# SC4001: ECG Generalisation to Unseen Patients

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kevin-Hou422/SC4001-ECG-Project/blob/main/notebooks/01_M1_Data_Setup.ipynb)

M1 data pipeline for the group project. The repository includes the **44 raw
MIT-BIH records used by this project** (about 86 MB), their reference heartbeat
annotations, publisher checksums, and a notebook that generates the model inputs.
The four paced records (102, 104, 107, 217) are deliberately excluded. This is a
project subset of the 48-record database, not a newly collected dataset.

## Start in Google Colab

1. Click **Open in Colab**, sign into your Google account and connect a runtime.
2. Run the notebook from top to bottom. **CPU is sufficient for M1**; M2 can
   choose a GPU for training later.
3. It clones this repository, installs the data dependencies, verifies raw data,
   runs protocol tests, builds the `.npz` files and audits them.
4. Use `data/processed/patient_{train,val,test}.npz` for the agreed conventional
   DS1/DS2 experiments. For a strictly unseen-subject experiment use
   `patient_strict_{train,val,test}.npz` consistently for all three partitions.
5. The last notebook section optionally backs up processed data to Google Drive.
   Colab runtime files disappear when the runtime is recycled; GitHub source and
   the original raw data remain available for rebuilding.

The notebook is a setup and handoff notebook. It does **not** train the group's
CNN or report invented performance results. M2 owns training; M3 owns evaluation.

## Repository layout

```text
notebooks/01_M1_Data_Setup.ipynb   # Colab entry point
src/data.py                     # download, verify, segment, split, audit, dataset
src/prepare.py                  # safe first-run build / existing-cache reuse
src/augmentation.py             # training-only waveform augmentation
tests/test_data.py              # synthetic protocol tests, not reported metrics
data/raw/mitdb/                  # 44 original .hea/.dat/.atr triplets
data/processed/                  # generated NPZs ignored by Git; audit metadata kept
docs/project_plan.md            # supplied project plan, preserved verbatim
docs/M1_HANDOFF.md              # interface and experiment caveats for M2/M3
docs/report/                    # M1 methods, related work and AI-use drafts
docs/DATA_SOURCES.md            # attribution and dataset terms
results/                        # M2/M3 outputs (not versioned)
requirements.txt                # portable data environment
requirements-lock.txt           # exact locally tested M1 environment
```

`src/models.py`, `src/train.py` and `src/evaluate.py` will be added by M2/M3.
Checkpoints and generated arrays are not repeatedly committed. Raw source files
are unchanged and each is under 2 MB; Git LFS is unnecessary for this subset.

## Data contract

Each `<protocol>_<partition>.npz` contains the following numeric arrays, safely
loadable with `np.load(path, allow_pickle=False)`:

| Key | Shape / dtype | Meaning |
|---|---|---|
| `X` | `(n, 1, 256)`, float32 | MLII waveform, standardised using training statistics |
| `y` | `(n,)`, int64 | N=0, S=1, V=2, F=3 |
| `pid` | `(n,)`, int64 | Recording ID, **not necessarily a unique patient** |
| `subject_id` | `(n,)`, int64 | Same as `pid`, except 202 maps to subject 201 |
| `sample` | `(n,)`, int64 | Original heartbeat annotation location in samples |
| `rr` | `(n, 4)`, float32 | RR features, standardised using training statistics |
| `rr_raw` | `(n, 4)`, float32 | Unstandardised seconds / ratio values |

The archive also has `protocol`, `partition`, `class_names` and `rr_names` fields.
Do not sort rows independently of labels/predictions. `(pid, sample)` uniquely
identifies a beat; preserve this pair for traceable analysis when useful.

### Preprocessing

- WFDB physical units, MLII selected **by channel name**, including channel 1
  for record 114. The second lead remains in the original raw files but is unused.
- Official expert-reviewed beat annotations provide centres; this project does
  not implement a heartbeat detector. Annotation centres are treated as R peaks.
- Window `[centre - 128, centre + 128)` at 360 Hz: 256 samples, about 0.711 s.
- No resampling, bandpass filter or per-beat normalisation. Scalar waveform
  mean/population SD are fitted on training beats only, then frozen.
- Mapping: N={N,L,R,e,j}, S={A,a,J,S}, V={V,E}, F={F}. Paced and unknown
  targets {/,f,Q} are dropped. Other annotations (rhythm, noise, flutter markers,
  etc.) are not supervised beat targets. N is a superclass, not only normal sinus.
- Drop incomplete windows, non-finite windows, and beats without pre/post RR
  context. Consequently counts differ from the preliminary plan; see the generated
  `class_statistics.csv` and per-record exclusion counters in `manifest.json`.

### RR features (E5)

1. Previous RR interval in seconds.
2. Next RR interval in seconds.
3. Mean of the last up to ten RR intervals, including the current previous RR.
4. Previous RR divided by that local mean.

Intervals are computed on consecutive **beat annotations**, never on rhythm
markers, never across recordings, and without class labels. Q beats remain valid
timing context even though they are not classification targets. Per-feature mean
and population SD are fitted on the relevant training partition only.
The next RR uses future context: **this is offline classification**, not a
validated causal/real-time detector. No labels or fitted test-set statistics enter
the feature calculation. The random-beat protocol can share adjacent context.

## Three reproducible protocols

All protocols are generated together. The split seed is **42**, independent of
M2's three training seeds; keep the data split fixed while varying model seeds.

| Prefix | Train / validation / test | Interpretation |
|---|---|---|
| `random` | Stratified 80/20 development/test, then 80/20 of development for train/val (about 64/16/20 overall) | Intentionally allows shared patients; adjacent waveform/context may overlap |
| `patient` | DS1 minus four validation records / four DS1 records / DS2 | Conventional benchmark; shares one subject through records 201/202 |
| `patient_strict` | Same, but remove record 201 from training | Strict subject-disjoint partitions; DS2 test records unchanged |

Validation records are fixed to **108, 118, 203, 223**, selected on training-side
class coverage; they are not tuned using DS2. All four target classes must be
present in every saved partition. Each protocol fits its **own** training
normalisation; do not mix files across prefixes.

**Important correction to the supplied plan:** 22 recordings is not necessarily
22 independent patients across partitions. Records 201 and 202 are the same
subject. The conventional protocol remains available to honour the agreed
benchmark, but a strictly unseen-patient claim requires `patient_strict`.
Removing record 201 is disclosed; it is not the unmodified de Chazal split.

Changing the split also changes patient composition/training size. A difference
between E0/E1 should not be attributed solely to leakage, nor should a 99% score
or a performance drop be assumed before running the experiments.

## Training-only augmentation (E4)

`AugmentConfig` defines scaling 0.9-1.1, Gaussian noise SD 0.02, sinusoidal baseline
wander amplitude 0.05 at 0.1-0.5 Hz, and edge-padded shifts up to 8 samples.
Each transform has probability 0.5. Amplitudes refer to standardised waveform
units. Defaults are an initial experimental setting, not validated physiology.

Augmentation happens on the fly, never in cached validation/test arrays. It is
deterministic for `(training seed, epoch, row index)` and leaves labels/RR intact:
the transformations model measurement variability, not changed cardiac timing.
Do not normalise each augmented beat afterwards, which would cancel gain changes.

```python
from torch.utils.data import DataLoader
from src.data import ECGDataset

train = ECGDataset('data/processed/patient_train.npz', augment=True, seed=42)
val = ECGDataset('data/processed/patient_val.npz')
loader = DataLoader(train, batch_size=128, shuffle=True,
                    num_workers=0, persistent_workers=False)
for epoch in range(20):
    train.set_epoch(epoch)
    for batch in loader:
        x, y, rr = batch['X'], batch['y'], batch['rr']
        # M2: baseline takes x; the E5 model also takes rr.
```

For multi-worker augmentation, recreate workers each epoch
(`persistent_workers=False`) so they receive the current epoch. M2 should also
seed PyTorch and the DataLoader shuffle generator for reproducible training.

## Local reproduction

Python 3.10+; tested locally with Python 3.12. PyTorch is optional for M1 data
preparation; it is needed for `ECGDataset` tensor access and M2 training. Colab
normally provides PyTorch already; choose a suitable build for local hardware.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.prepare
python -m src.data audit
```

Original raw files are already included. To independently fetch verified copies
from PhysioNet into a separate directory:

```bash
python -m src.data download --raw-dir data/raw/mitdb_refetch
python -m src.data build --raw-dir data/raw/mitdb_refetch --output-dir data/processed_refetch
```

`src.prepare` handles a fresh Git clone, which contains metadata but no NPZs:
it builds missing arrays in a staging directory and then publishes them. Complete
caches are audited and reused. Partial or mismatched caches are rejected without
overwriting arrays. For a new configuration choose an **empty output directory**.
The lower-level `src.data build` still requires an empty output directory.
The manifest records raw-file checksums, software versions,
normalisation parameters, all record lists, class counts, exclusions and overlaps.
Its NPZ checksums detect later modifications. The audit reopens all files and
checks the actual arrays rather than trusting the summary alone.

The GitHub Actions workflow runs the protocol tests, prepares the real data from
a fresh Linux checkout and executes the portable notebook cells. This supplements
local verification; it is not a substitute for a Google-hosted Colab run.

## M1 report and handoff

- [Data and features draft](docs/report/M1_DATA_METHODS.md): actual retained
  counts, partitions, RR definitions, augmentation and limitations. Regenerate
  after data changes with `python tools/write_m1_report.py`.
- [Related-work contribution](docs/report/M1_RELATED_WORK.md): focused primary
  sources and the exact scope of claims verified.
- [AI-use contribution](docs/report/M1_AI_USE_DRAFT.md): merge with the team's
  actual disclosures; no independent human reproduction is claimed prematurely.

These are M1 contributions for M3 to shorten and merge into the ten-page report.
No classification scores are included before M2/M3 run the experiments.

## Attribution and AI use

MIT-BIH data retain their original attribution/licence; see
[data sources](docs/DATA_SOURCES.md). This repository does not claim authorship of
the dataset. Do not interpret this selected arrhythmia dataset as a representative
clinical population or these experiments as clinical validation.

M1 setup code, tests, documentation and the Colab notebook were drafted with
Codex assistance. Actual checks and remaining limits are recorded in
`docs/M1_HANDOFF.md`. Each member must review their contribution and the final
report's AI Use Statement must describe the team's actual tool use and checks.
