# Project Plan: ECG Generalisation to Unseen Patients

**Target:** finished by **Sun 1 Nov** (2 weeks). Official deadline is 16 Nov, so 2 Nov – 16 Nov is buffer.

## Roles

| Member | Role | Owns |
|---|---|---|
| M1: ______ | **Data & features** | Data pipeline, both splits, augmentation, RR-interval features, autoencoder pretraining (stretch), README / reproduction |
| M2: ______ | **Models & training** | 1D CNN, training loop, loss functions, running all experiments with multiple seeds, architecture comparison (stretch) |
| M3: ______ | **Evaluation & report** | Metrics, figures, per-patient + failure analysis, saliency (stretch), assembling the report |

Everyone writes the report sections for their own work, contributes to the literature review, and reviews one teammate's code each week.

## Shared conventions (agree on day 1)

These let all three of us work in parallel without waiting on each other.

- **Repo layout:** `src/data.py`, `src/models.py`, `src/train.py`, `src/evaluate.py`, `results/`, `README.md`. Shared GitHub repo + Google Drive folder for data and checkpoints.
- **Classes:** N, S, V, F (Q dropped, only 15 beats). Paced records 102, 104, 107, 217 excluded.
- **Input:** MLII lead, fixed window around each R-peak (e.g. 256 samples ≈ 0.7 s).
- **Splits:**
    - *Random split:* all beats pooled, random 80/20 (shows the inflated number).
    - *Patient split:* train = DS1 (22 patients), test = DS2 (22 patients).
    - *Validation:* hold out ~4 DS1 patients (with S and V beats) for early stopping and tuning. **DS2 is only used for final evaluation.**
- **Data file:** `data/processed/<split>.npz` with `X` (beats × 1 × 256), `y` (0–3), `pid` (record ID), `rr` (RR features).
- **Prediction file:** `results/<exp>/seed<k>.csv` with columns `pid, y_true, y_pred, p_N, p_S, p_V, p_F`.
- **Seeds:** 3 per experiment, report mean ± std.

## Experiments

| ID | Split | Change from baseline | Purpose |
|---|---|---|---|
| E0 | Random | Baseline CNN, cross-entropy | Reproduce the "99%" claim |
| E1 | Patient | Baseline CNN, cross-entropy | Measure the generalisation gap |
| E2 | Patient | + class-weighted loss | Fix 1: imbalance |
| E3 | Patient | + focal loss | Fix 1b: imbalance |
| E4 | Patient | + augmentation (noise, scaling, baseline wander, shift) | Fix 2: robustness to new patients |
| E5 | Patient | + RR-interval features | Fix 3: S beats are premature, so timing may matter more than shape |
| E6 | Patient | + autoencoder pretraining on DS1 | Stretch |
| E7 | Patient | LSTM or small transformer instead of CNN | Stretch |

**Metrics:** per-class recall, precision and F1; macro-F1; confusion matrix; per-patient breakdown; S-class results **with and without record 232** (it holds 75% of the test S beats).

## Week 1 (19–25 Oct): pipeline and baseline

| Day | M1: Data | M2: Models | M3: Evaluation & report |
|---|---|---|---|
| **Mon 19** | *All: kickoff, agree conventions above, set up repo + Drive* | | |
| Mon–Tue | Download MIT-BIH, segment beats, build both splits, save `.npz`, class statistics | Write CNN + training loop, test on random dummy data | Write `evaluate.py` on dummy predictions; outline intro + related work |
| **Wed 21** | *Checkpoint: data handed to M2* | | |
| Wed–Fri | Add augmentation + RR features to the pipeline | Run E0 and E1, tune on validation patients | Evaluate E0/E1: confusion matrices, per-patient plot; draft intro |
| **Sun 25** | *Checkpoint: E0 vs E1 confirms the gap. Freeze model + hyperparameters* | | |

## Week 2 (26 Oct – 1 Nov): experiments and writing

| Day | M1: Data | M2: Models | M3: Evaluation & report |
|---|---|---|---|
| Mon–Tue | Autoencoder pretraining (E6, stretch); start README | Run E2–E5 × 3 seeds | Results tables + figures as runs finish; record 232 analysis |
| **Wed 28** | *Checkpoint: results freeze, no new main experiments after this* | | |
| Wed–Thu | Write data + features sections; finish README | Write model + training sections; E7 if time | Failure analysis; write results + discussion |
| **Fri 30** | *M3 assembles full draft* | | |
| **Sat 31** | *All: read the whole report; one person reruns the README from a fresh Colab* | | |
| **Sun 1 Nov** | *All: final fixes, AI Use Statement, ZIP code + PDF* | | |

## Report outline (max 10 pages, Arial 10pt, single column)

| Section | Pages | Lead |
|---|---|---|
| Introduction | 1 | M3 |
| Review of existing techniques | 1.5 | All (one area each) |
| Data and methods | 2.5 | M1 (data) + M2 (models) |
| Experiments and results | 3 | M2 + M3 |
| Discussion and limitations | 1.5 | M3, input from all |
| Conclusion | 0.5 | M3 |
| AI Use Statement | – | All |

## Risks

- **PhysioNet errors:** use `pn_dir="mitdb/1.0.0"`, or download the dataset once to Drive.
- **Colab disconnects:** save checkpoints to Drive (runs are short anyway).
- **A fix doesn't help:** that's a valid result. Report it and explain why.
- **S results depend on one patient:** always report with and without record 232.
