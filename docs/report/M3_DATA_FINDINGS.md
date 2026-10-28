# Dataset findings (for Section 3 Data and Section 4/5)

> M3, 20 Oct. These are **properties of the labelled data, not model results**.
> Every number is reproduced by `python -m src.evaluate describe-data`
> (outputs in `results/data_analysis/`: `data_facts.json`,
> `class_counts_by_record_*.csv`, `rr_timing_by_record_*.csv`,
> `s_beat_rhythm_context.csv`, `figures/`). Figures are copied to
> `docs/report/figures/`. M1 owns the Data section; use what fits.
>
> **Leakage warning.** Findings about the test set (DS2) may motivate
> *hypotheses* and *analysis*, but must not be used to choose or tune any method
> (augmentation ranges, class weights, thresholds, RR features). Any design choice
> must be justified from training/validation records only.

## F1. Class counts after preprocessing

| Partition | Records | Beats | N | S | V | F |
|---|---|---|---|---|---|---|
| Train (`patient`: DS1 minus 4 validation records) | 18 | 41,355 | 37,352 | 768 | 2,838 | 397 |
| Train (`patient_strict`: also without 201) | 17 | 39,394 | 35,719 | 640 | 2,640 | 395 |
| Validation (108, 118, 203, 223) | 4 | 9,614 | 8,472 | 175 | 950 | 17 |
| Test (DS2) | 22 | 49,661 | 44,218 | 1,836 | 3,219 | 388 |
| Random split: train / val / test | 44 | 64,404 / 16,101 / 20,125 | | | | |

S beats are 1.9% of the training beats and 3.7% of the test beats. The
validation set has only 17 F beats, so validation F scores (and a validation
macro-F1 used for early stopping) will be noisy.

## F2. Abnormal beats are concentrated in a few records

| Class | Training: largest record | Test: largest record | Records containing the class (train / test) |
|---|---|---|---|
| S | 209 holds 383 of 768 (49.9%) | **232 holds 1,381 of 1,836 (75.2%)** | 12 / 16 |
| V | 208 holds 992 of 2,838 (35.0%) | 233 holds 830 of 3,219 (25.8%) | 13 / 16 |
| F | **208 holds 372 of 397 (93.7%)** | **213 holds 362 of 388 (93.3%)** | 7 / 7 |

F is effectively a one-patient-to-one-patient transfer problem: nearly all
training F beats come from record 208 and nearly all test F beats from record
213. Of the 22 test records, 6 contain no S beats and 15 contain no F beats.
Under the random split the same records still dominate (232 holds 48.6% of
random-test S beats), but they appear on both sides of the split.

**Figure `class_concentration`.** *Cumulative share of each abnormal class held
by the k largest contributing records, training (left) and test (right).
A single test record holds 75% of S and 93% of F beats.*

## F3. Accuracy is uninformative here

Predicting N for every beat scores 89.0% accuracy on the patient test set
(89.5% on the random test set) with macro-F1 0.236 (N F1 0.942; S, V and F F1 = 0).
Report macro-F1 and per-class Se/+P/F1; mention accuracy only next to this baseline.

## F4. S-beat timing differs between training and test patients

We use the RR ratio = previous RR / mean of the last 10 RR intervals (M1's
feature 4); a ratio below 0.9 marks a beat as early relative to the local
rhythm.

| Partition | S beats with ratio < 0.9 | Median S ratio | N beats with ratio < 0.9 |
|---|---|---|---|
| Train (`patient`) | 57.3% | 0.84 | 1.9% |
| Train (`patient_strict`) | 48.9% | 0.91 | 1.1% |
| Validation | 89.7% | 0.79 | 10.3% |
| Test (DS2) | 87.7% | 0.73 | 7.0% |
| Random split: train / test | 79.1% / 79.5% | 0.76 / 0.76 | 5.2% / 5.3% |

Under the random split the training and test distributions match. Under the
patient split they do not. The expert rhythm annotations explain why
(`s_beat_rhythm_context.csv`): **51.8% of training S beats occur during
supraventricular tachyarrhythmia (SVTA) runs** (mainly records 209 and 207, with
median ratios 0.97 and 0.99). Inside a run the local mean RR is itself short, so
these S beats are not early relative to it. In the test set, **75.2% of S
beats occur during sinus bradycardia** (all in record 232) and only 3.5% during
SVTA. Removing record 201 for the strict protocol removes 128 of the most
premature training S beats (median ratio 0.51), lowering the premature share
from 57.3% to 48.9%.

**Figure `rr_prematurity_S`.** *RR ratio of S beats per record (box: IQR,
white line: median, whiskers: 10th–90th percentile) for records with ≥ 20 S
beats; the shaded band is the 10th–90th percentile of N beats. Many training
S beats (records 207, 124 and half of 209) are not early relative to the local
rhythm, whereas most test S beats are (88%; record 234 is the exception).*

## F5. Record 232: early timing, N-like shape, unusual rhythm

- **Shape.** The mean S template of record 232 correlates at r = 0.975 with
  record 232's own N template, but only r = 0.762 with the pooled training S
  template and r = 0.795 with the pooled training N template (standardised MLII).
  Its R-peak amplitude is lower than in training (mean template at the R peak:
  1.82 for 232 S and 1.91 for 232 N, against 2.18 for training S and 2.91 for training N, in
  training-standardised units).
- **Timing.** Its S beats are early relative to the local rhythm (median ratio
  0.74), but their absolute previous RR (median 0.73 s) is typical of N beats in
  other records (for example, record 100's N beats have a median of 0.80 s). Its N beats follow long pauses
  (median previous RR 1.87 s, ratio 1.76).
- **Implication (hypothesis, not a result).** A waveform-only model has little
  shape information to separate S from N in this record. The relative RR
  feature carries the useful signal, the absolute RR does not, and training S
  beats mostly do not show the relative cue (F4).

**Figure `templates_232`.** *Median beat (line) and interquartile band of N
and S beats: pooled training records (left) and test record 232 (right). In
record 232, S and N beats are almost indistinguishable in shape.*

## F6. Subject overlap in the conventional split

Records 201 (DS1) and 202 (DS2) come from the same subject [MITDB]. Test record
202 contains 55 S, 19 V and 1 F beats. Under the conventional `patient`
protocol this patient is not unseen; `patient_strict` removes 201 from
training. Either way, report record 202 separately in per-patient results.
