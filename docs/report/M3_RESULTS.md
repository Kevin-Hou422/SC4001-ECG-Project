# 4 Experiments and results (M3 part) and failure analysis

> Draft v1 (M3, 27 Oct). Numbers come from `python -m src.predict` and
> `python -m src.evaluate summarise --reference E1` on M2's checkpoints of
> 27 Oct (E0–E5, seeds 42–44, `patient_strict`). Frozen copies:
> `tables/summary_27oct.md` and `tables/export_check.csv`. M2 writes the
> training setup (4.1); this file covers 4.2 onwards. Hypotheses H1–H6 were
> fixed on 20/25 Oct in `M3_RESULTS_PLAN.md`, before any training run.
> Analyses marked *post-hoc* were added after the first test evaluation.

## 4.1 Setup (M2 to expand)

All patient-split runs use `patient_strict`: training on 17 DS1 records, early
stopping on 4 held-out DS1 records (108, 118, 203, 223), testing once on the 22
DS2 records (49,661 beats). E0 uses the random beat split (20,125 test beats).
The model is a 13,508-parameter 1D CNN (M1's convolutional encoder, global
average pooling, a linear head; E5 concatenates the four standardised RR
features before the head). AdamW, learning rate 1e-3, weight decay 1e-4, batch
256, at most 30 epochs; the checkpoint with the best validation macro-F1 is
kept, with patience 4. E2 weights the loss by inverse training frequency
(N 0.025, S 1.389, V 0.337, F 2.250); E3 uses focal loss with γ = 2. Each
configuration is trained with seeds 42, 43 and 44; we report mean ± SD.

## 4.2 The generalisation gap (H1: supported)

**Table 2.** Main results on the full test set (mean ± SD over three seeds).

| Exp | Split | Change | Macro-F1 | S Se | S +P | S F1 | V F1 | F F1 |
|---|---|---|---|---|---|---|---|---|
| — | patient | always predict N | 0.236 | 0 | n/a | 0 | 0 | 0 |
| E0 | random | baseline | 0.874 ± 0.009 | 0.618 ± 0.027 | 0.910 ± 0.017 | 0.736 ± 0.023 | 0.955 ± 0.005 | 0.814 ± 0.010 |
| E1 | patient | baseline | 0.429 ± 0.002 | 0.000 ± 0.000 | n/a | 0.000 ± 0.000 | 0.756 ± 0.008 | 0.000 ± 0.000 |
| E2 | patient | + class-weighted loss | 0.371 ± 0.067 | 0.830 ± 0.060 | 0.115 ± 0.013 | 0.201 ± 0.018 | 0.629 ± 0.102 | 0.004 ± 0.002 |
| E3 | patient | + focal loss | 0.444 ± 0.017 | 0.000 ± 0.000 | n/a | 0.000 ± 0.000 | 0.811 ± 0.067 | 0.001 ± 0.002 |
| E4 | patient | + augmentation | 0.433 ± 0.017 | 0.000 ± 0.000 | n/a | 0.000 ± 0.000 | 0.771 ± 0.061 | 0.001 ± 0.003 |
| E5 | patient | + RR features | 0.445 ± 0.030 | 0.001 ± 0.001 | 0.167* | 0.001 ± 0.002 | 0.799 ± 0.092 | 0.027 ± 0.047 |
| E6 | patient | + autoencoder pretraining | **[TBD: not yet trained]** | | | | | |

Se = sensitivity (recall), +P = positive predictivity (precision). n/a: the
class was never predicted. \*Only one seed predicted any S beat.

The same network that reaches macro-F1 0.874 on a random beat split reaches
0.429 on unseen patients (Table 2, Fig. 2). Accuracy hides this: it falls only
from 0.980 to 0.923, just 3.3 points above the 0.890 of always predicting N.
As H1 predicted, the loss is concentrated in the rare classes. S F1 falls from
0.736 to 0 and F F1 from 0.814 to 0, while V F1 falls less, from 0.955 to 0.756.
On unseen patients the baseline never predicts S: 98.6% of the 1,836 test S
beats are called N. The F beats of record 213 are split between N (50%) and V
(50%). The two splits also differ in which patients are tested, so the gap is
not caused by leakage alone. However, the random split's scores clearly do not
carry over to new patients.

## 4.3 Do the fixes help? (H2–H5)

**Table 3.** Change from E1 with 95% patient-bootstrap intervals (2,000
resamples of the 22 test records, the same draws for every model, seed-averaged).

| Exp | Δ Macro-F1 | Δ S F1 | Δ V F1 | Δ F F1 | S F1 without 232 | F F1 without 213 |
|---|---|---|---|---|---|---|
| E2 | −0.058 [−0.135, +0.042] | +0.201 [+0.008, +0.473] | −0.127 [−0.227, +0.016] | +0.004 [0.000, +0.017] | 0.027 ± 0.009 | 0.001 ± 0.001 |
| E3 | +0.015 [−0.001, +0.034] | 0 | +0.056 [−0.006, +0.126] | +0.001 [0.000, +0.020] | 0 | 0.002 ± 0.003 |
| E4 | +0.004 [−0.003, +0.016] | 0 | +0.015 [−0.014, +0.049] | +0.001 [0.000, +0.051] | 0 | 0.008 ± 0.013 |
| E5 | +0.016 [+0.004, +0.053] | +0.001 [0.000, +0.012] | +0.043 [+0.012, +0.079] | +0.027 [0.000, +0.172] | 0.006 ± 0.010 | 0.001 ± 0.001 |

- **H2 (weighted and focal loss raise rare-class recall at the cost of +P): partly
  supported.** Class weighting (E2) raises S sensitivity from 0 to 0.83, but
  S +P is 0.115, so about eight of every nine S predictions are false. Macro-F1
  falls by 0.058 (interval includes 0), and V F1 and accuracy (0.530 ± 0.155)
  fall with it. Focal loss (E3) does not change S or F at all; its small gains
  are in V. Without record 232 the E2 gain in S almost disappears (S F1 0.027).
  Section 4.4 shows that even the gain in record 232 is not beat detection.
- **H3 (augmentation gives at most a small gain): supported.** E4 differs from
  E1 by +0.004 macro-F1 [−0.003, +0.016] and detects no S beats.
- **H4 (RR features give the largest S gain): not supported for decisions.**
  E5 predicts S for almost no test beat (S Se 0.001). Its macro-F1 gain over E1,
  +0.016 [+0.004, +0.053], is the only macro-F1 interval that excludes zero, but
  it comes from V (+0.043) and not from S. Section 4.4 shows that the timing information
  is learned but never changes a decision.
- **H5 (no robust F conclusion): supported.** F F1 is at most 0.03 for every
  patient-split model; the largest, E5's 0.027, has an interval from 0 to 0.172
  and comes from one seed. Training F beats come almost entirely from record 208
  (94%) and test F beats from record 213 (93%).
- **H6 (pretraining gives little S gain): [TBD: E6 classifier runs.]**

The seed spread is large for the patient-split models (for example, V F1
± 0.09 for E5) and the bootstrap intervals are wide. Which patients are tested
matters as much as which fix is used. Fig. 4 shows S recall per test record.

## 4.4 Failure analysis

**Where are the errors?** For E1, the largest error cells are record 232
(1,379 S beats called N, 99.9% of its S beats), record 222 (all 209 S beats
called N), record 213 (363 N beats called V, F beats split between N and V), and
record 233 (202 V beats called N, 24%). Two records account for 86% of all S
errors.

**Did the models learn S on the training patients?** *(Diagnostic on training
and validation data only.)* The selected checkpoints of E1, E3, E4 and two of
three E5 seeds detect at most 3% of the S beats in their own *training* set.
Early stopping chose epochs 1–5 for 14 of the 15 patient-split runs (epoch 9
for the other), against 18–29 for E0, because validation macro-F1 stopped
improving early. The models trained long
enough to fit S still fail on new patients. E2 detects 85–90% of training S
beats and E5 (seed 42, epoch 9) 61%, but these models find at most 8% of the
validation patients' S beats. Two failures therefore stack: model selection
stops before S is learned, and S learned on training patients does not transfer.

**E2 labels patients, not beats.** *(Post-hoc.)* Class weighting makes E2 call
98% of all beats in record 232 S, including 99% of that record's 397 N beats.
Because 78% of record 232's beats are S, this produces most of the S "gain".
The same behaviour produces the false positives: 92% of record 212's N beats
(2,530 of 2,746) are called S, as are 80% of record 222's and 68% of record
100's. Within record 232, E2's S probability separates S from N beats no better
than chance (AUROC 0.48 ± 0.06).

**E5 ranks S beats correctly but never predicts them.** *(Post-hoc.)* We
measured how well each model's S probability ranks S beats above other beats,
across all test beats pooled and within each record (Table 4). Pooled AUROC
looks moderate for every patient-split model (0.75–0.86). Within records, E1–E4
are at or below chance (median 0.46–0.48), so their pooled score reflects
differences between patients, not between beats. E5 is the exception: within
record 232 its S probability separates S from N almost perfectly (AUROC 0.994),
and the median over records is 0.895. Yet the argmax never selects S.
Patients differ in heart rate, rhythm and beat shape, so a fixed decision
boundary falls in a different place relative to each patient's beats.

**Table 4.** *(Post-hoc.)* AUROC of p_S, S vs all other beats, mean over seeds.
"Within records": median over the 8 test records with ≥ 10 S and ≥ 10 other beats.

| Exp | Pooled | Within records (median) | Within record 232 | Share predicted S in 232 |
|---|---|---|---|---|
| E0 (random split) | 0.963 | 0.943 | 0.948 | 0.775 |
| E1 | 0.753 | 0.467 | 0.429 | 0.000 |
| E2 | 0.859 | 0.478 | 0.482 | 0.983 |
| E3 | 0.863 | 0.475 | 0.461 | 0.000 |
| E4 | 0.762 | 0.457 | 0.419 | 0.000 |
| E5 | 0.863 | 0.895 | 0.994 | 0.000 |

**Would a better threshold fix it?** *(Post-hoc; the threshold is chosen on
validation patients only.)* We predicted S whenever p_S exceeded a threshold that
maximised S F1 on the four validation records. For no model does this transfer.
For E1, E2 and E4 the chosen threshold labels most test beats S
(S Se 0.94–1.00, +P 0.04–0.07). E3 reaches S F1 0.21 ± 0.11, but its
within-record ranking is at chance, so this again labels patients rather than
beats. For E5 the result depends on the seed (S Se 0.62 ± 0.45, +P
0.11 ± 0.05, S F1 0.15 ± 0.08), and macro-F1 is no better than with the argmax
(0.420 vs 0.445). The validation patients' S
beats (records 118 and 223) do not place the threshold where the test patients
need it. This fits the within-record result: the scores are informative
within a patient but not comparable between patients.

**Why does timing not transfer?** In the strict training set, 62% of S beats
(395 of 640) occur inside supraventricular tachycardia runs, mostly record 209,
where they are not early relative to the preceding beats. Only 48.9% of training
S beats have an RR ratio below 0.9, against 87.7% of test S beats
(`M3_DATA_FINDINGS.md`, F4). The timing feature that defines most test S beats
is therefore a weak cue in training, so the model has little reason to move its
decision boundary towards it.

## Figures for this section

| Fig. | File (`figures/`) | Caption draft |
|---|---|---|
| 2 | `confusion_E0` + `confusion_E1` (side by side) | Confusion matrices of the same CNN on a random beat split (E0) and on unseen patients (E1), mean over three seeds. Row percentages; counts below. |
| 3 | `bootstrap_vs_reference` | Change from the baseline E1 for each fix, with 95% patient-bootstrap intervals (test records resampled; seed-averaged). |
| 4 | `per_patient_S` | S recall per test record (mean over seeds); records sorted by number of S beats. E2's recall in record 232 comes with S predictions for almost every beat of that record (Section 4.4). |
| (alt.) | `comparison_all` | Macro-F1 and per-class F1 for every experiment, mean ± SD over seeds. Use instead of Fig. 3 if space is short. |
