# Results, failure analysis and discussion: plan

> M3, 20 Oct. This file fixes the evaluation **before** any experiment has run,
> so the report can honestly say the hypotheses and analyses were pre-specified
> (the Git history dates them). Fill the tables with
> `python -m src.evaluate summarise`. Do not change a hypothesis after seeing
> results; report it as supported or not.

## Pre-specified hypotheses (written 20 Oct 2026, before any training run)

| ID | Hypothesis | Basis | Check |
|---|---|---|---|
| H1 | E0 (random split) scores far higher than E1 (patient split), mostly through S and F; V drops less. | Literature [Sellami2019, deChazal2004]; F and S concentration (findings F2) | E0 vs E1 per-class F1; not paired (different test sets) |
| H2 | Weighted and focal losses (E2, E3) raise S and F recall but lower their +P; the macro-F1 change is small and uncertain. | Rare classes get more weight; N beats from new patients become false positives | Paired patient-bootstrap Δ vs E1; Se and +P separately |
| H3 | Augmentation (E4) gives at most a small gain. | It simulates gain, noise and baseline changes, but S beats in record 232 differ from N in timing, not shape (findings F5) | Δ vs E1; per-record S recall |
| H4 | RR features (E5) give the largest S gain, concentrated in records with isolated early S beats (232, 222, 202, 100). The gain is limited because about half of training S beats come from tachycardia runs where they are not early (findings F4). | Findings F4, F5 | Δ S F1 vs E1, with and without 232; `timing_breakdown_E5.csv`: recall on early (< 0.9) vs on-time S beats |
| H5 | No F conclusion is robust: F performance mostly measures transfer from record 208 to record 213. | Findings F2 | Patient-bootstrap CI for F F1; F with and without 213 |
| H6 *(added 25 Oct, after M1's E6 pretraining but before any supervised E6 run)* | Autoencoder pretraining (E6) gives little or no S gain over the identical scratch encoder. | It learns waveform shape from training beats; S beats in record 232 are distinguished by timing, not shape (findings F5) | Δ vs the scratch encoder with the same architecture, per-record S recall |

## Tables (report)

**Table 1. Data** (Section 3): partition × class counts (findings F1).

**Table 2. Main results**, complete test set, mean ± SD over 3 seeds:

| Exp | Split | Change | Macro-F1 | S Se | S +P | S F1 | V F1 | F F1 |
|---|---|---|---|---|---|---|---|---|
| — | patient | always predict N | 0.236 | 0 | n/a | 0 | 0 | 0 |
| E0 | random | baseline | | | | | | |
| E1 | patient | baseline | | | | | | |
| E2 | patient | + class-weighted loss | | | | | | |
| E3 | patient | + focal loss | | | | | | |
| E4 | patient | + augmentation | | | | | | |
| E5 | patient | + RR features | | | | | | |
| E6 | patient | + autoencoder pretraining (vs identical scratch encoder) | | | | | | |

**Table 3. Robustness of the fixes**: Δ macro-F1 / Δ S F1 / Δ F F1 vs E1 with
95% patient-bootstrap intervals, and S F1 without record 232 and F F1 without
record 213. The source is `summary.md` (bootstrap and sensitivity sections).

## Figures (report; at most 4–5 in 10 pages)

1. `class_concentration` (data, ready): why per-patient analysis is needed.
2. `confusion_E0` next to `confusion_E1`: the generalisation gap.
3. `comparison_all` or `bootstrap_vs_reference`: fixes vs E1 with intervals.
4. `per_patient_S`: which test patients each fix helps.
5. Either `rr_prematurity_S` or `templates_232` (data, ready), or the saliency figure.

## Failure analysis procedure (Section 5)

1. **Where are the errors?** `top_confusions_<exp>.csv`: the largest
   (record, true, predicted) cells. Expected: 232 S→N, 213 F→N/V.
2. **Which patients?** `per_patient_<exp>.csv` / `per_patient_S|V|F` heatmaps.
   Name the records each fix helps or hurts; check whether a pooled gain comes
   from a single record.
3. **Is timing the reason?** `timing_breakdown_<exp>.csv`: S recall for early
   (ratio < 0.8, 0.8–0.9) vs on-time (0.9–1.1) beats, for record 232 and for
   the other records. If E1 misses early S beats in 232 and E5 recovers them,
   H4 is supported by mechanism, not only by score.
4. **New false positives.** `N_false_pos`/`S_false_pos` per record. Which
   patients' N beats become false S after reweighting?
5. **Record 202** (same subject as training record 201 under `patient`):
   compare its S recall with that of other records, and between the two
   protocols if both are run.
6. **Saliency (stretch).** `src/saliency.py` integrated gradients on correctly
   and incorrectly classified S beats from record 232 vs training S beats.

## Discussion and limitations (Section 5, outline)

- **What the gap means.** E0 vs E1 changes both leakage and the patient mix,
  so the drop is not attributable to leakage alone (as M1 notes).
- **Concentration.** With 22 test patients, "S performance" is mostly
  record-232 performance. Interpret claims through the per-record results and
  bootstrap intervals.
- **Data limits.** 47 subjects; one lead (MLII); annotated beat locations are
  used instead of a beat detector; the next-RR feature makes this offline
  classification; paced records and the Q class are excluded; there is no
  external test database.
- **Statistical limits.** 3 seeds; a 22-patient bootstrap is coarse; the
  validation set has 17 F beats; no correction for multiple comparisons across
  fixes (report intervals, not p-values).
- **No test tuning.** DS2 was used only for final evaluation (state how the
  number of epochs and other hyperparameters were chosen on validation records).

## Conclusion (Section 6, outline)

The size of the gap; which fix helped which class, and whether it holds without
the dominant records; the main lesson that S and F results on DS2 are
patient-specific; one sentence on future work (more subjects, external test set,
patient-level reporting as the default).
