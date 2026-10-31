# M3 handoff (evaluation and report)

## Update 31 Oct: full report draft

- **Read the full draft:** `docs/report/latex/main.pdf` (source `main.tex`; see
  `docs/report/README.md` for compiling locally or on Overleaf). The main body is
  7 of the 10 allowed pages; results are unchanged from 27 Oct.
- **M2, please check by Sun 1 Nov:** Section 2.1 (related work, your area) and
  Section 3.4 (model and training), which M3 wrote from your code, checkpoints and
  training histories; and fill in your part of the AI Use Statement (exact tool name
  and how you checked its output). Thanks for the histories: they are now Fig. 4.
- **Everyone:** put your full name on the cover and add the group number.
- **M1's PR #1 review, both open findings fixed:** the report and `src/predict.py`
  no longer claim the validation check proves identical data, and `src.predict`
  now runs `ensure_prepared` (raw checksums and cache audit) first; `summarise`
  fails on a prediction CSV that matches no test NPZ unless `--allow-unmatched`
  is given, and a `subject_id` column is checked too. 72 tests pass.
- **Seeds:** three seeds are kept (not more). The main conclusions are far
  outside seed noise, and the patient bootstrap, not the seed SD, dominates
  the uncertainty.
- The data figures now use the strict training split (record 209 holds 60% of
  training S beats, not 50%), and Fig. 4 (training curves) comes from M2's histories.

> M1 response, 28 Oct: the original reference manifest has been restored, both
> M1 methods contributions and setup defaults now use `patient_strict`, and the
> merged PR #1 has an AI-assisted M1 review in `M1_PR1_REVIEW.md` with execution
> evidence in `M1_HANDOFF_VALIDATION.json`. See `M1_HANDOFF.md` for M1's position
> on early stopping and E6. The original M3 requests below are retained for context.

Status 27 Oct: E0–E5 evaluated from M2's checkpoints (3 seeds each,
`patient_strict`). Results, failure analysis, discussion and conclusion are
drafted in `docs/report/M3_RESULTS.md` and `docs/report/M3_DISCUSSION.md`, and
the introduction no longer has placeholders. All 64 tests pass. Branch
`m3-evaluation` is based on M2's `train_result` (6fb469d).

## To do (please send your sections by Thu 29 Oct; M3 assembles the draft Fri 30)

**Both, today (results freeze):** decide on early stopping and E6 (see
"Decisions the team needs to make" below).

**M2:**
1. Push `results/E*/seed*_history.csv` and `results/training_summary.csv`.
2. Write the model and training-setup section (facts in
   `docs/report/M3_RESULTS.md` §4.1) and revise related work 2.1.
3. Add your AI-use notes to `docs/report/M1_AI_USE_DRAFT.md`.
4. Read the code review notes below (most urgent: note 1, `manifest.json`).
5. Confirm it is OK to merge your `train_result` commits into `main` through
   PR #1 (`m3-evaluation`).

**M1:**
1. Decide whether to restore your `manifest.json`
   (`git checkout a4d0f5c -- data/processed/manifest.json`; review note 1).
2. Update `M1_REPORT_INSERT.md` / `M1_DATA_METHODS.md`: the main patient
   split is now `patient_strict`, not `patient`.
3. If you have time, review PR #1 (the plan has each of us review a
   teammate's code).

## Headline results (details: `docs/report/tables/summary_27oct.md`)

| | Macro-F1 | S Se | S +P | V F1 | F F1 |
|---|---|---|---|---|---|
| E0 random split | 0.874 ± 0.009 | 0.618 | 0.910 | 0.955 | 0.814 |
| E1 patient split | 0.429 ± 0.002 | 0.000 | n/a | 0.756 | 0.000 |
| E2 + weighted CE | 0.371 ± 0.067 | 0.830 | 0.115 | 0.629 | 0.004 |
| E3 + focal | 0.444 ± 0.017 | 0.000 | n/a | 0.811 | 0.001 |
| E4 + augmentation | 0.433 ± 0.017 | 0.000 | n/a | 0.771 | 0.001 |
| E5 + RR features | 0.445 ± 0.030 | 0.001 | 0.167 | 0.799 | 0.027 |

1. The gap is large: 0.874 → 0.429 macro-F1. S and F F1 drop to zero.
2. No fix recovers S. E2's S recall comes from labelling whole patients S
   (98% of record 232's beats, 92% of record 212's N beats); within record 232
   its S score is at chance (AUROC 0.48).
3. E5 learned the timing cue: within record 232 it ranks S above N with
   AUROC 0.994 (E1: 0.43), but argmax never picks S, and a threshold chosen on
   validation patients does not transfer (post-hoc analyses, labelled as such).
4. The selected checkpoints of E1, E3, E4 and two E5 seeds detect ≤ 3% of S beats
   **on their own training set**. Early stopping chose epoch 1–5 in 14 of 15
   patient-split runs.

## How the numbers were produced

```bash
python -m src.predict --experiments all --seeds 42 43 44   # test + val CSVs, checks
python -m src.evaluate summarise --reference E1 --n-boot 2000
```

`src/predict.py` (new, M3) does what `python -m src.train --mode predict-test`
does, reusing M2's code, but runs on CPU. Instead of a byte-level hash check it
checks the data behaviourally: every checkpoint reproduced its stored validation
macro-F1 on our data within 3 × 10⁻⁴ (`results/export_check.csv`, copied to
`docs/report/tables/`). It also records each checkpoint's per-class recall on
the training and validation sets.

New in `src/evaluate.py`: the seed-file filter (fix, see below), within-record
S ranking (`ranking_S_<exp>.csv`) and the validation-threshold transfer
analysis (`threshold_transfer_S_<exp>.csv`). Both are reported in
`summary.md` under "Post-hoc".

## Decisions the team needs to make (before the Wed 28 Oct freeze)

1. **Early stopping (M2 + all).** Patient-split models stopped at epoch 1–5,
   before learning S on their own training data. Our conclusion holds anyway,
   because models that did learn S (E2, E5 seed 42) still miss it on new
   patients. But E3–E5 partly compare under-trained models. Options:
   (a) keep the results as they are and report the diagnosis as a finding and
   a limitation (drafted); or (b) M2 also runs E1 and E5 with a fixed number of
   epochs (e.g. 30, last epoch, no early stopping) as a **sensitivity analysis**,
   stated in the report as added after the first test evaluation. M3
   recommends (a), plus (b) if M2 has time. Do not replace the main results.
2. **E6 (stretch).** M1's encoders are ready; no supervised E6 run exists. H6
   stays "not tested" unless M2 runs it by Wed.
3. **Saliency (stretch, M3).** The ranking analysis already answers the
   interpretability question for the report. M3 suggests dropping Section 2.4
   to save space unless someone wants the integrated-gradients figure.

## Requests to M2

- Push `results/E*/seed*_history.csv` and `results/training_summary.csv`
  (results/ is git-ignored; use `git add -f` or the Drive folder). They are
  needed for a training-curve figure and to support the early-stopping
  paragraph.
- Write Section 4.1 (training setup) and the model part of Section 3. The
  facts are in `docs/report/M3_RESULTS.md` §4.1: 13,508 parameters, AdamW
  1e-3, weight decay 1e-4, batch 256, ≤ 30 epochs, patience 4, and the E2
  class weights. Note that the script defaults are 25 epochs and patience 6,
  so state the values actually used, and put the exact command in the README.
- Add your AI-use notes to `docs/report/M1_AI_USE_DRAFT.md`.

## Code review notes for M2 (week 2)

1. **`data/processed/manifest.json` was overwritten with locally rebuilt
   hashes (af46bb6).** With it, `python -m src.data audit` and the M3
   notebook's `ensure_prepared` fail for anyone whose NPZs match M1's original
   build ("File changed: random_train.npz"). Counts, metadata and validation
   scores agree, so the arrays are probably the same and only the compressed
   bytes differ (possibly a different zlib in Python 3.14 on Windows); as M1
   notes, that is not proof. **Resolved 28 Oct: M1 restored the manifest.** Suggest
   restoring M1's manifest (`git checkout a4d0f5c -- data/processed/manifest.json`)
   and not committing locally regenerated metadata (M1 handoff note 2).
2. **The checkpoint hash check is machine-specific.** `export_test` requires the
   training NPZ's file hash to equal the one stored in the checkpoint, so
   `--mode predict-test` fails on any machine whose zlib differs from yours,
   including possibly the marker's. `src.predict` uses the behavioural check
   instead; consider using it or loosening the check.
3. **CUDA-only.** `resolve_device` rejects CPU, so CPU-only machines cannot
   reproduce the results. Inference takes about 2 minutes for all 18 models on CPU.
4. **Test predictions were never exported.** `--mode predict-test` had not been
   run. M3 exported them with `src.predict`. If you also run predict-test on the
   GPU, the CSVs should agree except for a few borderline beats.
5. **`checkpoints/` is in `.gitignore`** but the 18 checkpoints were
   force-added. That is fine (1 MB in total and needed for reproduction), but
   update the README so it does not say checkpoints are unversioned.
6. **Validation macro-F1 counts F (17 beats) and S (175 beats from two
   records).** With so few validation S/F beats, the score mostly tracks N and
   V, which is one reason early stopping fired early.
7. `src/test_models.py` lives in `src/`; the other tests are in `tests/`.
   Moving it would let `python -m unittest discover -s tests` run everything.

## Fixed in M3's own code

- `load_experiment` read every `seed*.csv`, so M2's `seed42_val.csv` and
  `seed42_history.csv` would have been treated as extra seeds. It now reads only
  `seed<k>.csv`, and there is a test for this.

## Data findings (details in `docs/report/M3_DATA_FINDINGS.md`)

- Record 232 holds 75.2% of test S beats; record 213 holds 93.3% of test F
  beats; record 208 holds 94% of training F beats.
- Always predicting N gives 89.0% accuracy and macro-F1 0.236 on the test set.
- Under `patient_strict`, 62% of training S beats (395/640) occur in SVTA runs,
  mostly in record 209, where they are not early; 48.9% of training S beats
  have RR ratio < 0.9, against 87.7% of test S beats.
- In record 232, S beats correlate at r = 0.975 with that record's N beats in
  shape; their timing is what differs.
