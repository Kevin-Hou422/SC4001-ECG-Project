# M1 handoff

## Scope implemented

- Public raw data mirrored into the repository (44 non-paced MIT-BIH records).
- SHA256-verified download and offline verification.
- MLII extraction, four-class mapping, 256-sample segmentation and exclusion audit.
- Random-beat, conventional DS1/DS2, and strict subject-independent variants.
- Separate validation records; train-only waveform and RR standardisation.
- Four offline RR features and four on-the-fly training augmentations.
- NPZ loader, PyTorch-compatible dictionary dataset and Colab setup notebook.
- Reproduction instructions, provenance metadata and protocol tests.

## For M2

Use the README data contract. For a given comparison choose a consistent
`patient` or `patient_strict` prefix for E1-E5. Use the same raw data split for all
three training seeds. Fit any class weights from that prefix's training `y` only.
Call `dataset.set_epoch(epoch)` before each augmented training epoch.

The NPZ `rr` is already standardised. All four columns, including the ratio,
are standardised; units/formulae are available in `rr_raw` and `rr_names`.
Do not pass RR to E0-E4 models unless changing and documenting the experiment.
No model or loss implementation is supplied here; that is M2's work.

## For M3

- N=0, S=1, V=2, F=3. `pid` is a record ID. Use `subject_id` if grouping people.
- Baseline prediction CSV: `pid,y_true,y_pred,p_N,p_S,p_V,p_F` in test NPZ row
  order. Optionally add `sample` for unambiguous traceability; extra columns
  should not break the agreed evaluation reader.
- Preserve the complete test set as the main evaluation. `pid != 232` is a
  supplementary sensitivity analysis; do not remove the patient to improve scores.
- Some patients lack a class; report support and explicitly handle undefined
  per-patient metrics. Do not silently turn absent S into zero performance.
- Record 232 concentration is a dataset property. Whether methods improve other
  patients is an experimental question. No performance or generalisation gap
  has been measured by M1 setup.

## Deliberate clarifications to the preliminary plan

1. Classical DS1/DS2 is preserved, with its 201/202 subject overlap disclosed.
   `patient_strict` removes 201 from training while retaining the DS2 test set.
2. Random 80/20 denotes development/test. Early stopping uses 20% of development,
   giving approximately 64/16/20 train/val/test; no test tuning.
3. Actual counts exclude incomplete windows and missing pre/post context, so
   use generated counts rather than the unprocessed counts in early drafts.
4. Official beat locations supply segmentation and next-RR features. This is
   offline annotated-beat classification, not an end-to-end real-time ECG system.
5. No 99% result or patient-generalisation drop is assumed or promised.

## Stretch and external steps

Autoencoder pretraining (E6) is deferred as explicitly optional in the plan.
It must use the actual training subset only, excluding validation and DS2.
M2/M3 experiments, a human teammate's code review, and an actual Google-hosted
Colab run are not represented as completed merely by providing this setup.

The notebook can be validated cell-by-cell locally for portable code, but a
Google runtime session and optional Drive consent require the user's account.
See `validation.json` for the checks actually run on this repository snapshot.
