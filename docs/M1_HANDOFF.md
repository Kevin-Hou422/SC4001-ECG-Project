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
- Safe first-run generation from a Git clone containing metadata but no NPZs.
- M1 methods, related-work and AI-use report contributions in `docs/report/`.
- Linux CI workflow for protocol tests, real-data preparation and portable cells.
- E6 autoencoder implementation and six ten-epoch encoder exports, with training
  histories, resume support, audit and a dedicated Colab notebook.
- Concise report integration text and an explicit implementation review record.

## For M2

Use the README data contract. For a given comparison choose a consistent
`patient` or `patient_strict` prefix for E1-E5. Use the same raw data split for all
three training seeds. Fit any class weights from that prefix's training `y` only.
Call `dataset.set_epoch(epoch)` before each augmented training epoch.

The NPZ `rr` is already standardised. All four columns, including the ratio,
are standardised; units/formulae are available in `rr_raw` and `rr_names`.
Do not pass RR to E0-E4 models unless changing and documenting the experiment.
No supervised classifier or classification loss implementation is supplied here;
that is M2's work. M1's independent E6 autoencoder and its transfer contract are
documented in `E6_PRETRAINING.md`. E1/E6 must use the same downstream architecture.

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

## E6 and external steps

E6 pretraining is implemented and executed for both patient protocols, using
seeds 42/43/44 and ten fixed epochs. The six encoder exports and diagnostics are
in `artifacts/pretraining/` and `docs/report/M1_E6_RESULTS.md`.
No validation/test waveforms or class targets enter pretraining optimisation.
M2 has not yet supplied its baseline architecture. It must adopt the documented
encoder for both scratch/pretrained conditions, or align pretraining with its
chosen encoder before running a controlled E1/E6 comparison.
M2/M3 experiments, a human teammate's code review, and an actual Google-hosted
Colab run are not represented as completed merely by providing this setup.

The notebook can be validated cell-by-cell locally for portable code, but a
Google runtime session and optional Drive consent require the user's account.
See `validation.json` for the checks actually run on this repository snapshot.

## Remaining human/integration steps

- Run the updated notebook in a signed-in Google Colab runtime and save outputs.
  On 23 October browser automation was stopped because its safety check could
  not establish the Chrome URL. The hosted run is not falsely marked completed.
- M2/M3 should use one agreed patient protocol consistently. Both remain available;
  only `patient_strict` supports the strict subject-disjoint claim.
- Integrate and shorten M1's report drafts; add the team's actual training results.
- A teammate should review M1's code; M1 should review a teammate's implementation
  when available, as required by the group plan. This review is not yet recorded.
  `M1_REVIEW.md` is an AI-assisted review and does not impersonate a teammate.
