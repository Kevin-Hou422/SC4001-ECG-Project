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

Use the README data contract. E1-E5 use the main `patient_strict` prefix, matching
M2's completed runs; `patient` remains a conventional reference. Use the same raw data split for all
three training seeds. Fit any class weights from that prefix's training `y` only.
Call `dataset.set_epoch(epoch)` before each augmented training epoch.

The NPZ `rr` is already standardised. All four columns, including the ratio,
are standardised; units/formulae are available in `rr_raw` and `rr_names`.
Do not pass RR to E0-E4 models unless changing and documenting the experiment.
M2's supervised classifier and classification losses are now in `src/models.py`
and `src/train.py`. M1's independent E6 autoencoder and its transfer contract are
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
M2's `ECGCNN` now imports M1's `ECGEncoder`; the scratch/pretrained architecture
contract is compatible. E0-E5 checkpoints and M3's evaluation are merged through
PR #1. No supervised E6 run exists in that handoff, so H6 remains not tested.
A human teammate's code review and an actual Google-hosted Colab run are not
represented as completed merely by providing this setup.

The notebook can be validated cell-by-cell locally for portable code, but a
Google runtime session and optional Drive consent require the user's account.
See `validation.json` for the checks actually run on this repository snapshot.

## Remaining human/integration steps

- Run the updated notebook in a signed-in Google Colab runtime and save outputs.
  On 23 October browser automation was stopped because its safety check could
  not establish the Chrome URL. The hosted run is not falsely marked completed.
- Keep the main E1-E5 protocol `patient_strict` consistent throughout the report.
- Integrate and shorten M1's report drafts; add the team's actual training results.
- A teammate should review M1's code. M1's reciprocal AI-assisted review of the
  merged PR #1 is recorded in `M1_PR1_REVIEW.md`; a member must read and own the
  findings before recording human review. `M1_REVIEW.md` is the earlier self-review.

## Response to M3's 27 October handoff (completed 28 October)

1. Restored the original `a4d0f5c` manifest. The merged replacement reproduces
   `File changed: random_train.npz` on M1's original arrays; the restored manifest
   audits those arrays successfully. Fresh builds retain their own local metadata.
   See the README's safe new-directory procedure for existing incompatible caches.
2. Updated both report-method contributions and their generators to state that
   E1-E5 use `patient_strict`; aligned both M1 notebook defaults and README examples.
3. Reviewed PR #1 at merge `f24e181`, including model/encoder compatibility,
   prediction provenance, test-row alignment and metric/bootstrap code. Findings
   and executed evidence are in `M1_PR1_REVIEW.md` and `M1_HANDOFF_VALIDATION.json`.

M1's recommendation is to retain the frozen E0-E5 results with the early-stopping
limitation already drafted by M3. Any fixed-epoch follow-up belongs in a labelled
post-hoc sensitivity analysis; it must not replace the main results. This records
M1's position, not a decision on behalf of the whole team. E6 reconstruction
diagnostics cannot support H6 without a supervised scratch/pretrained comparison.
