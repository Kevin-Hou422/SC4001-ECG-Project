# M1 review of PR #1 and response to M3

Reviewed 28 October 2026: [PR #1](https://github.com/Kevin-Hou422/SC4001-ECG-Project/pull/1),
already merged as `f24e1814d542500468f630d30608adaf10867c73`.
Scope: M2 model/training interfaces, M3 prediction/evaluation code, preprocessing
provenance and the M1 actions in `M3_HANDOFF.md`. This is a Codex-assisted review
prepared for M1's member to inspect and own; it does not certify human peer review.

## Findings and disposition

1. **[P1, fixed] Replacing the reference manifest breaks existing M1 caches.**
   The merged manifest rejects the original `random_train.npz` in `src.data.audit`.
   All nine archive hashes and the recorded Python/NumPy versions differ, while
   configuration, source/pipeline hashes, normalisation and cohort metadata agree.
   Restored the complete original manifest from `a4d0f5c`, rather than selectively
   editing checksums. The original nine archives now pass audit. M2's original
   arrays are unavailable locally, so metadata agreement is not called proof of
   tensor identity or of a particular compression-library cause. The README gives
   a safe new-directory rebuild procedure for users with a different local cache.
   Two regression tests cover rebuilding metadata-only clones with foreign hashes
   and rejecting repacked complete caches without overwriting them.

2. **[P2, open for M3] Validation-score agreement is not data provenance.**
   `src/predict.py:76-99` accepts different training bytes when validation
   Macro-F1 agrees; it does not audit the local manifest or bind all three arrays
   to checkpoint tensor fingerprints. Altering training/test arrays while keeping
   validation unchanged would leave this gate unchanged. Different predictions
   can also yield the same aggregate F1. Keep this as a numerical reproduction
   diagnostic; do not claim it confirms identical data/normalisation in
   `M3_RESULTS.md` section 4.1 or the prediction module's docstring. Before export,
   run `src.prepare` to verify raw files and audit the local cache. For a stronger
   portable checkpoint contract, store canonical array/cohort/preprocessing
   fingerprints at training time and verify them on inference. Do not simply
   remove the original exact-hash safeguard in `src.train.export_test`.

3. **[P2, open for M3] An unmatched prediction CSV can still enter the report.**
   `src/evaluate.py:129-143` only warns when a caller supplies a processed-data
   directory but no test NPZ matches the CSV. Aggregated metrics can therefore be
   written for mismatched labels/order; beat-level analyses are then unverified.
   Recommend failing when an explicitly supplied reference cannot be matched,
   with a separate opt-in mode for generic CSV analysis. Also validate an existing
   `subject_id` column before replacing it in `attach_beat_identity` (line 112).
   The current committed checkpoint validation checks do not exercise this bad
   input case; this finding does not assert that the frozen results are wrong.

4. **[P3, fixed] Model tests were excluded from normal discovery.**
   Moved `src/test_models.py` to `tests/test_models.py`, preserving its five test
   cases and adding the existing suite's optional-PyTorch skip convention.
   The standard test directory now includes the waveform/RR interfaces,
   backpropagation and exact encoder-transfer checks. CI also installs
   scikit-learn so the two independent metric cross-checks execute.

## Verified integration and remaining constraints

- All 18 committed checkpoints identify E0 as `random` and E1-E5 as
  `patient_strict`, with seeds 42/43/44, an epoch limit of 30 and patience 4.
  Local CPU validation reproduced every stored Macro-F1 within the exporter
  tolerance of 0.001; the largest difference was **0.0003281931**. None of their
  training archive byte hashes matches the original M1 files, which is recorded
  rather than hidden. These are validation checks, not new test-set results.
- M2 imports the exact `ECGEncoder` used by M1. The waveform classifier has
  13,508 parameters; the RR variant has 13,524. All six published E6 encoder
  exports pass the independent provenance/interface audit on the original cohort.
- `src.train` remains CUDA-only; `src.predict` is the documented CPU inference
  route. `ECGCNN.load_pretrained` retains an exact archive-hash requirement, so
  a rebuilt cache may require audited re-pretraining before a future E6 run.
  There is no supervised E6 result in this handoff: **H6 is not tested**.
- The seed-file filter correctly excludes validation/history CSVs. Confusion
  orientation, undefined metrics, record bootstrap and post-hoc threshold/ranking
  logic were read and exercised by the tests. Matching conventional/strict test
  beat identities can support identity/raw-RR analysis; their normalised X/RR
  arrays must not be described as identical or mixed in model inputs.

## Executed verification

See `M1_HANDOFF_VALIDATION.json` for machine-readable results:

- 71 discovered tests passed, with zero failures, errors or skips.
- Original raw data and all nine original processed archives passed audit.
- A real metadata-only clone rebuilt all nine archives in a new temporary
  directory; every stored array equalled the original local build exactly.
  This comparison used the same local environment and does not establish
  cross-platform bitwise equality.
- Both updated M1 notebooks compiled and their portable cells executed with
  `patient_strict`; all six encoder exports and all 18 validation scores checked.
- Both report contributions were regenerated from their updated generators.

No classifier was retrained, no test predictions/results were replaced, and no
Google-hosted Colab run or independent human review is claimed. M1 recommends
keeping the frozen E0-E5 results with M3's early-stopping limitation; any fixed-epoch
follow-up should remain an explicitly post-hoc sensitivity analysis.
