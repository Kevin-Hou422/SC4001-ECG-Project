# E6 executed pretraining diagnostics

These are actual full-training-cohort runs performed on Windows CPU using the
settings in `docs/E6_PRETRAINING.md`. Every run completed ten epochs. No validation
or test waveform was used for optimisation or epoch selection. The initial/final
MSE values below are evaluated on each run's full **training** cohort with a fixed
model; per-epoch online losses in the summaries are a different diagnostic.

| Protocol | Seed | Training beats | Initial training MSE | Final training MSE |
|---|---:|---:|---:|---:|
| patient | 42 | 41,355 | 4.469169 | 0.007019 |
| patient | 43 | 41,355 | 3.335566 | 0.004591 |
| patient | 44 | 41,355 | 1.920249 | 0.003666 |
| patient_strict | 42 | 39,394 | 4.462384 | 0.007677 |
| patient_strict | 43 | 39,394 | 3.332288 | 0.005304 |
| patient_strict | 44 | 39,394 | 1.924186 | 0.006162 |

- patient: final training reconstruction MSE 0.005092 ± 0.001732 (sample SD, three seeds).
- patient_strict: final training reconstruction MSE 0.006381 ± 0.001202 (sample SD, three seeds).

The standard deviations quantify variation across these three initialisation and
shuffle seeds on a fixed cohort. They are not confidence intervals for new
patients. Lower reconstruction error shows successful optimisation of the
pretraining objective; it does not demonstrate improved heartbeat classification.
No supervised E6 result is reported here. M2/M3 must compare a scratch encoder
with the identical pretrained architecture using the same supervised protocol.

## Report-ready E6 methods contribution

For the optional E6 condition, we pretrained a convolutional autoencoder on the
waveforms from the supervised training partition only. Three stride-two
convolutional stages with 16, 32 and 64 channels, GroupNorm and ReLU produced a
64-by-32 representation from each 256-sample beat. A mirrored transposed-convolution
decoder reconstructed the standardised waveform. Training minimised MSE with
AdamW (learning rate 0.001, weight decay 0.0001, batch size 256) for a prespecified
ten epochs, using seeds 42, 43 and 44. We performed this pretraining separately for
the conventional and strict patient protocols. Labels, validation and DS2 were
excluded from optimisation, and the final epoch supplied the encoder export.
M2's supervised comparison must retain identical encoder and head architectures
for the scratch and pretrained conditions. The reconstruction diagnostics above
are reported as pretraining checks, not as evidence of patient generalisation.

See `artifacts/pretraining/` for encoder weights, settings, source/data checksums
and complete epoch histories. Reproduce with `python -m src.pretrain`; independently
check all exports with `python -m src.pretrain_audit`.
