# M1 delivery status — 23 October 2026

## Delivered and executed

| Plan responsibility | Deliverable and evidence |
|---|---|
| Data acquisition | 44 original ECG records / 132 source files in the repository; official SHA256 verification |
| Beat segmentation | MLII, 256 samples, four-class targets, auditable exclusions; 100,630 retained beats |
| Both data splits | Random and conventional DS1/DS2, separate validation; strict subject-disjoint variant also supplied |
| Augmentation | Noise, gain, baseline wander, shift; training-only with deterministic epoch/seed behaviour |
| RR features | Four documented features; per-training-partition standardisation; offline timing limits disclosed |
| Autoencoder pretraining | Completed six ten-epoch runs, 3 seeds x 2 protocols; encoder exports, history, hashes and audit |
| Reproduction | Data and E6 Colab notebooks, source CLIs, safe cache preparation, epoch-level resume, Linux CI |
| Report contribution | Detailed methods, concise integration text, primary-source related work, BibTeX, E6 diagnostics and AI-use contribution |
| Implementation review | `M1_REVIEW.md`, automated safeguards and regression tests; distinct from a human teammate review |
| M2/M3 handoff | Array/CSV conventions, strict encoder transfer contract, cohort/normalisation checks and experiment caveats |

All machine-executable M1 implementation and local training deliverables above
are available. The six E6 runs use 41,355 (`patient`) or 39,394 (`patient_strict`)
training beats per seed. Their reconstruction metrics are not classifier scores.

## External steps that cannot honestly be marked completed by M1 code

1. **Google-hosted Colab runtime evidence and Drive consent.** Both notebooks are
   ready. The browser automation safety check stopped the 23 October attempt
   because it could not confidently identify the current Chrome URL. A user must
   manually open the notebooks, connect a runtime, run them and save outputs.
   Linux CI and local notebook-cell execution are recorded separately.
2. **Human peer review.** A teammate must review M1's code. M1's reciprocal review
   awaits M2/M3 source, which is absent from the shared repository at this snapshot.
   AI review is not relabelled as a human review.
3. **Group integration.** M2/M3 own classifier training/evaluation and the combined
   final report. They must fix one patient protocol and use matching architecture
   for E1/E6. M1 supplies complete reusable inputs; no missing classifier outcomes
   are invented to make the project appear finished.

The optional E6 *pretraining* item is no longer deferred. Its supervised E6
experiment belongs to M2/M3, as do E0-E5 classifier scores and failure analysis.
