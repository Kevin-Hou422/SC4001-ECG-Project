# M1 related-work contribution: timing features and data augmentation

This is a focused contribution to the group's shared literature review, not a
claim to survey all ECG classification work. Source scope is recorded below;
no unverified performance numbers are imported into the report.

## Integration-ready paragraph

Heartbeat classification can combine waveform morphology with temporal context.
de Chazal et al. [1] investigated ECG morphology and heartbeat-interval features
using separate development and evaluation record sets. Their formulation motivates
our comparison between an isolated-waveform CNN and a model additionally consuming
RR features. We use a different neural model, four target classes and an explicitly
defined four-feature RR vector, so our work is an investigation inspired by this
approach rather than a reproduction of its classifier or reported scores.

Training-time augmentation provides another way to test robustness without
increasing model size. Nonaka and Seita [2] investigated ECG augmentation for
single-lead atrial-fibrillation classification. This supports considering waveform
transformations, but their task does not establish that the same choices help
four-class MIT-BIH heartbeat classification. Raghu et al. [3] found augmentation
effects to depend on the ECG prediction task and introduced TaskAug, which learns
a task-specific policy. We instead fix a small, interpretable collection of noise,
gain, baseline and alignment perturbations and evaluate it as a controlled change.
Our implementation does not reproduce ECG Augment or TaskAug, and we do not claim
their published results as evidence of an improvement in our own experiments.

## Verified sources and how they are used

| Source | Verified source material | Supported use here | Boundary |
|---|---|---|---|
| [1] de Chazal et al., 2004 | Original-paper abstract and bibliographic metadata via PubMed | Morphology plus timing features; separate record sets | No unsupported assertion that every conventional record split has unique subjects |
| [2] Nonaka and Seita, 2020 | Authors' arXiv abstract | ECG augmentation studied for single-lead AF | Preprint; different task; not an exact implementation source |
| [3] Raghu et al., 2022 | Publisher's PMLR paper page and abstract | Task-dependent augmentation; learned TaskAug policy | Our fixed transforms do not implement TaskAug |
| [4] PhysioNet MIT-BIH v1.0.0 | Official database description, record directory and source annotations | Sampling, dataset provenance and 201/202 subject identity | Dataset properties do not prove any model effect |

The abstract-level review is sufficient for the limited claims above. Detailed
architectural, optimisation or numerical comparisons would require checking the
corresponding full papers before adding those claims to the final report.

## References

[1] P. de Chazal, M. O'Dwyer and R. B. Reilly. Automatic classification of
heartbeats using ECG morphology and heartbeat interval features. *IEEE Transactions
on Biomedical Engineering*, 51(7):1196-1206, 2004.
https://doi.org/10.1109/TBME.2004.827359
Original abstract: https://pubmed.ncbi.nlm.nih.gov/15248536/

[2] N. Nonaka and J. Seita. Data Augmentation for Electrocardiogram Classification
with Deep Neural Network. *arXiv:2009.04398*, 2020.
https://arxiv.org/abs/2009.04398

[3] A. Raghu, D. Shanmugam, E. Pomerantsev, J. Guttag and C. M. Stultz.
Data Augmentation for Electrocardiograms. *Proceedings of the Conference on
Health, Inference, and Learning*, PMLR 174:282-310, 2022.
https://proceedings.mlr.press/v174/raghu22a.html

[4] G. Moody and R. Mark. MIT-BIH Arrhythmia Database, version 1.0.0. PhysioNet,
2005. https://doi.org/10.13026/C2F305
