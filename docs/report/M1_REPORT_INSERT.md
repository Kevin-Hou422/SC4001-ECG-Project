# M1 report insert: data and feature methods

Concise integration text for M3. The group must keep its stated patient protocol
consistent with the final supervised experiments. The detailed M1 methods and
training diagnostics are in the companion files, and no classifier result is
asserted here.

## Data and preprocessing

We used the MIT-BIH Arrhythmia Database v1.0.0 [1], excluding paced records 102,
104, 107 and 217. The 44 retained recordings' original waveform, header and
reference annotation files were verified against publisher checksums. Four
superclasses were mapped as N={N,L,R,e,j}, S={A,a,J,S}, V={V,E} and F={F}; N includes
non-ectopic types beyond normal sinus beats. Q targets and non-beat annotations
were excluded. Using expert annotation centres, we extracted MLII by channel name
and retained 256 samples per beat, spanning [-128,+128) samples at 360 Hz. No
resampling or bandpass filter was applied. Incomplete windows and beats lacking
previous/next RR context were removed, leaving **100,630 labelled beats**.

For each protocol, waveform mean and population SD were fitted on its training
beats and frozen for validation/test; RR columns were standardised likewise.
All experiments use the same retained pool, including the waveform-only models,
so adding RR does not change the evaluated samples.

## Partitions and timing features

The random protocol used a stratified 80/20 development/test split and reserved
20% of development for validation (approximately 64/16/20 overall). Split seed 42
was fixed across model seeds. Same-patient and neighbouring-beat context can occur
across random partitions. The conventional protocol used DS1/DS2 record lists
associated with [2], with DS1 records 108, 118, 203 and 223 held out for validation.
DS2 was reserved for final evaluation. Since records 201 and 202 originate from
one subject [1], conventional DS1/DS2 is not strictly subject-independent. A strict
variant removes training record 201 while preserving validation and DS2. Changing
split also changes cohort composition and training size, so an E0/E1 difference
cannot be attributed solely to patient overlap.

| Protocol | Train beats | Validation beats | Test beats |
|---|---:|---:|---:|
| Random | 64,404 | 16,101 | 20,125 |
| Conventional | 41,355 | 9,614 | 49,661 |
| Strict | 39,394 | 9,614 | 49,661 |

The four RR features are previous RR, next RR, the mean of up to ten preceding RR
intervals including the current previous interval, and the previous RR divided by
that mean. Morphology-plus-interval classification in [2] motivates this comparison;
our CNN and feature definitions are not an exact reproduction. Only consecutive
beat locations within a recording define intervals; Q beats remain timing context,
while rhythm markers do not. Next-RR and annotated centres make this an offline
classification task rather than an end-to-end real-time detector.

## Augmentation and optional pretraining

Training-only augmentation independently applies gain scaling [0.9,1.1], Gaussian
noise (SD 0.02), sinusoidal baseline wander (amplitude 0.05, frequency 0.1-0.5 Hz,
random phase), and edge-padded shifts within eight samples, each with probability
0.5. Additive amplitudes are in standardised units. Labels and RR features stay
fixed because the transforms represent measurement variation. No augmentation is
applied to validation/test. Augmentation effects are task dependent [3], so an
improvement must be established experimentally.

For E6, M1 pretrained three-stage convolutional autoencoders (16/32/64 channels,
GroupNorm/ReLU, mirrored decoder) for ten fixed epochs on each training cohort,
using AdamW, LR 0.001, weight decay 0.0001 and batch size 256, with seeds 42/43/44.
Only waveform reconstruction MSE was optimised; labels, validation and DS2 were
unused. Six encoder exports and training histories are available. The supervised
E1/E6 comparison must use identical architectures and differ only in encoder
initialisation; reconstruction improvement alone does not establish a classification
or generalisation benefit.

## Reproducibility and limits

NPZ archives preserve record ID, canonical subject ID and annotation location.
Audits check file/tensor hashes, training-only fitting, beat identities and strict
subject separation. Record 232 supplies 1,381/1,836 retained DS2 S beats (75.22%);
the full test set remains primary, with an exclusion analysis as a sensitivity
check. Small subject numbers, selected data, annotated centres and future timing
limit deployment claims. No unseen-patient performance is inferred from setup or
pretraining diagnostics.

## References

[1] Moody and Mark. MIT-BIH Arrhythmia Database v1.0.0. PhysioNet, 2005.
https://doi.org/10.13026/C2F305

[2] de Chazal et al. Automatic classification of heartbeats using ECG morphology
and heartbeat interval features. IEEE TBME 51(7):1196-1206, 2004.
https://doi.org/10.1109/TBME.2004.827359

[3] Raghu et al. Data Augmentation for Electrocardiograms. PMLR 174:282-310, 2022.
https://proceedings.mlr.press/v174/raghu22a.html
