# M1 report contribution: data and features

Draft for integration into the group's ten-page report. This describes the
implemented and locally verified preprocessing. It contains **no CNN results**.
The main patient protocol for the completed E1-E5 runs is `patient_strict`, as
recorded in M3's 27 October handoff. E0 uses `random`. The conventional `patient`
protocol remains available as a reference and was not used for those E1-E5 runs.

## Dataset and task

We use the MIT-BIH Arrhythmia Database v1.0.0, distributed by PhysioNet [1].
Following the project protocol, we exclude paced records 102, 104, 107 and 217,
leaving 44 recordings. Source waveform, header and expert-reviewed annotation
files are retained without modification and checked against publisher SHA-256
checksums. The classification target is an annotated individual heartbeat.

Four superclass labels are used: N={N,L,R,e,j}, S={A,a,J,S}, V={V,E} and
F={F}. Their integer encodings are 0, 1, 2 and 3. N includes non-ectopic beat
types beyond normal sinus beats. The remaining Q-class targets are excluded.
Other annotations such as rhythm changes and signal-quality markers are not
treated as labelled heartbeat samples. This four-class task does not reproduce
the complete five-class protocol of de Chazal et al. [2].

## Segmentation and normalisation

For each record, we select MLII by its channel name rather than assuming it is
the first lead; record 114 uses the second channel. At the original 360 Hz sampling
rate, each annotation centre supplies a 256-sample window extending 128 samples
before and 128 samples after the centre (right endpoint excluded), approximately
0.711 seconds. We use physical signal values and apply no resampling or bandpass
filter. Beat centres are supplied by reference annotations, so this experiment
evaluates classification conditional on annotated beat locations rather than a
complete beat-detection system.

The preprocessing excludes 15 Q-class target annotations,
40 target beats whose waveform windows cross recording
boundaries, and 48 further target beats lacking
previous or next beat context. These exclusions are mutually counted in processing
order. A further 2,991 non-target annotations are
ignored, not counted as rejected supervised beats. The retained dataset contains
**100,630 heartbeats**. The same retained pool is used for baseline and RR-feature
experiments, preventing the E5 comparison from changing its evaluated samples.

For each protocol, a scalar waveform mean and population standard deviation are
fitted using only its training beats and applied unchanged to validation and test
beats. RR features are standardised column-wise using the corresponding training
subset. A lower bound of 1e-8 prevents division by zero. No class oversampling or
waveform augmentation is applied to validation or test arrays.

## Data partitions

The random-beat protocol first separates a stratified 20% test subset, then takes
20% of the remaining development data for validation, giving approximately
64%/16%/20% train/validation/test. The data seed is fixed at 42 and does not vary
with the three model-training seeds. This protocol deliberately permits the same
subject in multiple partitions; nearby windows and timing context may overlap.

The conventional record-based protocol uses the commonly adopted DS1/DS2 record
lists associated with [2]. DS2 is reserved for final evaluation. Records 108, 118,
203 and 223 are fixed as DS1-side validation records, leaving 18 training records.
Validation records provide all four target classes and are not selected using DS2
model performance. However, the official database documentation identifies records
201 and 202 as the same subject [1]. Their placement on opposite sides means that
this conventional benchmark is not completely subject-independent.

Our main `patient_strict` protocol therefore removes record 201 from training,
keeps the same validation records and leaves DS2 unchanged. Its 17 training
subjects, four validation subjects and 22 test subjects are disjoint. This is an
explicit project modification, not the unmodified de Chazal protocol. A comparison
of random and patient splits also changes data composition and training size;
an observed performance difference cannot isolate patient overlap as its sole
cause. No direction or magnitude of that difference is assumed in advance.

The main E1-E5 training cohort contains 39,394 beats (N=35,719, S=640, V=2,640,
F=395), validation contains 9,614 beats (N=8,472, S=175, V=950, F=17), and DS2
contains 49,661 beats (N=44,218, S=1,836, V=3,219, F=388). All E1-E5 experiments
and all three model seeds use these fixed partitions and strict training-fitted
normalisation. E0 alone uses the random protocol. Files from different protocol
prefixes must not be mixed, even where their beat identities coincide.

| Protocol | Partition | Records | Beats | N | S | V | F |
|---|---|---:|---:|---:|---:|---:|---:|
| random | train | 44 | 64,404 | 57,627 | 1,778 | 4,485 | 514 |
| random | val | 44 | 16,101 | 14,407 | 445 | 1,121 | 128 |
| random | test | 44 | 20,125 | 18,008 | 556 | 1,401 | 160 |
| patient | train | 18 | 41,355 | 37,352 | 768 | 2,838 | 397 |
| patient | val | 4 | 9,614 | 8,472 | 175 | 950 | 17 |
| patient | test | 22 | 49,661 | 44,218 | 1,836 | 3,219 | 388 |
| patient_strict | train | 17 | 39,394 | 35,719 | 640 | 2,640 | 395 |
| patient_strict | val | 4 | 9,614 | 8,472 | 175 | 950 | 17 |
| patient_strict | test | 22 | 49,661 | 44,218 | 1,836 | 3,219 | 388 |

*Table: actual retained counts after segmentation and RR-context exclusions.
The strict and conventional test rows identify the same beats but use different
training-fitted normalisation parameters. The table is descriptive data auditing,
not a model-performance result.*

## RR-interval features

We augment waveform information with four timing features: previous RR in seconds,
next RR in seconds, the mean of up to the last ten RR intervals including the
current previous interval, and the previous RR divided by that local mean. These
features are motivated by the use of morphology and heartbeat intervals in [2],
but constitute our specific implementation rather than an exact reproduction.
Intervals are calculated from successive beat locations within each recording,
including Q beat locations as timing context. Rhythm markers do not define RR
intervals, class identities are not used as numerical inputs, and intervals never
cross recording boundaries. Next-RR requires future context; the resulting task
is offline classification. M2's waveform-only baselines do not consume RR features.

## Training augmentation

The E4 data loader can independently apply each of four transforms with probability
0.5: multiplication by a uniform gain in [0.9, 1.1], additive Gaussian noise with
standard deviation 0.02, sinusoidal baseline wander of amplitude 0.05 and frequency
uniformly selected from [0.1, 0.5] Hz with random phase, and an integer alignment
shift in [-8, 8] samples using edge padding rather than circular wraparound.
Additive magnitudes refer to training-standardised signal units. These are fixed
initial experimental settings; no clinical optimality is claimed. The transforms
alter measured waveform appearance while retaining beat timing features and labels.

Augmentation is generated on the fly from the training seed, epoch and row index.
It cannot be enabled on validation or test partitions through the supplied loader.
The intended hypothesis is improved robustness to measurement variation; whether
these transforms help this task must be determined by M2/M3's controlled experiments.
Task-dependent augmentation effects in prior ECG research [3] motivate evaluating
this hypothesis rather than assuming a benefit.

## Audit, reproducibility and limitations

Each derived archive stores waveform and RR arrays, labels, record ID, canonical
subject ID and the source annotation sample. The manifest records source checksums,
processing source hashes, software versions, normalisation parameters, exclusions,
class counts and partition overlaps. Automated auditing checks beat identities,
strict subject disjointness, array shapes, file checksums and fitted transforms.

In DS2, record 232 supplies **1,381 of 1,836 retained S beats
(75.22%)**. We retain it in the principal test set and plan a
supplementary analysis excluding that record to measure sensitivity to this
concentration. This concentration alone is not evidence of classifier failure.
Additional limits include the selected nature and small number of subjects of
MIT-BIH, the conventional protocol's shared subject, and reliance on annotated
centres and future timing. These prevent interpreting benchmark accuracy as
clinical validation or guaranteed generalisation to a new population.

## References

[1] G. Moody and R. Mark, *MIT-BIH Arrhythmia Database*, version 1.0.0, PhysioNet,
2005. https://doi.org/10.13026/C2F305

[2] P. de Chazal, M. O'Dwyer and R. B. Reilly, “Automatic classification of
heartbeats using ECG morphology and heartbeat interval features,” *IEEE TBME*,
51(7), 1196-1206, 2004. https://doi.org/10.1109/TBME.2004.827359

[3] A. Raghu, D. Shanmugam, E. Pomerantsev, J. Guttag and C. M. Stultz,
“Data Augmentation for Electrocardiograms,” *PMLR*, 174, 282-310, 2022.
https://proceedings.mlr.press/v174/raghu22a.html
