# 1 Introduction

> Draft v2 (M3, 27 Oct: results filled in from `M3_RESULTS.md`). About 700
> words, targeting one page. Citation keys refer to `REFERENCES_VERIFIED.md`. All dataset numbers come from
> `python -m src.evaluate describe-data` (`results/data_analysis/data_facts.json`).

Automatic heartbeat classification from the electrocardiogram (ECG) supports
long-term monitoring, where at 60–100 beats per minute a single day of
recording contains roughly 100,000 beats, far more than a clinician can inspect.
The MIT-BIH Arrhythmia Database [Moody2001, Goldberger2000] is the standard
benchmark, with beats grouped into the AAMI classes normal (N),
supraventricular ectopic (S), ventricular ectopic (V), fusion (F) and unknown
(Q) [AAMI2012].

Reported results depend heavily on how the beats are split. If beats from all
recordings are pooled and divided at random (the *intra-patient* paradigm),
beats from the same person appear in both the training and the test set, so a
model can succeed by recognising the person rather than the arrhythmia. Under
this paradigm deep models report near-perfect scores; for example, Mousavi and
Afghah report 100% sensitivity for S beats at 96.5% positive predictivity
[Mousavi2019]. When training and test recordings are disjoint (the
*inter-patient* split of de Chazal et al. [deChazal2004]) the task is much
harder: their classifier found 75.9% of S beats at a positive predictivity of
only 38.5%. Sellami and Hwang report 99.48% accuracy for their CNN
intra-patient but 88.34%, with 48.25% positive predictivity, inter-patient
[Sellami2019]. The same pattern appears across clinical machine learning:
record-wise cross-validation often massively overestimates accuracy compared
with subject-wise validation [Saeb2017], and leakage between training and test
data has produced over-optimistic conclusions in many fields [Kapoor2023]. A
deployed classifier always meets a new patient, so only a patient-wise
evaluation answers the clinically relevant question.

The patient-wise benchmark has a difficulty of its own: abnormal beats are rare
and concentrated in a few people. In the standard test set (DS2; 49,661 beats
after our preprocessing) S beats make up 3.7% of beats, and a single recording,
232, contains 75% of them. Recording 213 contains 93% of the test F beats, and
94% of the training F beats come from a different single recording, 208. An
aggregate S or F score therefore mostly describes one or two individuals, and
accuracy says little: predicting N for every beat already scores 89.0%
accuracy, with a macro-F1 of only 0.24.

This project asks **how much a 1D convolutional neural network loses when it is
evaluated on patients it has never seen, and which inexpensive changes recover
performance on the rare abnormal classes.** We train the same baseline on a
random-beat split and on a patient-wise split, and then test changes that each
target one suspected cause of the gap: class-weighted and focal losses for
class imbalance [Cui2019, Lin2017]; waveform augmentation (gain, noise,
baseline wander, shift) for differences in recording conditions between
patients [Um2017, Iwana2021]; and RR-interval features, because in our data S
beats are mostly early relative to the local rhythm while their shape is close
to that of N beats (Section 3).

Our contributions are:

- A controlled measurement of the random-split versus patient-split gap for one
  model and one pipeline: macro-F1 falls from 0.874 to 0.429, and S and F F1
  fall from 0.74 and 0.81 to zero. Our patient split keeps subjects disjoint:
  records 201 and 202 come from the same person, which the conventional DS1/DS2
  split ignores, so we train without record 201.
- A comparison of four fixes over three training seeds, reporting per-class
  sensitivity, positive predictivity and F1 as well as macro-F1. None recovers
  S detection on new patients. Class weighting appears to, but only by labelling
  whole patients as S.
- A patient-level analysis: per-record results, results with and without the
  dominant records 232 (S) and 213 (F), and patient-bootstrap confidence
  intervals that show how much each conclusion depends on which patients are
  in the test set.
- A data analysis showing that the timing cue the RR fix relies on differs
  between patients: 62% of training S beats occur during supraventricular
  tachycardia runs, where they are not early relative to the local rhythm,
  while 75% of test S beats are isolated early beats from one patient with
  sinus bradycardia.
- A failure analysis showing that RR features teach the network to rank S beats
  correctly *within* a patient (AUROC 0.994 in record 232) even though it never
  predicts S, and that no threshold set on other patients fixes this.

Section 2 reviews related work, Section 3 describes the data and methods,
Section 4 presents the experiments, Section 5 discusses failure cases and
limitations, and Section 6 concludes.
