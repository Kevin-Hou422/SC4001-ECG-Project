# 2 Review of existing techniques

> Draft v2 (M3, 22 Oct). Targeting 1.5 pages in total. The plan gives one
> area to each member: **2.1 → M2, 2.2 → M1 (`M1_RELATED_WORK.md`), 2.3 and
> 2.4 → M3**. 2.1 is a starting draft for M2 to rewrite and shorten. Every claim below
> is listed as verified in `REFERENCES_VERIFIED.md`; please keep it that way when editing.

## 2.1 Heartbeat classification on MIT-BIH  *(owner: M2)*

The MIT-BIH Arrhythmia Database contains 48 half-hour two-channel recordings
from 47 subjects, sampled at 360 Hz [MITDB, Moody2001]. De Chazal et al.
[deChazal2004] set the reference inter-patient protocol: the 44 non-paced
recordings are divided into two sets of 22 (DS1, DS2), one used to choose the
classifier configuration and the other only for the final assessment. Their
statistical classifier combined morphology, heartbeat-interval and RR-interval
features. Later feature-based work targeted generalisation directly:
Llamedo and Martínez [Llamedo2011] selected features using cross-database
generalisation criteria, and Mondéjar-Guerra et al. [MondejarGuerra2019]
combined RR intervals with several morphological descriptors in an ensemble
of support vector machines.

Deep networks learn features directly from the waveform. Kiranyaz et al.
[Kiranyaz2016] trained a 1D CNN for each patient using common plus
patient-specific labelled beats, which avoids the inter-patient problem but
needs labelled beats from every new patient. Acharya et al. [Acharya2017]
trained a 9-layer CNN on data augmented to balance the five classes and
reported 94.03% accuracy, falling to 89.07% when trained on the original
imbalanced data; Kachuee et al. [Kachuee2018] reported 93.4% average accuracy
and transferred the representation to myocardial infarction detection. Because
about 89% of beats are N, accuracy at this level is close to what a classifier
that always predicts N achieves (Section 1). Work that reports the
inter-patient paradigm includes Niu et al. [Niu2020], who reduced inter-patient
variation with a symbolic representation that includes baseline correction
(F1 76.6% for S, 89.7% for V), and Mousavi and Afghah [Mousavi2019] and Sellami
and Hwang [Sellami2019], who report both paradigms. At a much larger scale,
Hannun et al. [Hannun2019] trained a deep network on 91,232 single-lead ECGs
from 53,549 patients and evaluated it on an independent test set, which shows
what is possible with many patients. MIT-BIH offers 47 subjects, so here
patient-level generalisation is the main constraint.

**Class imbalance** *(M2, since E2/E3 are M2's losses)*. Buda et al.
[Buda2018] found class imbalance to be detrimental for CNNs and oversampling
to be the most effective remedy in almost all their scenarios; they also note
that accuracy is a poor metric for imbalanced data. An alternative is to change
the loss. Class weights based on the effective number of samples [Cui2019]
increase the cost of errors on rare classes, and focal loss [Lin2017]
down-weights well-classified examples so that training focuses on hard ones.
For heartbeats, Sellami and Hwang [Sellami2019] proposed a batch-weighted loss
to counter class imbalance, and Acharya et al. [Acharya2017] balanced the
classes by augmentation.

## 2.2 Timing features and data augmentation  *(owner: M1)*

Use M1's integration-ready paragraph in `M1_RELATED_WORK.md` (de Chazal 2004;
Nonaka and Seita 2020; Raghu et al. 2022). If space allows, it could add that
time-series augmentation for CNNs has also been studied for wearable sensors
[Um2017] and benchmarked across many datasets [Iwana2021].

## 2.3 Evaluation protocols, leakage and patient-level uncertainty  *(owner: M3)*

An evaluation is only meaningful if the relationship between training and test
data matches that between training data and future use. Saeb et al.
[Saeb2017] showed that record-wise cross-validation often massively
overestimates accuracy compared with subject-wise validation, and found the
record-wise design in almost half of the clinical wearable studies they
reviewed. Kapoor and Narayanan [Kapoor2023] found leakage in 17 fields,
affecting 294 papers, sometimes with wildly over-optimistic conclusions. Heartbeat
classification is a clear case. The same method can show near-perfect
intra-patient scores but far lower inter-patient positive predictivity
[Sellami2019, Mousavi2019], and even the conventional inter-patient split is
not fully subject-disjoint: records 201 (DS1) and 202 (DS2) come from the same
person, and PhysioNet advises placing them in the same partition for
subject-independent evaluation [MITDB].

How results are summarised matters as well. ANSI/AAMI EC57 emphasises
record-by-record presentation of results [AAMI2012], yet the headline results
of the inter-patient papers above are single figures per class, pooled over DS2. In
DS2 these pooled figures are dominated by a few recordings (Section 3), and
beats from the same patient are not independent, so intervals computed by
treating beats as independent samples would be too narrow. Resampling whole
clusters (here, patients) is a standard bootstrap approach for clustered data
[Field2007, Davison1997, Efron1993]. We therefore report per-record results,
results without the dominant records, and patient-bootstrap intervals,
alongside the conventional pooled metrics.

## 2.4 Interpretability  *(owner: M3; keep only if the saliency analysis is done)*

Integrated gradients [Sundararajan2017] attribute a prediction to input
samples by integrating gradients along a path from a baseline input, and
satisfy the sensitivity and implementation-invariance axioms. We use it to ask
whether the CNN's S predictions depend on the parts of the beat that differ
between S and N beats **[TBD: result]**.
