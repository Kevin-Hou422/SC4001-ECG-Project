# 5 Discussion and limitations, 6 Conclusion

> Draft v1 (M3, 27 Oct). Builds on `M3_RESULTS.md`; every number is from
> `tables/summary_27oct.md` or `tables/export_check.csv`. Citation keys refer
> to `REFERENCES_VERIFIED.md`. About 800 words for Section 5 (target 1.5 pages)
> and 200 for Section 6.

## 5 Discussion

**The headline number depends on the split.** The same 13.5k-parameter CNN,
trained in the same way, scores macro-F1 0.874 when beats are split at random
and 0.429 when the test patients are new, and on new patients it detects no
S or F beats. This matches the direction reported in the literature
[deChazal2004, Sellami2019]. It also shows how much the random-split number
overstates what a deployed model would do. The random split also changes which
patients are tested, so the gap is not caused by leakage alone. The
clinically relevant number is still the patient-wise one.

**Why the inexpensive fixes did not recover S.** Each fix targeted a plausible
cause, and the failure analysis shows why each fell short:

- *Imbalance.* Reweighting the loss (E2) did not teach the network what an S
  beat looks like in a new patient. It taught the network to call whole
  patients S. Within record 232 its S score was no better than chance, and it
  called most N beats of records 212, 222 and 100 S. Focal loss (E3) changed
  almost nothing. The class imbalance is real, but it is not what blocks
  generalisation.
- *Recording differences.* Augmentation (E4) simulates gain, noise and baseline
  changes, but the S beats of record 232 differ from its N beats in timing, not
  shape (template correlation r = 0.975, `M3_DATA_FINDINGS.md` F5). As
  expected (H3), it did not help.
- *Timing.* RR features (E5) did teach the network the right cue. Within record
  232 it ranks S beats above N beats almost perfectly (AUROC 0.994), against
  0.43 for the baseline. But no test beat crosses the decision boundary.
  Choosing a threshold on the validation patients does not help, because S
  scores are not comparable between patients. In the strict training set, 62%
  of S beats sit inside one patient's tachycardia runs, where they are not
  early. The training data therefore teaches a weaker timing cue than the
  test patients need.

The most useful lesson for evaluation is the gap between pooled and
within-patient AUROC (0.75–0.86 against 0.46–0.48 for E1–E4). A pooled ranking
metric can look reasonable when the model only separates patients that have
many S beats from patients that have few. Within-patient analysis exposes this.

**Model selection under patient shift.** Early stopping on validation macro-F1
chose epoch 1–5 in 14 of 15 patient-split runs. At those checkpoints the
network had not yet learned S even on its own training patients (≤ 3% training
S recall for E1, E3, E4 and two E5 seeds). This is itself a symptom of the
shift: learning S on the training patients did not raise validation macro-F1,
because the S beats of the validation patients look different. Where S was
learned (E2 and one E5 seed, training S recall 0.61–0.90), validation S
recall stayed ≤ 0.08. So the failure to generalise does not depend on the
early stop. Still, our E3–E5 comparisons partly compare under-trained models,
and a selection rule that does not rely on four validation patients would
make them cleaner. **[TBD, team: state whether a fixed-epoch sensitivity run was
done; see `M3_HANDOFF.md`.]**

**Limitations.**

- *Few patients.* There are 22 test patients, and two of them hold most of the S
  (232: 75%) and F (213: 93%) beats. Our S and F conclusions describe a handful
  of individuals. The patient-bootstrap intervals are wide, and with 22 records
  they are themselves approximate [Efron1993].
- *Validation set.* Four patients (175 S, 17 F beats) are a weak basis for
  early stopping and threshold choice, and this showed in the results.
- *Seeds and multiple comparisons.* Three seeds per configuration, and SDs as
  large as 0.155 (E2 accuracy). We report intervals, not significance tests,
  and do not correct for comparing four fixes.
- *Data.* One lead (MLII). Beats are located from the annotations instead of a
  detector. The post-RR feature makes this an offline task. Paced records and
  the Q class are excluded, and there is no external test database.
- *Post-hoc analyses.* The ranking and threshold analyses were designed after
  we saw the first test results. They explain the failure; they are not tuned
  fixes, and no model or threshold was selected on test data.

**Future work.** The within-patient ranking result suggests that the missing
step is per-patient calibration rather than more features. Options include
normalising RR features by each patient's own rhythm, adapting the decision
threshold with a few labelled beats from the new patient, as in
patient-specific training [Kiranyaz2016], or unsupervised test-time adaptation.
Any of these needs a validation design with more patients, for example
leave-one-patient-out on DS1, so that selection is not driven by four records.

## 6 Conclusion

On MIT-BIH, a 1D CNN that scores macro-F1 0.874 on a random beat split
scores 0.429 on patients it has not seen, and detects none of their S or F
beats. Four inexpensive fixes did not change this. Class weighting raised S
sensitivity to 0.83, but only by labelling entire patients S (S +P 0.115).
Focal loss and augmentation had no effect on S. RR features gave the only
reliable macro-F1 gain (+0.016, through V). Their main effect was hidden: the
model learned to rank S beats correctly within each patient (AUROC 0.994 in
record 232) but never predicted them. A threshold chosen on other patients did
not fix this. Patient-wise evaluation should report per-patient results and
within-patient ranking alongside pooled scores, because pooled scores can
hide both failures and partial successes. The next step is to calibrate
decisions per patient.
