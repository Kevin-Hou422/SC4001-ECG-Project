"""Generate E6 diagnostics and a concise M1 report insert from executed evidence."""
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data/processed/manifest.json").read_text(encoding="utf-8"))
    runs = []
    for protocol in ("patient", "patient_strict"):
        for seed in (42, 43, 44):
            path = root / f"artifacts/pretraining/{protocol}/seed{seed}/summary.json"
            run = json.loads(path.read_text(encoding="utf-8"))
            assert run["status"] == "completed" and run["config"]["epochs"] == 10
            runs.append(run)
    report_dir = root / "docs/report"
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        rows.append({"protocol": run["data"]["protocol"], "seed": run["config"]["seed"],
                     "n_train_beats": run["data"]["n_beats"], "epochs": run["config"]["epochs"],
                     "initial_train_mse": run["initial_train_mse"],
                     "final_train_mse": run["final_train_mse"],
                     "elapsed_seconds": run["elapsed_seconds"]})
    with (report_dir / "e6_training_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    table = "\n".join(f"| {row['protocol']} | {row['seed']} | {row['n_train_beats']:,} | "
                        f"{row['initial_train_mse']:.6f} | {row['final_train_mse']:.6f} |"
                        for row in rows)
    aggregates = []
    for protocol in ("patient", "patient_strict"):
        values = [row["final_train_mse"] for row in rows if row["protocol"] == protocol]
        aggregates.append(f"- {protocol}: final training reconstruction MSE "
                          f"{mean(values):.6f} ± {stdev(values):.6f} (sample SD, three seeds).")
    text = """# E6 executed pretraining diagnostics

These are actual full-training-cohort runs performed on Windows CPU using the
settings in `docs/E6_PRETRAINING.md`. Every run completed ten epochs. No validation
or test waveform was used for optimisation or epoch selection. The initial/final
MSE values below are evaluated on each run's full **training** cohort with a fixed
model; per-epoch online losses in the summaries are a different diagnostic.

| Protocol | Seed | Training beats | Initial training MSE | Final training MSE |
|---|---:|---:|---:|---:|
""" + table + "\n\n" + "\n".join(aggregates) + """

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
"""
    (report_dir / "M1_E6_RESULTS.md").write_text(text, encoding="utf-8", newline="\n")
    partition_rows = []
    for protocol, name in (("random", "Random"), ("patient", "Conventional"), ("patient_strict", "Strict")):
        info = manifest["protocols"][protocol]["files"]
        partition_rows.append(f"| {name} | {info['train']['n']:,} | {info['val']['n']:,} | {info['test']['n']:,} |")
    compact = """# M1 report insert: data and feature methods

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
""" + "\n".join(partition_rows) + """

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
"""
    (report_dir / "M1_REPORT_INSERT.md").write_text(compact, encoding="utf-8", newline="\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for protocol, ax in zip(("patient", "patient_strict"), axes):
        for run in runs:
            if run["data"]["protocol"] != protocol:
                continue
            ax.plot([row["epoch"] for row in run["history"]],
                    [row["train_online_mse"] for row in run["history"]],
                    label=f"seed {run['config']['seed']}")
        ax.set(xlabel="Epoch", ylabel="Online training reconstruction MSE", yscale="log", title=protocol)
        ax.legend()
    fig.savefig(root / "docs/pretraining_curves.png", dpi=150)
    plt.close(fig)
    print("Generated E6 diagnostics, concise report insert and training curves.")


if __name__ == "__main__":
    main()
