# M3 handoff (evaluation and report)

## Update 9 Nov: final check against the handout; three code/README items

M3 checked the report and the code against the project handout. The report
meets the format rules (A4, Arial 10, single column, main text 8 of 10 pages)
and has every required part. Every number in it matches the results. The
reproduction route also works from the committed checkpoints. All 72 tests
pass, the data audit passes, and `src.predict` followed by
`src.evaluate summarise` gives the same `summary.md` as before. All 18
checkpoints match their stored validation macro-F1 to within 3 × 10⁻⁴.

`main.tex` now holds M3's revised sections. In your sections, M3 changed only
formatting: inline maths ([−128, +128), 10⁻³, γ = 2, 64 × 32 and so on) is now
plain text, so the whole PDF is in Arial. The team paragraph of the AI Use
Statement now also acknowledges the libraries we built on, since the handout
asks us to acknowledge existing code.

Three items are left in the code and README. Please fix them before we build
the submission ZIP.

1. **M2: `src/train.py` defaults do not match the runs (lines 364–365).** The
   defaults are `--epochs 25` and `--patience 6`, but all 18 checkpoints were
   trained with 30 and 4 (stored in each checkpoint as `epochs_limit` and
   `patience`). The docstring's command `python -m src.train --experiments all
   --seeds 42 43 44` would therefore train different models from the ones in the
   report. Please change the defaults to 30 and 4. The README and Appendix A
   pass the flags explicitly, so they are already correct.

2. **M2: unfinished sentence in the `src/train.py` docstring (line 6).** It
   reads "AFTER model selection is frozen," and then stops. Suggested
   replacement:

   ```text
   After model selection is frozen, export test predictions with
     python -m src.predict --experiments all --seeds 42 43 44   (CPU or GPU; used for the report)
   or
     python -m src.train --mode predict-test                    (CUDA only)
   ```

3. **M1 and M2: the README has no single "how to reproduce" section.** The
   handout asks for instructions that cover dependencies, data preparation and
   commands. Ours are all there, but split across "Local reproduction",
   "Evaluation (M3)" and "M2 report". Also, the training command (line 323)
   is not in a code block, and `requirements.txt` has no PyTorch, which
   prediction and training both need. Please add a section near the top, for
   example:

   ````markdown
   ## Reproducing the main results

   Python 3.10+. Run everything from the repository root.

   ```bash
   # 1. Dependencies (includes PyTorch; for training, install a CUDA build of PyTorch)
   python -m pip install -r requirements-pretrain.txt
   # 2. Data: verify raw checksums, build and audit the NPZ arrays, run the tests
   python -m src.prepare
   python -m src.data audit
   python -m unittest discover -s tests -v
   # 3. Optional retraining of E0-E5 (NVIDIA GPU with CUDA). The 18 checkpoints
   #    are committed; new directories avoid overwriting them.
   python -m src.train --experiments all --seeds 42 43 44 --patient-protocol patient_strict \
       --epochs 30 --patience 4 --checkpoints-dir checkpoints_retrain --results-dir results_retrain
   # 4. Test predictions from the committed checkpoints (CPU is enough)
   python -m src.predict --experiments all --seeds 42 43 44
   # 5. Tables and figures in the report
   python -m src.evaluate describe-data                     # Table 1, Figs 1 and 5
   python -m src.evaluate summarise --reference E1 --n-boot 2000   # Tables 2-4, Figs 2-4, C1
   ```
   ````

   While editing, please also update the opening line ("M1 data pipeline for the
   group project") to describe the whole project. The internal notes in
   "M1 report and handoff" ("for M3 to shorten and merge…") are out of date now
   that the report is finished.

## Update 8 Nov: M2's sections done

M2 finished 2.1, 3.4 and the AI-use sentence (80f5b19); the 7 Nov items below
are closed. M3 checked the facts and references and made formatting-only fixes:
sentence-case headings, citation spacing, and "ChatGPT (GPT-5)". `main.pdf` is
rebuilt (13 pages; main text plus AI statement 8). Still open: the group number
on the cover, and M3's own revision of the M3 sections and AI-use sentence.

## Update 7 Nov: M3 review of the report (please read, M2)

**M1:** all M1 parts are in and checked. Nothing outstanding, thank you.

**M2:** thanks for filling your boxes. Your parts are in
`docs/report/latex/main.tex` on `main`; line numbers below are as of the
7 Nov formatting commit. Please finish items 2–4 by **Mon 9 Nov**, so there
is a week for the final checks before 16 Nov.

1. **Done by M3: formatting and cover.** Your text used Markdown, so the report
   did not compile. M3 converted it to LaTeX without changing your words
   (bold, headings, code blocks, `\texttt{patient\_strict}`, `\cite{…}`),
   recompiled `main.pdf` (no errors, all citations resolved, main text plus AI
   statement 8 pages) and put "Nguyen Khanh" on the cover. When you edit, please
   keep to LaTeX, or write plain sentences and M3 will format them:

   | Instead of (Markdown) | Write (LaTeX) |
   |---|---|
   | `**bold**` | `\textbf{bold}` |
   | `` `patient_strict` `` | `\texttt{patient\_strict}` (every `_` must be written `\_`) |
   | a block in triple backticks | `\begin{verbatim}` … `\end{verbatim}` |
   | `[Buda2018]` | `\cite{Buda2018}` (the keys are in `references.bib`) |
   | `# heading` | nothing; `#` stops LaTeX, and the headings already exist |

2. **2.1 Heartbeat classification on MIT-BIH (line 72), about 300 words.**
   It is now about 75 words on class imbalance only. Please add, before that
   paragraph: MIT-BIH and the DS1/DS2 inter-patient protocol
   (`\cite{dechazal2004automatic}`), and earlier deep classifiers on the
   intra-patient split (Acharya2017, Kachuee2018) versus inter-patient work
   (Niu2020, Mousavi2019, Sellami2019). Your longer revision from 1 Nov in
   `docs/report/M3_RELATED_WORK.md` (2.1) already has most of this. Keep the
   class-imbalance paragraph, and add `\cite{Cui2019}` where you mention
   class weighting.

3. **3.4 Model and training (line 124), about 200 words.** It should describe
   the model and how it was trained. The facts are in
   `docs/report/M3_RESULTS.md` §4.1:
   - the model: 13,508-parameter 1D CNN (M1's convolutional encoder, global
     average pooling, linear head; E5 adds the four RR features before the head,
     13,524 parameters);
   - training: AdamW, lr 1e-3, weight decay 1e-4, batch 256, at most 30 epochs,
     the checkpoint with the best validation macro-F1 is kept, patience 4, seeds
     42–44 (move these here from Appendix A, or keep both);
   - the losses: E2 class weights N 0.025, S 1.389, V 0.337, F 2.250 (inverse
     training frequency); E3 focal loss with γ = 2;
   - the hardware you trained on (GPU model).

   Please remove the run command and the "GPU acceleration is recommended"
   sentence (both are reproduction details and already in Appendix A). Your
   validation paragraph is good; one correction: 169 of the 175 validation
   S beats come from records 118 and 223, so write "mostly from two records".
   "Although, early stopping may have…" should read "Early stopping may
   therefore have…".

4. **AI use (line 282).** Please give the tool's exact name as it appears in
   the app (for example "ChatGPT, GPT-5"), and say what *you* checked. The
   current sentence "Outputs were verified, checked, protocol tests and checks
   with primary literature" is incomplete and repeats M1's checks. For example:
   "Code was checked by running the training script and the unit tests; text
   was checked against the cited papers."

5. **Optional, Appendix A model training (line 322).** The content is fine. If
   you widen `--device` to `choices=["cuda", "cpu", "auto"]` in `src/train.py`
   (line 370; `resolve_device` already supports them), change "Training requires
   an NVIDIA GPU" to "A GPU is recommended".

**Team:** agree the group number for the cover (line 42).

## M1 report contribution completed (1 Nov)

Hou Junning filled M1's sections directly in `docs/report/latex/main.tex`: 2.2,
3.1--3.3 (including the partition/class-count table and the prematurity-index
definition), M1's AI-use paragraph, the data commands in Appendix A, and optional
Appendix B. The cover has M1's full name; the group number is left for the team
to fill in, as requested. M2's report and training-command placeholders remain
for M2. The Markdown-to-LaTeX generator was not rerun; future edits should continue
in `main.tex`.

The matching `main.pdf` was compiled with the XeTeX-based Tectonic engine and
Biber 2.17, with resolved citations/references and no overfull boxes. All pages
were visually checked: A4, single column, Arial body text at 10 TeX points. The
current main text plus AI statement occupies 8 pages; the 12-page PDF also includes
the cover, references and appendices. Allow for M2's remaining text before the
team's final page-count check.

## Answers for M2 (31 Oct evening)

1. **Where does early stopping go?** Describe the *setup* in 3.4 Model and training:
   the checkpoint with the best validation macro-F1 on the four held-out DS1 records
   (108, 118, 203, 223) is kept, at most 30 epochs, patience 4. The *finding* (it
   stopped at epochs 1–5, before S was learned even on the training patients) is
   already in 4.4 (Fig. 4, from your histories) and 5 Discussion. Refer to it
   rather than repeating it. No fixed-epoch rerun: the frozen results stay, as M1
   and M3 agreed, and it is listed as a limitation.
2. **E6: not run; the report says "not tested"** (4.1 and Appendix B), unless the
   team decides otherwise. Reasons: (a) E6 would use the same early stopping, so it
   would also stop after 1–5 epochs and an E6-vs-E1 difference would say little;
   (b) `ECGCNN.load_pretrained` requires the training NPZ to be byte-identical to
   M1's, and your rebuilt data are not; (c) the team target is 1 Nov. If the team
   wants to use the time before 16 Nov instead: an E6 entry in `src/train.py` and
   `src/predict.py`, M1's original NPZs, three seeds, then M3 adds the results.
3. **Your report parts are still empty boxes in `docs/report/latex/main.tex`**
   (branch `m3-report`): 2.1, 3.4 and your AI-use part. The new README section is
   useful for reproduction, but it is not the report. Your edit to
   `M1_AI_USE_DRAFT.md` deleted your own lines and put your ChatGPT sentence
   inside M1's paragraph; please write your part in the report's AI-use box instead.
4. **Two code notes on `train_result`:** (a) the file `tories and summary…` at the
   repository root is a saved `less` help screen, committed by accident in b6ae297;
   please `git rm` it. (b) `resolve_device` now supports `cpu` and `auto`, but
   `--device` still has `choices=["cuda"]`, so the command line cannot select them.

## Update 31 Oct: report skeleton, please write your sections by Sun 1 Nov

The report is `docs/report/latex/main.tex` (PDF: `main.pdf`, branch `m3-report`).
M3's sections are written: 1 Introduction, 2.3, 3.5 Evaluation, 4 Experiments and
results, 5 Discussion, 6 Conclusion, and M3's part of the AI Use Statement. **Each
of your sections is an orange box** in the PDF saying what to write and where your
existing draft or the facts are. Replace the `\placeholder{...}` with your text in
`main.tex`. Compile with `latexmk -xelatex main.tex` in `docs/report/latex/`, or
upload that folder to Overleaf (compiler: XeLaTeX). M3's sections take about
6.5 of the 10 pages, so together you have about 3 pages.

- **M1:** 2.2 Timing features and augmentation; 3.1 Data and preprocessing;
  3.2 Partitions (with a table of beats per partition; add `\label{tab:1}` if you
  want it numbered first); 3.3 RR features and augmentation (please name the fourth
  RR feature the *prematurity index*, which Section 4.4 uses); your AI-use part;
  the data commands in Appendix A; optionally Appendix B (E6). Your
  `M1_REPORT_INSERT.md`, `M1_RELATED_WORK.md` and `M1_AI_USE_DRAFT.md` already hold
  most of this.
- **M2:** 2.1 Heartbeat classification and class imbalance; 3.4 Model and training
  (the facts are in `docs/report/M3_RESULTS.md` §4.1: 30 epochs and patience 4 were
  used, not the script defaults); your AI-use part (exact tool and model name, and
  how you checked its output); the training command in Appendix A. Thanks for the
  histories: they are now Fig. 4.
- **Everyone:** put your full name on the cover and add the group number.
- Table and figure numbers are automatic (`\ref`), so adding a table or figure
  renumbers the rest.
- **M1's PR #1 review, both open findings fixed:** the report and `src/predict.py`
  no longer claim the validation check proves identical data, and `src.predict`
  now runs `ensure_prepared` (raw checksums and cache audit) first; `summarise`
  fails on a prediction CSV that matches no test NPZ unless `--allow-unmatched`
  is given, and a `subject_id` column is checked too. 72 tests pass.
- **Seeds:** three seeds are kept (not more). The main conclusions are far
  outside seed noise, and the patient bootstrap, not the seed SD, dominates
  the uncertainty.
- The data figures now use the strict training split (record 209 holds 60% of
  training S beats, not 50%), and Fig. 4 (training curves) comes from M2's histories.

> M1 response, 28 Oct: the original reference manifest has been restored, both
> M1 methods contributions and setup defaults now use `patient_strict`, and the
> merged PR #1 has an AI-assisted M1 review in `M1_PR1_REVIEW.md` with execution
> evidence in `M1_HANDOFF_VALIDATION.json`. See `M1_HANDOFF.md` for M1's position
> on early stopping and E6. The original M3 requests below are retained for context.

Status 27 Oct: E0–E5 evaluated from M2's checkpoints (3 seeds each,
`patient_strict`). Results, failure analysis, discussion and conclusion are
drafted in `docs/report/M3_RESULTS.md` and `docs/report/M3_DISCUSSION.md`, and
the introduction no longer has placeholders. All 64 tests pass. Branch
`m3-evaluation` is based on M2's `train_result` (6fb469d).

## To do (please send your sections by Thu 29 Oct; M3 assembles the draft Fri 30)

**Both, today (results freeze):** decide on early stopping and E6 (see
"Decisions the team needs to make" below).

**M2:**
1. Push `results/E*/seed*_history.csv` and `results/training_summary.csv`.
2. Write the model and training-setup section (facts in
   `docs/report/M3_RESULTS.md` §4.1) and revise related work 2.1.
3. Add your AI-use notes to `docs/report/M1_AI_USE_DRAFT.md`.
4. Read the code review notes below (most urgent: note 1, `manifest.json`).
5. Confirm it is OK to merge your `train_result` commits into `main` through
   PR #1 (`m3-evaluation`).

**M1:**
1. Decide whether to restore your `manifest.json`
   (`git checkout a4d0f5c -- data/processed/manifest.json`; review note 1).
2. Update `M1_REPORT_INSERT.md` / `M1_DATA_METHODS.md`: the main patient
   split is now `patient_strict`, not `patient`.
3. If you have time, review PR #1 (the plan has each of us review a
   teammate's code).

## Headline results (details: `docs/report/tables/summary_27oct.md`)

| | Macro-F1 | S Se | S +P | V F1 | F F1 |
|---|---|---|---|---|---|
| E0 random split | 0.874 ± 0.009 | 0.618 | 0.910 | 0.955 | 0.814 |
| E1 patient split | 0.429 ± 0.002 | 0.000 | n/a | 0.756 | 0.000 |
| E2 + weighted CE | 0.371 ± 0.067 | 0.830 | 0.115 | 0.629 | 0.004 |
| E3 + focal | 0.444 ± 0.017 | 0.000 | n/a | 0.811 | 0.001 |
| E4 + augmentation | 0.433 ± 0.017 | 0.000 | n/a | 0.771 | 0.001 |
| E5 + RR features | 0.445 ± 0.030 | 0.001 | 0.167 | 0.799 | 0.027 |

1. The gap is large: 0.874 → 0.429 macro-F1. S and F F1 drop to zero.
2. No fix recovers S. E2's S recall comes from labelling whole patients S
   (98% of record 232's beats, 92% of record 212's N beats); within record 232
   its S score is at chance (AUROC 0.48).
3. E5 learned the timing cue: within record 232 it ranks S above N with
   AUROC 0.994 (E1: 0.43), but argmax never picks S, and a threshold chosen on
   validation patients does not transfer (post-hoc analyses, labelled as such).
4. The selected checkpoints of E1, E3, E4 and two E5 seeds detect ≤ 3% of S beats
   **on their own training set**. Early stopping chose epoch 1–5 in 14 of 15
   patient-split runs.

## How the numbers were produced

```bash
python -m src.predict --experiments all --seeds 42 43 44   # test + val CSVs, checks
python -m src.evaluate summarise --reference E1 --n-boot 2000
```

`src/predict.py` (new, M3) does what `python -m src.train --mode predict-test`
does, reusing M2's code, but runs on CPU. Instead of a byte-level hash check it
checks the data behaviourally: every checkpoint reproduced its stored validation
macro-F1 on our data within 3 × 10⁻⁴ (`results/export_check.csv`, copied to
`docs/report/tables/`). It also records each checkpoint's per-class recall on
the training and validation sets.

New in `src/evaluate.py`: the seed-file filter (fix, see below), within-record
S ranking (`ranking_S_<exp>.csv`) and the validation-threshold transfer
analysis (`threshold_transfer_S_<exp>.csv`). Both are reported in
`summary.md` under "Post-hoc".

## Decisions the team needs to make (before the Wed 28 Oct freeze)

1. **Early stopping (M2 + all).** Patient-split models stopped at epoch 1–5,
   before learning S on their own training data. Our conclusion holds anyway,
   because models that did learn S (E2, E5 seed 42) still miss it on new
   patients. But E3–E5 partly compare under-trained models. Options:
   (a) keep the results as they are and report the diagnosis as a finding and
   a limitation (drafted); or (b) M2 also runs E1 and E5 with a fixed number of
   epochs (e.g. 30, last epoch, no early stopping) as a **sensitivity analysis**,
   stated in the report as added after the first test evaluation. M3
   recommends (a), plus (b) if M2 has time. Do not replace the main results.
2. **E6 (stretch).** M1's encoders are ready; no supervised E6 run exists. H6
   stays "not tested" unless M2 runs it by Wed.
3. **Saliency (stretch, M3).** The ranking analysis already answers the
   interpretability question for the report. M3 suggests dropping Section 2.4
   to save space unless someone wants the integrated-gradients figure.

## Requests to M2

- Push `results/E*/seed*_history.csv` and `results/training_summary.csv`
  (results/ is git-ignored; use `git add -f` or the Drive folder). They are
  needed for a training-curve figure and to support the early-stopping
  paragraph.
- Write Section 4.1 (training setup) and the model part of Section 3. The
  facts are in `docs/report/M3_RESULTS.md` §4.1: 13,508 parameters, AdamW
  1e-3, weight decay 1e-4, batch 256, ≤ 30 epochs, patience 4, and the E2
  class weights. Note that the script defaults are 25 epochs and patience 6,
  so state the values actually used, and put the exact command in the README.
- Add your AI-use notes to `docs/report/M1_AI_USE_DRAFT.md`.

## Code review notes for M2 (week 2)

1. **`data/processed/manifest.json` was overwritten with locally rebuilt
   hashes (af46bb6).** With it, `python -m src.data audit` and the M3
   notebook's `ensure_prepared` fail for anyone whose NPZs match M1's original
   build ("File changed: random_train.npz"). Counts, metadata and validation
   scores agree, so the arrays are probably the same and only the compressed
   bytes differ (possibly a different zlib in Python 3.14 on Windows); as M1
   notes, that is not proof. **Resolved 28 Oct: M1 restored the manifest.** Suggest
   restoring M1's manifest (`git checkout a4d0f5c -- data/processed/manifest.json`)
   and not committing locally regenerated metadata (M1 handoff note 2).
2. **The checkpoint hash check is machine-specific.** `export_test` requires the
   training NPZ's file hash to equal the one stored in the checkpoint, so
   `--mode predict-test` fails on any machine whose zlib differs from yours,
   including possibly the marker's. `src.predict` uses the behavioural check
   instead; consider using it or loosening the check.
3. **CUDA-only.** `resolve_device` rejects CPU, so CPU-only machines cannot
   reproduce the results. Inference takes about 2 minutes for all 18 models on CPU.
4. **Test predictions were never exported.** `--mode predict-test` had not been
   run. M3 exported them with `src.predict`. If you also run predict-test on the
   GPU, the CSVs should agree except for a few borderline beats.
5. **`checkpoints/` is in `.gitignore`** but the 18 checkpoints were
   force-added. That is fine (1 MB in total and needed for reproduction), but
   update the README so it does not say checkpoints are unversioned.
6. **Validation macro-F1 counts F (17 beats) and S (175 beats from two
   records).** With so few validation S/F beats, the score mostly tracks N and
   V, which is one reason early stopping fired early.
7. `src/test_models.py` lives in `src/`; the other tests are in `tests/`.
   Moving it would let `python -m unittest discover -s tests` run everything.

## Fixed in M3's own code

- `load_experiment` read every `seed*.csv`, so M2's `seed42_val.csv` and
  `seed42_history.csv` would have been treated as extra seeds. It now reads only
  `seed<k>.csv`, and there is a test for this.

## Data findings (details in `docs/report/M3_DATA_FINDINGS.md`)

- Record 232 holds 75.2% of test S beats; record 213 holds 93.3% of test F
  beats; record 208 holds 94% of training F beats.
- Always predicting N gives 89.0% accuracy and macro-F1 0.236 on the test set.
- Under `patient_strict`, 62% of training S beats (395/640) occur in SVTA runs,
  mostly in record 209, where they are not early; 48.9% of training S beats
  have RR ratio < 0.9, against 87.7% of test S beats.
- In record 232, S beats correlate at r = 0.975 with that record's N beats in
  shape; their timing is what differs.
