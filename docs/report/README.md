# Report workspace

Section drafts in Markdown so they can be reviewed in Git, then merged by M3
into the final document (A4, single column, Arial 10 pt, at most 10 pages
excluding cover, contents, references and appendix). Full draft: Fri 30 Oct.

| Report section (target length) | Owner | Draft file(s) | Status (27 Oct) |
|---|---|---|---|
| 1 Introduction (1 p) | M3 | `M3_INTRODUCTION.md` | Draft v2, results filled in |
| 2 Review of existing techniques (1.5 p) | All | 2.1 classification + imbalance (M2): starter text in `M3_RELATED_WORK.md`; 2.2 timing features + augmentation (M1): `M1_RELATED_WORK.md`; 2.3 evaluation + 2.4 interpretability (M3): `M3_RELATED_WORK.md` | M1 and M3 parts drafted; M2 to revise 2.1; 2.4 to drop unless saliency is done |
| 3 Data and methods (2.5 p) | M1 (data), M2 (models) | `M1_REPORT_INSERT.md` (concise, use this), `M1_DATA_METHODS.md` (detailed); E6 pretraining methods in `M1_E6_RESULTS.md`; dataset findings in `M3_DATA_FINDINGS.md` | Data and E6 parts drafted; models part not started (M2) |
| 4 Experiments and results (3 p) | M2 + M3 | 4.1 setup: M2 (facts listed in `M3_RESULTS.md`); 4.2–4.4 results + failure analysis: `M3_RESULTS.md`; hypotheses: `M3_RESULTS_PLAN.md` | Draft v1 for E0–E5; E6 not run |
| 5 Discussion and limitations (1.5 p) | M3, input from all | `M3_DISCUSSION.md` | Draft v1 |
| 6 Conclusion (0.5 p) | M3 | `M3_DISCUSSION.md` | Draft v1 |
| AI Use Statement | All | `M1_AI_USE_DRAFT.md` + M3 notes below | M2 to add |
| References | All | `REFERENCES_VERIFIED.md` (what each source may be cited for) and `references.bib` | 28 entries verified |
| Figures | M3 | `figures/` (PNG + PDF): 3 data figures and 5 result figures (copied from `results/summary/figures/` on 27 Oct) | Ready |
| Result tables | M3 | `tables/summary_27oct.md`, `tables/export_check.csv` (frozen copies of the evaluated results) | Ready |

**Protocol.** The main patient protocol is `patient_strict` (M2's runs, 27 Oct).
M1's methods draft still calls `patient` the default; align it.

## Rules we follow

- **Numbers.** Every number comes from code output (`results/data_analysis/`,
  `results/summary/`, `data/processed/class_statistics.csv`), never typed in by
  hand from memory. Outputs made from dummy predictions are stamped and never used.
- **Citations.** Cite only claims listed as verified in `REFERENCES_VERIFIED.md`
  (or M1's verified-sources table), or that you have checked yourself and added.
  The handout deducts marks for hallucinated content.
- **Test set.** DS2 is for final evaluation only; findings about it can shape
  analysis and hypotheses, not methods.
- **AI Use Statement.** Record tool use as it happens (who, which tool, for
  what, how it was checked). M1: see `M1_AI_USE_DRAFT.md` (OpenAI Codex). M3:
  the evaluation code, figures, reference checks and these drafts were prepared
  with Claude Code (Claude Opus 5.5), then tested (`tests/test_evaluate.py`,
  `tests/test_saliency.py`); each reference was checked against Crossref,
  PubMed/publisher abstracts and arXiv. **[TBD: M3 to add own review and edits.]**
