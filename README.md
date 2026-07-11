# Assignment 3 — Topic Modeling of Apache Tika Design Issues

**Data Science for Software Engineering** · Group 01
Brian Stevo Aranha · Achara Obinna Vincent · Namit Joshi · Anil Kumar

This is the consolidated submission covering all three weeks: issue data
preparation, topic modeling (LDA + BERTopic), and significance tests on topic
characteristics and co-occurrences. The selected project is **Apache Tika**
(211 issues, each labelled with three design-decision types: *existence*,
*property*, *executive*).

## Report

- `report.tex` — full report source (compilable).
- `report.pdf` — compiled report (Weeks 1–3, RQ1–RQ3).
- Build with **Tectonic**: `tectonic report.tex` (or `latexmk -pdf report.tex`).
- `REPORT_ADDITIONS.md` / `.tex` — the Week-2 (BERTopic/RQ1) and Week-3 sections
  as standalone snippets (already merged into `report.tex`).

## Folder structure

```
Group01_Assignment3_04_07_2026/
├── report.tex / report.pdf        Final report
├── requirements.txt
├── scripts/
│   ├── 01_read_tika_issue_list.py        Week 1: read issue list from issues.xlsx
│   ├── 02_download_tika_jira_issues.py   Week 1: download Jira issue data
│   ├── 03_preprocess_tika_text.py        Week 1: clean/tokenize/lemmatize
│   ├── 04_create_tika_vocabulary.py      Week 1: vocabulary + counts
│   ├── 05_create_tika_document_term_matrix.py  Week 1: document-term matrix
│   ├── 05b_run_lda.py                    Week 2: LDA (iter1 + coherence + iter2)
│   ├── 06_run_bertopic.py                Week 2: BERTopic (k=7)
│   ├── week3_common.py                   Week 3: shared data loader
│   ├── 07_week3_rq2_characteristics.py   Week 3: RQ2 (characteristics + tests)
│   └── 08_week3_rq3_cooccurrence.py      Week 3: RQ3 (co-occurrence tests)
├── data/
│   ├── raw/        issues.xlsx, tika_jira_issues_raw.{csv,json}
│   ├── processed/  tika_issue_list.csv, tika_preprocessed_issues.csv
│   └── outputs/
│       ├── tika_vocabulary.csv, tika_document_term_matrix.csv, ...  (Week 1)
│       ├── week2/  lda_* , bertopic_* , lda_repro/                  (Week 2)
│       └── week3/  rq2_* , rq3_*  (CSVs, box plots, heatmaps)       (Week 3)
├── pdf/            Original submitted PDF
└── archive/        Superseded Week-1-only report (kept for history)
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Week 1  (regenerates data/ — the committed data is the canonical version)
python scripts/01_read_tika_issue_list.py
python scripts/02_download_tika_jira_issues.py
python scripts/03_preprocess_tika_text.py
python scripts/04_create_tika_vocabulary.py
python scripts/05_create_tika_document_term_matrix.py

# Week 2
python scripts/05b_run_lda.py        # LDA (writes to data/outputs/week2/lda_repro/)
python scripts/06_run_bertopic.py    # BERTopic (writes to data/outputs/week2/)

# Week 3
python scripts/07_week3_rq2_characteristics.py
python scripts/08_week3_rq3_cooccurrence.py
```

## Notes

- **Canonical data.** The committed `data/` files are the ones the report and
  Week-2/3 analysis were built on. Re-running the Week-1 scripts regenerates
  them (the Jira download can vary over time).
- **LDA reproducibility.** `05b_run_lda.py` reproduces the coherence scores
  exactly (peak at k=7, C_V=0.3462). Iteration 2 needs the original
  `data/raw/ontology_sheet.xlsx` for an exact match; if absent, an illustrative
  fallback ontology is used and outputs go to `data/outputs/week2/lda_repro/`.
- **BERTopic.** HDBSCAN under-clusters this small corpus (2 topics), so
  `06_run_bertopic.py` falls back to KMeans(k=7) for a balanced, comparable set.
- **Statistics.** RQ2 uses Kruskal–Wallis + Dunn; RQ3 uses χ² with standardized
  residuals and Fisher exact tests with Benjamini–Hochberg (FDR) correction.
  Chi-square small-cell warnings are recorded as `pct_expected_lt5` in the CSVs.

## Remaining tasks (for teammates)

The analysis, results and report are complete. What's left before submission:

1. **Fill in your effort-table rows** in `report.tex` (Appendix A, tables
   `tab:w1`/`tab:w2`/`tab:w3`) — rows marked *(to be filled)*:
   - **Brian** — Week 1, Week 3
   - **Obinna** — Week 2, Week 3
   - **Anil** — Week 2, Week 3

   Edit `report.tex`, then rebuild the PDF (see below).
2. *(Optional)* add the original `data/raw/ontology_sheet.xlsx` so LDA
   Iteration 2 (`scripts/05b_run_lda.py`) reproduces exactly; without it a
   labelled fallback ontology is used.

### Building the report

Install a LaTeX engine, then compile `report.tex`:

```bash
# Tectonic (self-contained, recommended): https://tectonic-typesetting.github.io
tectonic report.tex
# or, with a full TeX Live install:
latexmk -pdf report.tex
```

Please open a pull request (or push to a branch) with your changes rather than
committing directly to `main`, so we can review before the final submission.
