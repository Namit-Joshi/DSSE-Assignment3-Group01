"""
week3_common.py -- shared data assembly for Week 3 (RQ2 & RQ3).

Builds one issue-level table that joins, per Tika issue:
  * topic assignments   : lda_topic (LDA iteration 2, k=7) and bert_topic (k=7)
  * design decisions     : existence, property, executive  (booleans)
  * characteristics      : comment_count, attachment_count, description length
                           (chars) and token_count (preprocessed size)
  * issue_type           : Jira issue type
"""

import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "raw", "tika_jira_issues_raw.csv")
PROC = os.path.join(ROOT, "data", "processed", "tika_preprocessed_issues.csv")
W2 = os.path.join(ROOT, "data", "outputs", "week2")
W3 = os.path.join(ROOT, "data", "outputs", "week3")
os.makedirs(W3, exist_ok=True)

DECISIONS = ["existence", "property", "executive"]


def _to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def build_base(lda_file="lda_iter2_issue_assignments.csv"):
    """Return the merged issue-level dataframe used by RQ2 and RQ3."""
    raw = pd.read_csv(RAW).drop_duplicates("issue_key")
    proc = pd.read_csv(PROC).drop_duplicates("issue_key")

    raw["desc_len_chars"] = raw["description"].fillna("").astype(str).str.len()
    base = raw[["issue_key", "issue_type", "comment_count", "attachment_count",
                "desc_len_chars"] + DECISIONS].copy()
    for d in DECISIONS:
        base[d] = _to_bool(base[d])

    base = base.merge(proc[["issue_key", "token_count"]], on="issue_key", how="left")

    lda = pd.read_csv(os.path.join(W2, lda_file))
    base = base.merge(lda[["issue_key", "dominant_topic"]]
                      .rename(columns={"dominant_topic": "lda_topic"}),
                      on="issue_key", how="left")

    bert = pd.read_csv(os.path.join(W2, "bertopic_issue_assignments.csv"))
    base = base.merge(bert[["issue_key", "bert_topic"]],
                      on="issue_key", how="left")

    # Drop the rare issue with no BERT/LDA topic (outlier), if any.
    base = base.dropna(subset=["lda_topic", "bert_topic"])
    base["lda_topic"] = base["lda_topic"].astype(int)
    base["bert_topic"] = base["bert_topic"].astype(int)
    return base


# Characteristics analysed in RQ2: (column, human label)
NUMERIC_CHARS = [
    ("comment_count", "Number of comments"),
    ("attachment_count", "Number of attachments"),
    ("desc_len_chars", "Description length (chars)"),
    ("token_count", "Preprocessed token count"),
]

TOPIC_COLS = [("lda_topic", "LDA"), ("bert_topic", "BERTopic")]


if __name__ == "__main__":
    b = build_base()
    print(f"base shape: {b.shape}")
    print(b.head())
    print("\nLDA topic sizes:\n", b["lda_topic"].value_counts().sort_index())
    print("\nBERT topic sizes:\n", b["bert_topic"].value_counts().sort_index())
