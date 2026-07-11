"""
07_week3_rq2_characteristics.py -- Week 3, Research Question 2.

RQ2: What are the characteristics of the issues in each topic?

For each topic model (LDA k=7 and BERTopic k=7) we describe and test whether
the following issue characteristics differ across topics:
  * number of comments
  * number of attachments
  * description length (characters)
  * preprocessed token count
  * issue type (Bug / Improvement / New Feature / ...)

Method
  * Numeric characteristics: box plots per topic + Kruskal-Wallis omnibus test
    (non-parametric: comment/attachment counts are heavily right-skewed).
    If the omnibus test is significant (p < 0.05) we run Dunn's post-hoc test
    with Bonferroni correction to see which topic pairs differ.
  * Issue type (categorical): topic x issue_type contingency table + chi-square
    test of independence, plus a heatmap.

Outputs (data/outputs/week3/):
  rq2_topic_characteristics_summary.csv   per-topic descriptive stats
  rq2_kruskal_tests.csv                    omnibus H, p, significance
  rq2_dunn_<model>_<char>.csv              post-hoc pairwise p (when significant)
  rq2_issue_type_chi2.csv                  chi-square for topic x issue_type
  rq2_box_<model>_<char>.png               box plots
  rq2_issue_type_<model>.png               issue-type heatmap
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

from week3_common import build_base, W3, NUMERIC_CHARS, TOPIC_COLS

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
ALPHA = 0.05

base = build_base()
kruskal_rows = []
summary_rows = []

for tcol, tname in TOPIC_COLS:
    topics = sorted(base[tcol].unique())

    # ---- descriptive summary per topic -------------------------------------
    for t in topics:
        sub = base[base[tcol] == t]
        row = {"model": tname, "topic": t, "n_issues": len(sub)}
        for col, _ in NUMERIC_CHARS:
            row[f"{col}_median"] = round(float(sub[col].median()), 2)
            row[f"{col}_mean"] = round(float(sub[col].mean()), 2)
        row["top_issue_type"] = sub["issue_type"].mode().iat[0] if len(sub) else ""
        summary_rows.append(row)

    # ---- Kruskal-Wallis + Dunn per numeric characteristic ------------------
    for col, label in NUMERIC_CHARS:
        groups = [base[base[tcol] == t][col].values for t in topics]
        H, p = stats.kruskal(*groups)
        kruskal_rows.append({
            "model": tname, "characteristic": col,
            "H_statistic": round(float(H), 4), "p_value": round(float(p), 5),
            "significant": bool(p < ALPHA),
        })
        if p < ALPHA:
            dunn = sp.posthoc_dunn(base, val_col=col, group_col=tcol,
                                   p_adjust="bonferroni")
            dunn.to_csv(os.path.join(W3, f"rq2_dunn_{tname}_{col}.csv"))

        # box plot
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=base, x=tcol, y=col, palette="Set2")
        sig = "significant" if p < ALPHA else "not significant"
        plt.title(f"{label} per {tname} topic\nKruskal-Wallis H={H:.2f}, "
                  f"p={p:.4g} ({sig})")
        plt.xlabel(f"{tname} topic")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(os.path.join(W3, f"rq2_box_{tname}_{col}.png"), dpi=130)
        plt.close()

    # ---- issue_type x topic : chi-square + heatmap -------------------------
    ct = pd.crosstab(base[tcol], base["issue_type"])
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    low_exp = float((expected < 5).mean()) * 100
    kruskal_rows.append({
        "model": tname, "characteristic": "issue_type (chi2)",
        "H_statistic": round(float(chi2), 4), "p_value": round(float(p), 5),
        "significant": bool(p < ALPHA),
    })
    pd.DataFrame({
        "model": [tname], "chi2": [round(float(chi2), 4)],
        "dof": [int(dof)], "p_value": [round(float(p), 5)],
        "pct_expected_lt5": [round(low_exp, 1)],
        "significant": [bool(p < ALPHA)],
    }).to_csv(os.path.join(W3, f"rq2_issue_type_chi2_{tname}.csv"), index=False)

    plt.figure(figsize=(9, 5))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", cbar_kws={"label": "issues"})
    plt.title(f"Issue type by {tname} topic (chi2={chi2:.1f}, p={p:.4g})")
    plt.ylabel(f"{tname} topic")
    plt.xlabel("Issue type")
    plt.tight_layout()
    plt.savefig(os.path.join(W3, f"rq2_issue_type_{tname}.png"), dpi=130)
    plt.close()

# ---- write tables ----------------------------------------------------------
pd.DataFrame(summary_rows).to_csv(
    os.path.join(W3, "rq2_topic_characteristics_summary.csv"), index=False)
kruskal_df = pd.DataFrame(kruskal_rows)
kruskal_df.to_csv(os.path.join(W3, "rq2_kruskal_tests.csv"), index=False)

print("=== RQ2 omnibus tests ===")
print(kruskal_df.to_string(index=False))
print(f"\nOutputs written to {W3}")
