"""
08_week3_rq3_cooccurrence.py -- Week 3, Research Question 3.

RQ3:
  (a) What topics significantly co-occur between LDA topics and BERTopics?
  (b) What LDA / BERT topics significantly co-occur with the three design
      decision types (existence, property, executive)?
  (c) Do the three design-decision types significantly co-occur with each other?

Methods
  (a) LDA topic x BERT topic : a 7x7 contingency table, an omnibus chi-square
      test of independence, and standardized (Pearson) residuals to identify
      which specific (LDA, BERT) topic pairs co-occur more / less than expected
      (|residual| > 2 ~ locally significant at ~0.05).

  (b) topic x decision : each issue belongs to exactly one topic but may carry
      several decision types (multi-label), so for every (topic, decision) pair
      we build a 2x2 table (in-topic vs not) x (has-decision vs not) and run a
      Fisher exact test (exact, robust to small cells). Odds ratios > 1 mean the
      decision is over-represented in that topic. p-values are corrected across
      all pairs per model with Benjamini-Hochberg (FDR).

Outputs (data/outputs/week3/):
  rq3_lda_vs_bert_contingency.csv    7x7 counts
  rq3_lda_vs_bert_residuals.csv      standardized residuals
  rq3_lda_vs_bert_chi2.csv           omnibus chi-square
  rq3_lda_vs_bert_heatmap.png
  rq3_topic_decision_fisher.csv      every (model, topic, decision): OR, p, p_adj
  rq3_topic_decision_omnibus.csv     chi-square topic vs each decision
  rq3_<model>_decision_heatmap.png   proportion of each topic carrying a decision
  rq3_decision_cooccurrence.csv      pairwise decision x decision: OR, phi, p, p_adj
  rq3_decision_combination_counts.csv  8-way breakdown of decision combinations
  rq3_decision_cooccurrence_heatmap.png  phi association + co-occurrence counts
"""

import os
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from week3_common import build_base, W3, DECISIONS, TOPIC_COLS

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
ALPHA = 0.05

base = build_base()

# ============================================================================
# (a) LDA topic x BERT topic co-occurrence
# ============================================================================
ct = pd.crosstab(base["lda_topic"], base["bert_topic"])
chi2, p, dof, expected = stats.chi2_contingency(ct)
residuals = (ct - expected) / np.sqrt(expected)   # standardized (Pearson) resid
low_exp = float((expected < 5).mean()) * 100

ct.to_csv(os.path.join(W3, "rq3_lda_vs_bert_contingency.csv"))
residuals.round(3).to_csv(os.path.join(W3, "rq3_lda_vs_bert_residuals.csv"))
pd.DataFrame({
    "test": ["LDA_topic x BERT_topic"], "chi2": [round(float(chi2), 4)],
    "dof": [int(dof)], "p_value": [round(float(p), 6)],
    "pct_expected_lt5": [round(low_exp, 1)], "significant": [bool(p < ALPHA)],
}).to_csv(os.path.join(W3, "rq3_lda_vs_bert_chi2.csv"), index=False)

# significant pairs (|resid| > 2)
sig_pairs = [
    {"lda_topic": int(l), "bert_topic": int(b),
     "count": int(ct.loc[l, b]), "std_residual": round(float(residuals.loc[l, b]), 2),
     "direction": "over" if residuals.loc[l, b] > 0 else "under"}
    for l in ct.index for b in ct.columns if abs(residuals.loc[l, b]) > 2
]
pd.DataFrame(sig_pairs).to_csv(
    os.path.join(W3, "rq3_lda_vs_bert_sig_pairs.csv"), index=False)

plt.figure(figsize=(8, 6))
sns.heatmap(ct, annot=True, fmt="d", cmap="viridis",
            cbar_kws={"label": "shared issues"})
plt.title(f"LDA topic x BERT topic co-occurrence\nchi2={chi2:.1f}, "
          f"p={p:.4g}, dof={dof}")
plt.xlabel("BERTopic")
plt.ylabel("LDA topic")
plt.tight_layout()
plt.savefig(os.path.join(W3, "rq3_lda_vs_bert_heatmap.png"), dpi=130)
plt.close()

# ============================================================================
# (b) topic x decision-type co-occurrence
# ============================================================================
fisher_rows = []
omnibus_rows = []

for tcol, tname in TOPIC_COLS:
    topics = sorted(base[tcol].unique())

    # proportion of each topic carrying each decision (for heatmap)
    prop = pd.DataFrame(index=topics, columns=DECISIONS, dtype=float)

    for d in DECISIONS:
        # omnibus: topic (categorical) vs decision (binary)
        octab = pd.crosstab(base[tcol], base[d])
        c2, pv, dof_, exp_ = stats.chi2_contingency(octab)
        omnibus_rows.append({
            "model": tname, "decision": d, "chi2": round(float(c2), 4),
            "dof": int(dof_), "p_value": round(float(pv), 5),
            "pct_expected_lt5": round(float((exp_ < 5).mean()) * 100, 1),
            "significant": bool(pv < ALPHA),
        })

        for t in topics:
            in_t = base[tcol] == t
            has_d = base[d]
            a = int((in_t & has_d).sum())     # in topic & has decision
            b = int((in_t & ~has_d).sum())    # in topic & no decision
            c = int((~in_t & has_d).sum())    # not topic & has decision
            e = int((~in_t & ~has_d).sum())   # not topic & no decision
            OR, pval = stats.fisher_exact([[a, b], [c, e]])
            prop.loc[t, d] = a / max(1, (a + b))
            fisher_rows.append({
                "model": tname, "topic": int(t), "decision": d,
                "n_topic": a + b, "n_topic_with_decision": a,
                "pct_topic_with_decision": round(100 * a / max(1, a + b), 1),
                "odds_ratio": round(float(OR), 3) if np.isfinite(OR) else np.inf,
                "p_value": round(float(pval), 5),
            })

    # heatmap of decision proportions per topic
    plt.figure(figsize=(6, 5))
    sns.heatmap(prop.astype(float), annot=True, fmt=".2f", cmap="rocket_r",
                vmin=0, vmax=1, cbar_kws={"label": "fraction with decision"})
    plt.title(f"Design-decision share per {tname} topic")
    plt.xlabel("Decision type")
    plt.ylabel(f"{tname} topic")
    plt.tight_layout()
    plt.savefig(os.path.join(W3, f"rq3_{tname}_decision_heatmap.png"), dpi=130)
    plt.close()

fisher_df = pd.DataFrame(fisher_rows)
# Benjamini-Hochberg FDR correction within each model.
fisher_df["p_adj"] = np.nan
for tname in fisher_df["model"].unique():
    m = fisher_df["model"] == tname
    fisher_df.loc[m, "p_adj"] = multipletests(
        fisher_df.loc[m, "p_value"], method="fdr_bh")[1].round(5)
fisher_df["significant"] = fisher_df["p_adj"] < ALPHA

fisher_df.to_csv(os.path.join(W3, "rq3_topic_decision_fisher.csv"), index=False)
pd.DataFrame(omnibus_rows).to_csv(
    os.path.join(W3, "rq3_topic_decision_omnibus.csv"), index=False)

# ============================================================================
# (c) decision-type x decision-type co-occurrence
# ============================================================================
# Each decision is a binary flag on the issue. For every pair we build a 2x2
# table and test association with Fisher exact (odds ratio) and report the phi
# coefficient (effect size). p-values are FDR-corrected across the three pairs.
dec_rows = []
for d1, d2 in combinations(DECISIONS, 2):
    x, y = base[d1], base[d2]
    a = int(( x &  y).sum())   # both
    b = int(( x & ~y).sum())   # d1 only
    c = int((~x &  y).sum())   # d2 only
    e = int((~x & ~y).sum())   # neither
    OR, pval = stats.fisher_exact([[a, b], [c, e]])
    denom = np.sqrt((a + b) * (c + e) * (a + c) * (b + e))
    phi = (a * e - b * c) / denom if denom else 0.0
    dec_rows.append({
        "decision_1": d1, "decision_2": d2,
        "both": a, "only_1": b, "only_2": c, "neither": e,
        "odds_ratio": round(float(OR), 3) if np.isfinite(OR) else np.inf,
        "phi": round(float(phi), 3),
        "p_value": round(float(pval), 5),
    })
dec_df = pd.DataFrame(dec_rows)
dec_df["p_adj"] = multipletests(dec_df["p_value"], method="fdr_bh")[1].round(5)
dec_df["significant"] = dec_df["p_adj"] < ALPHA
dec_df.to_csv(os.path.join(W3, "rq3_decision_cooccurrence.csv"), index=False)

# full 8-way combination breakdown (which decision sets actually appear).
combo = (base[DECISIONS].astype(int)
         .groupby(DECISIONS).size().reset_index(name="n_issues")
         .sort_values("n_issues", ascending=False))
combo["pct"] = (100 * combo["n_issues"] / len(base)).round(1)
combo.to_csv(os.path.join(W3, "rq3_decision_combination_counts.csv"), index=False)

# symmetric matrices: co-occurrence counts (diagonal = total) and phi.
count_mat = pd.DataFrame(0, index=DECISIONS, columns=DECISIONS, dtype=int)
phi_mat = pd.DataFrame(np.nan, index=DECISIONS, columns=DECISIONS, dtype=float)
for d in DECISIONS:
    count_mat.loc[d, d] = int(base[d].sum())
    phi_mat.loc[d, d] = 1.0
for r in dec_rows:
    d1, d2 = r["decision_1"], r["decision_2"]
    count_mat.loc[d1, d2] = count_mat.loc[d2, d1] = r["both"]
    phi_mat.loc[d1, d2] = phi_mat.loc[d2, d1] = r["phi"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(count_mat, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            cbar_kws={"label": "issues"})
axes[0].set_title("Decision co-occurrence counts\n(diagonal = total issues)")
sns.heatmap(phi_mat.astype(float), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, ax=axes[1],
            cbar_kws={"label": "phi coefficient"})
axes[1].set_title("Decision association (phi)")
plt.tight_layout()
plt.savefig(os.path.join(W3, "rq3_decision_cooccurrence_heatmap.png"), dpi=130)
plt.close()

# ============================================================================
# console summary
# ============================================================================
print("=== RQ3(a) LDA x BERT ===")
print(f"chi2={chi2:.2f}, p={p:.4g}, dof={dof}, "
      f"expected<5: {low_exp:.0f}%  -> "
      f"{'SIGNIFICANT' if p < ALPHA else 'not significant'}")
print(f"significant (|resid|>2) topic pairs: {len(sig_pairs)}")

print("\n=== RQ3(b) omnibus topic x decision ===")
print(pd.DataFrame(omnibus_rows).to_string(index=False))

print("\n=== RQ3(b) significant (FDR<0.05) topic-decision pairs ===")
sig = fisher_df[fisher_df["significant"]]
if len(sig):
    print(sig[["model", "topic", "decision", "pct_topic_with_decision",
               "odds_ratio", "p_value", "p_adj"]].to_string(index=False))
else:
    print("none survived FDR correction")

print("\n=== RQ3(c) decision x decision co-occurrence ===")
print(dec_df[["decision_1", "decision_2", "both", "odds_ratio", "phi",
              "p_value", "p_adj", "significant"]].to_string(index=False))
print("\ndecision combination breakdown:")
print(combo.to_string(index=False))
print(f"\nOutputs written to {W3}")
