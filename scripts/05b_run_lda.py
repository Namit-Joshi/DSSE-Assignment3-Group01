"""
05b_run_lda.py -- Week 2 : LDA topic modeling (reconstructed pipeline).

Reproduces the LDA stage of Week 2:
  Iteration 1 : LDA with 10 topics (alpha = eta = 0.01).
  Coherence   : C_V coherence for k = 1..10 to pick the optimal k.
  Iteration 2 : replace tokens with ontology class labels, re-run LDA at best k.

Parameters match the report: alpha = 0.01, eta(beta) = 0.01, 20 passes,
400 iterations, random_state = 50.

NOTE ON REPRODUCIBILITY
  * Exact topic terms depend on the Gensim version and the ontology mapping.
  * Iteration 2 used an ontology sheet (data/raw/ontology_sheet.xlsx) that is
    not included in this repository. If that file is present it is used;
    otherwise a small ILLUSTRATIVE fallback ontology (built from the class
    examples in the assignment brief) is applied and a warning is printed.
  * To avoid overwriting the canonical Week-2 outputs used in the report, this
    script writes to data/outputs/week2/lda_repro/.
"""

import os
import warnings
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed", "tika_preprocessed_issues.csv")
RAW = os.path.join(ROOT, "data", "raw", "tika_jira_issues_raw.csv")
ONTO = os.path.join(ROOT, "data", "raw", "ontology_sheet.xlsx")
OUT = os.path.join(ROOT, "data", "outputs", "week2", "lda_repro")
os.makedirs(OUT, exist_ok=True)

SEED = 50
ALPHA, ETA = 0.01, 0.01
PASSES, ITERS = 20, 400
K1 = 10          # Iteration 1 topic count
K_RANGE = range(1, 11)
TOPN = 10

# Illustrative fallback ontology (used only if ontology_sheet.xlsx is absent).
FALLBACK_ONTOLOGY = {
    "COMPONENT": ["machine", "service", "class", "method", "module", "component",
                  "handler", "parser", "server", "processor"],
    "CONNECTOR": ["send", "write", "retrieve", "read", "transfer", "receive",
                  "fetch", "return", "call"],
    "DATA": ["message", "object", "dump", "record", "stream", "field", "content"],
    "SOLUTION": ["layer", "mvc", "replication", "authentication", "cache",
                 "pattern", "pipeline"],
    "TECHNOLOGY": ["java", "pdf", "xml", "json", "maven", "jar", "library",
                   "net", "microsoft", "sonatype"],
    "QUALITY_ATTRIBUTE": ["performance", "security", "reliability", "memory",
                          "vulnerability", "vulnerable", "cve"],
}


def load_token_lists():
    df = pd.read_csv(PROC)
    col = "tokens" if "tokens" in df.columns else "preprocessed_text"
    df[col] = df[col].fillna("")
    return df["issue_key"].tolist(), [str(t).split() for t in df[col]]


def load_ontology_map():
    """token -> CLASS mapping. Prefer the real sheet; else fallback."""
    if os.path.exists(ONTO):
        sheet = pd.read_excel(ONTO)
        # Expected columns: term, class  (adjust to your sheet's header names).
        lc = {c.lower(): c for c in sheet.columns}
        term_col = lc.get("term") or lc.get("token") or sheet.columns[0]
        class_col = lc.get("class") or lc.get("ontology") or sheet.columns[1]
        m = {str(r[term_col]).lower().strip(): str(r[class_col]).strip().upper()
             for _, r in sheet.iterrows() if str(r[term_col]).strip()}
        print(f"Loaded ontology sheet: {len(m)} term mappings.")
        return m
    print("WARNING: ontology_sheet.xlsx not found -> using ILLUSTRATIVE "
          "fallback ontology (results will differ from the canonical run).")
    return {t: cls for cls, terms in FALLBACK_ONTOLOGY.items() for t in terms}


def train_lda(token_lists, k):
    dic = corpora.Dictionary(token_lists)
    corpus = [dic.doc2bow(t) for t in token_lists]
    lda = LdaModel(corpus=corpus, id2word=dic, num_topics=k,
                   alpha=ALPHA, eta=ETA, passes=PASSES, iterations=ITERS,
                   random_state=SEED)
    return lda, dic, corpus


def save_topics(lda, k, path):
    rows = [{"topic": t, "term": w, "probability": round(float(p), 4)}
            for t in range(k) for w, p in lda.show_topic(t, TOPN)]
    pd.DataFrame(rows).to_csv(path, index=False)


def save_assignments(lda, corpus, keys, k, path):
    rows = []
    for key, bow in zip(keys, corpus):
        dist = dict(lda.get_document_topics(bow, minimum_probability=0))
        probs = [round(float(dist.get(t, 0)), 4) for t in range(k)]
        dom = max(range(k), key=lambda t: probs[t])
        row = {"issue_key": key, "dominant_topic": dom,
               "dominant_prob": probs[dom]}
        row.update({f"topic_{t}_prob": probs[t] for t in range(k)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    keys, tokens = load_token_lists()
    print(f"Loaded {len(tokens)} issues.")

    # ---- Iteration 1 -------------------------------------------------------
    lda1, dic1, corpus1 = train_lda(tokens, K1)
    save_topics(lda1, K1, os.path.join(OUT, "lda_iter1_topics.csv"))
    save_assignments(lda1, corpus1, keys, K1,
                     os.path.join(OUT, "lda_iter1_issue_assignments.csv"))
    print(f"Iteration 1: {K1} topics trained.")

    # ---- Coherence sweep ---------------------------------------------------
    coh_rows = []
    best_k, best_c = K1, -1
    for k in K_RANGE:
        lda_k, dic_k, corpus_k = train_lda(tokens, k)
        cm = CoherenceModel(model=lda_k, texts=tokens, dictionary=dic_k,
                            coherence="c_v")
        c = float(cm.get_coherence())
        coh_rows.append({"k": k, "coherence_cv": c})
        if c > best_c:
            best_c, best_k = c, k
        print(f"  k={k:2d}  C_V={c:.4f}")
    pd.DataFrame(coh_rows).to_csv(
        os.path.join(OUT, "lda_coherence_scores.csv"), index=False)
    print(f"Best k = {best_k} (C_V = {best_c:.4f}).")

    # ---- Iteration 2 : ontology replacement --------------------------------
    onto = load_ontology_map()
    n_repl = 0
    tokens2 = []
    for tl in tokens:
        new = []
        for w in tl:
            if w in onto:
                new.append(onto[w]); n_repl += 1
            else:
                new.append(w)
        tokens2.append(new)
    print(f"Iteration 2: {n_repl} ontology replacements.")

    lda2, dic2, corpus2 = train_lda(tokens2, best_k)
    save_topics(lda2, best_k, os.path.join(OUT, "lda_iter2_topics.csv"))
    save_assignments(lda2, corpus2, keys, best_k,
                     os.path.join(OUT, "lda_iter2_issue_assignments.csv"))
    print(f"Iteration 2: {best_k} topics trained. Outputs in {OUT}")


if __name__ == "__main__":
    main()
