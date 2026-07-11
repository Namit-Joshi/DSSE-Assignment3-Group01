"""
06_run_bertopic.py  -- Week 2 (re-run) : BERTopic with finer topic granularity.

The first BERTopic run collapsed the 211 Apache Tika issues into only 2 topics
(178 + 33), which is too coarse for the Week 3 co-occurrence significance tests.
This script re-runs BERTopic tuned for a small, short-document corpus:

  * SentenceTransformer 'all-MiniLM-L6-v2' embeddings of the preprocessed text
  * UMAP with a small neighbourhood (fixed random_state for reproducibility)
  * HDBSCAN with a small min_cluster_size so several topics can form
  * outlier reduction, so every issue is assigned to a real topic (no -1),
    which is required so the BERT topic can be cross-tabulated per issue.

If HDBSCAN still yields < MIN_ACCEPTABLE_TOPICS topics, we fall back to KMeans
with a fixed k (comparable to the LDA k=7) to guarantee a usable partition.

Outputs (data/outputs/week2/), matching the original file schema:
  bertopic_topic_info.csv        Topic, Count, Name, Representation, Representative_Docs
  bertopic_topics.csv            topic, term, score
  bertopic_issue_assignments.csv issue_key, bert_topic, bert_prob, existence,
                                 property, executive, issue_type, status
The previous 2-topic files are backed up to *_2topics_backup.csv first.
"""

import os
import shutil
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic

# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PROC = os.path.join(ROOT, "data", "processed", "tika_preprocessed_issues.csv")
RAW = os.path.join(ROOT, "data", "raw", "tika_jira_issues_raw.csv")
OUT = os.path.join(ROOT, "data", "outputs", "week2")
os.makedirs(OUT, exist_ok=True)

SEED = 42
MIN_ACCEPTABLE_TOPICS = 4     # below this we switch to KMeans
KMEANS_K = 7                  # comparable to the LDA optimum (k = 7)
TOP_N_TERMS = 10

# ----------------------------------------------------------------------------
# 1. Load preprocessed text + the metadata Week 3 needs.
proc = pd.read_csv(PROC)
raw = pd.read_csv(RAW)

text_col = "preprocessed_text" if "preprocessed_text" in proc.columns else "tokens"
proc = proc.dropna(subset=[text_col]).reset_index(drop=True)
docs = proc[text_col].astype(str).tolist()
keys = proc["issue_key"].tolist()
print(f"Loaded {len(docs)} preprocessed issue documents.")

meta = raw[["issue_key", "existence", "property", "executive",
            "issue_type", "status"]].drop_duplicates("issue_key")

# ----------------------------------------------------------------------------
# 2. Embeddings (downloads the MiniLM model on first run, ~90 MB).
print("Embedding documents with all-MiniLM-L6-v2 ...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(docs, show_progress_bar=False)

# ----------------------------------------------------------------------------
# 3. Tuned BERTopic (HDBSCAN path).
umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                  metric="cosine", random_state=SEED)
hdbscan_model = HDBSCAN(min_cluster_size=5, min_samples=3, metric="euclidean",
                        cluster_selection_method="eom", prediction_data=True)
vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 1))

topic_model = BERTopic(
    embedding_model=embedder,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    calculate_probabilities=True,
    verbose=False,
)
topics, probs = topic_model.fit_transform(docs, embeddings)

n_topics = len([t for t in set(topics) if t != -1])
n_outliers = int(np.sum(np.array(topics) == -1))
print(f"HDBSCAN path: {n_topics} topics, {n_outliers} outliers.")

# ----------------------------------------------------------------------------
# 4. Fall back to KMeans if too few topics formed.
used_method = "hdbscan"
if n_topics < MIN_ACCEPTABLE_TOPICS:
    print(f"< {MIN_ACCEPTABLE_TOPICS} topics -> switching to KMeans(k={KMEANS_K}).")
    used_method = f"kmeans_k{KMEANS_K}"
    cluster_model = KMeans(n_clusters=KMEANS_K, random_state=SEED, n_init=10)
    topic_model = BERTopic(
        embedding_model=embedder,
        umap_model=umap_model,
        hdbscan_model=cluster_model,   # BERTopic accepts any sklearn clusterer
        vectorizer_model=vectorizer,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(docs, embeddings)
else:
    # Reassign outliers to their nearest topic so every issue has a BERT topic.
    if n_outliers > 0:
        print("Reducing outliers (embeddings strategy) ...")
        new_topics = topic_model.reduce_outliers(docs, topics,
                                                 strategy="embeddings",
                                                 embeddings=embeddings)
        topic_model.update_topics(docs, topics=new_topics,
                                  vectorizer_model=vectorizer)
        topics = new_topics

topics = list(topics)
final_topics = sorted(t for t in set(topics) if t != -1)
print(f"Final: method={used_method}, {len(final_topics)} topics, "
      f"{sum(1 for t in topics if t == -1)} remaining outliers.")

# ----------------------------------------------------------------------------
# 5. Per-issue dominant probability (best-effort; 1.0 for KMeans).
def dominant_prob(i, t):
    if t == -1:
        return 0.0
    try:
        if probs is not None and np.ndim(probs) == 2:
            return round(float(probs[i][t]), 4)
        if probs is not None and np.ndim(probs) == 1:
            return round(float(probs[i]), 4)
    except Exception:
        pass
    return 1.0

assign = pd.DataFrame({
    "issue_key": keys,
    "bert_topic": topics,
    "bert_prob": [dominant_prob(i, t) for i, t in enumerate(topics)],
})
assign = assign.merge(meta, on="issue_key", how="left")

# ----------------------------------------------------------------------------
# 6. Save, backing up the old 2-topic files first.
def backup(name):
    src = os.path.join(OUT, name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(OUT, name.replace(".csv", "_2topics_backup.csv")))

for f in ["bertopic_topic_info.csv", "bertopic_topics.csv",
          "bertopic_issue_assignments.csv"]:
    backup(f)

info = topic_model.get_topic_info()
info.to_csv(os.path.join(OUT, "bertopic_topic_info.csv"), index=False)

rows = []
for t in final_topics:
    for term, score in topic_model.get_topic(t)[:TOP_N_TERMS]:
        rows.append({"topic": t, "term": term, "score": round(float(score), 4)})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "bertopic_topics.csv"), index=False)

assign.to_csv(os.path.join(OUT, "bertopic_issue_assignments.csv"), index=False)

# ----------------------------------------------------------------------------
# 7. Console summary.
print("\n=== BERTopic re-run summary ===")
print(f"method: {used_method}")
counts = assign["bert_topic"].value_counts().sort_index()
for t in final_topics:
    terms = ", ".join(w for w, _ in topic_model.get_topic(t)[:5])
    print(f"  topic {t:2d}: {int(counts.get(t, 0)):3d} issues | {terms}")
print(f"outputs written to {OUT}")
