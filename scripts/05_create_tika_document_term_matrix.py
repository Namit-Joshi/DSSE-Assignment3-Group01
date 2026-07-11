import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "tika_preprocessed_issues.csv"

OUTPUT_MATRIX = BASE_DIR / "data" / "outputs" / "tika_document_term_matrix.csv"
OUTPUT_FEATURES = BASE_DIR / "data" / "outputs" / "tika_feature_names.csv"
OUTPUT_SUMMARY = BASE_DIR / "data" / "outputs" / "tika_document_term_matrix_summary.txt"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} preprocessed Tika issues.")

    df["preprocessed_text"] = df["preprocessed_text"].fillna("")

    vectorizer = CountVectorizer(
        tokenizer=str.split,
        lowercase=False,
        token_pattern=None
    )

    dtm = vectorizer.fit_transform(df["preprocessed_text"])

    feature_names = vectorizer.get_feature_names_out()

    dtm_df = pd.DataFrame(
        dtm.toarray(),
        columns=feature_names
    )

    # Add issue keys and design-decision labels.
    metadata_df = pd.DataFrame({
        "issue_key": df["issue_key"],
        "decision_existence": df["existence"],
        "decision_property": df["property"],
        "decision_executive": df["executive"]
    })

    final_df = pd.concat([metadata_df, dtm_df], axis=1)

    features_df = pd.DataFrame({
        "feature_index": range(len(feature_names)),
        "token": feature_names
    })

    OUTPUT_MATRIX.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(OUTPUT_MATRIX, index=False, encoding="utf-8")
    features_df.to_csv(OUTPUT_FEATURES, index=False, encoding="utf-8")

    summary_lines = []
    summary_lines.append("Assignment 3 - Week 1 Document-Term Matrix Summary")
    summary_lines.append("Project: Apache Tika")
    summary_lines.append("")
    summary_lines.append(f"Number of issues/documents: {dtm.shape[0]}")
    summary_lines.append(f"Number of terms/features: {dtm.shape[1]}")
    summary_lines.append(f"Matrix shape without metadata columns: {dtm.shape[0]} x {dtm.shape[1]}")
    summary_lines.append(f"Matrix shape with metadata columns: {final_df.shape[0]} x {final_df.shape[1]}")
    summary_lines.append("")
    summary_lines.append("Explanation:")
    summary_lines.append("Each row represents one Tika Jira issue.")
    summary_lines.append("Each token column represents one token from the vocabulary.")
    summary_lines.append("Each token cell contains the frequency of that token in that issue.")
    summary_lines.append("The first four columns contain metadata: issue_key, decision_existence, decision_property, and decision_executive.")
    summary_lines.append("")
    summary_lines.append("This document-term matrix will be used as input for LDA topic modeling in Week 2.")

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\nDocument-term matrix creation complete.")
    print(f"Saved document-term matrix to: {OUTPUT_MATRIX}")
    print(f"Saved feature names to: {OUTPUT_FEATURES}")
    print(f"Saved summary to: {OUTPUT_SUMMARY}")

    print("\nMatrix summary:")
    print(f"Number of issues/documents: {dtm.shape[0]}")
    print(f"Number of terms/features: {dtm.shape[1]}")
    print(f"Matrix shape without metadata columns: {dtm.shape[0]} x {dtm.shape[1]}")
    print(f"Matrix shape with metadata columns: {final_df.shape[0]} x {final_df.shape[1]}")

    print("\nFirst five columns/features:")
    print(features_df.head())

    print("\nFirst five rows of matrix:")
    print(final_df.iloc[:5, :10])


if __name__ == "__main__":
    main()
