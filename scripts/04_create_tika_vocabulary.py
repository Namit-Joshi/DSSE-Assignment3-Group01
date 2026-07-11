import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "tika_preprocessed_issues.csv"

OUTPUT_VOCABULARY = BASE_DIR / "data" / "outputs" / "tika_vocabulary.csv"
OUTPUT_TOP_TOKENS = BASE_DIR / "data" / "outputs" / "tika_top_50_tokens.csv"
OUTPUT_SUMMARY = BASE_DIR / "data" / "outputs" / "tika_vocabulary_summary.txt"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} preprocessed Tika issues.")

    token_counter = Counter()
    document_frequency = defaultdict(int)

    for _, row in df.iterrows():
        issue_key = row["issue_key"]
        tokens_text = row.get("tokens", "")

        if pd.isna(tokens_text):
            tokens = []
        else:
            tokens = str(tokens_text).split()

        token_counter.update(tokens)

        unique_tokens_in_document = set(tokens)
        for token in unique_tokens_in_document:
            document_frequency[token] += 1

    vocabulary_rows = []

    for token, frequency in token_counter.items():
        vocabulary_rows.append({
            "token": token,
            "frequency": frequency,
            "document_frequency": document_frequency[token]
        })

    vocabulary_df = pd.DataFrame(vocabulary_rows)
    vocabulary_df = vocabulary_df.sort_values(
        by=["frequency", "document_frequency", "token"],
        ascending=[False, False, True]
    )

    OUTPUT_VOCABULARY.parent.mkdir(parents=True, exist_ok=True)

    vocabulary_df.to_csv(OUTPUT_VOCABULARY, index=False, encoding="utf-8")

    top_50_df = vocabulary_df.head(50)
    top_50_df.to_csv(OUTPUT_TOP_TOKENS, index=False, encoding="utf-8")

    total_tokens = sum(token_counter.values())
    unique_tokens = len(token_counter)

    summary_lines = []
    summary_lines.append("Assignment 3 - Week 1 Vocabulary Summary")
    summary_lines.append("Project: Apache Tika")
    summary_lines.append("")
    summary_lines.append(f"Number of issues: {len(df)}")
    summary_lines.append(f"Total number of tokens: {total_tokens}")
    summary_lines.append(f"Number of unique tokens: {unique_tokens}")
    summary_lines.append("")
    summary_lines.append("Top 50 tokens:")
    summary_lines.append("")

    for _, row in top_50_df.iterrows():
        summary_lines.append(
            f"{row['token']}: frequency={row['frequency']}, document_frequency={row['document_frequency']}"
        )

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("\nVocabulary creation complete.")
    print(f"Saved full vocabulary to: {OUTPUT_VOCABULARY}")
    print(f"Saved top 50 tokens to: {OUTPUT_TOP_TOKENS}")
    print(f"Saved vocabulary summary to: {OUTPUT_SUMMARY}")

    print("\nVocabulary summary:")
    print(f"Number of issues: {len(df)}")
    print(f"Total number of tokens: {total_tokens}")
    print(f"Number of unique tokens: {unique_tokens}")

    print("\nTop 30 tokens:")
    print(top_50_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
