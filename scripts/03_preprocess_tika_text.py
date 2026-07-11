import pandas as pd
from pathlib import Path
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "raw" / "tika_jira_issues_raw.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "tika_preprocessed_issues.csv"


def download_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]

    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)


def split_camel_case(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    text = split_camel_case(text)

    text = text.lower()

    # Remove Jira/code-like markup symbols but keep words
    text = re.sub(r"[^a-z\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text, stop_words, lemmatizer):
    cleaned = clean_text(text)

    tokens = cleaned.split()

    processed_tokens = []
    for token in tokens:
        if len(token) <= 2:
            continue

        if token in stop_words:
            continue

        lemma = lemmatizer.lemmatize(token)

        if lemma and lemma not in stop_words and len(lemma) > 2:
            processed_tokens.append(lemma)

    return processed_tokens


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    download_nltk_resources()

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} downloaded Tika issues.")

    stop_words = set(stopwords.words("english"))

    # Extra stop words common in Jira/project text.
    custom_stop_words = {
        "tika",
        "apache",
        "jira",
        "issue",
        "issues",
        "org",
        "http",
        "https",
        "www",
        "com",
        "would",
        "could",
        "should",
        "please",
        "thanks",
        "thank",
        "also",
        "one",
        "two",
        "new",
        "use",
        "using",
        "used",
        "get",
        "set",
        "make",
        "need",
        "want",
        "like"
    }

    stop_words.update(custom_stop_words)

    lemmatizer = WordNetLemmatizer()

    # Combine summary and description as required by the assignment
    df["summary"] = df["summary"].fillna("")
    df["description"] = df["description"].fillna("")
    df["combined_text"] = df["summary"] + " " + df["description"]

    df["tokens"] = df["combined_text"].apply(
        lambda text: preprocess_text(text, stop_words, lemmatizer)
    )

    df["preprocessed_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))
    df["token_count"] = df["tokens"].apply(len)
    df["description_length"] = df["description"].apply(lambda text: len(str(text)))

    # Saving tokens as readable text, not Python lists
    output_df = df.copy()
    output_df["tokens"] = output_df["tokens"].apply(lambda tokens: " ".join(tokens))

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    all_tokens = []
    for tokens in df["tokens"]:
        all_tokens.extend(tokens)

    token_counter = Counter(all_tokens)

    print("\nPreprocessing complete.")
    print(f"Saved preprocessed file to: {OUTPUT_FILE}")

    print("\nSummary:")
    print(f"Number of issues: {len(df)}")
    print(f"Issues with zero tokens after preprocessing: {(df['token_count'] == 0).sum()}")
    print(f"Average token count per issue: {df['token_count'].mean():.2f}")

    print("\nTop 30 tokens after preprocessing:")
    for token, count in token_counter.most_common(30):
        print(f"{token}: {count}")

    print("\nFirst five processed rows:")
    print(output_df[["issue_key", "summary", "token_count", "preprocessed_text"]].head())


if __name__ == "__main__":
    main()
