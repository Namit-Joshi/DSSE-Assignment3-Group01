import pandas as pd
from pathlib import Path
from jira import JIRA
from tqdm import tqdm
import json

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "processed" / "tika_issue_list.csv"
OUTPUT_CSV = BASE_DIR / "data" / "raw" / "tika_jira_issues_raw.csv"
OUTPUT_JSON = BASE_DIR / "data" / "raw" / "tika_jira_issues_raw.json"

APACHE_JIRA_SERVER = "https://issues.apache.org/jira"


def safe_getattr(obj, attr, default=None):
    return getattr(obj, attr, default) if obj is not None else default


def get_name(obj):
    return safe_getattr(obj, "name", None)


def get_key(obj):
    return safe_getattr(obj, "key", None)


def extract_issue_data(issue, labels_row):
    fields = issue.fields

    comments = safe_getattr(fields, "comment", None)
    attachments = safe_getattr(fields, "attachment", [])
    components = safe_getattr(fields, "components", [])
    labels = safe_getattr(fields, "labels", [])

    votes = safe_getattr(fields, "votes", None)
    watches = safe_getattr(fields, "watches", None)
    parent = safe_getattr(fields, "parent", None)

    issue_data = {
        "issue_key": issue.key,

        "existence": labels_row["existence"],
        "property": labels_row["property"],
        "executive": labels_row["executive"],

        # Main text fields for topic modeling
        "summary": safe_getattr(fields, "summary", ""),
        "description": safe_getattr(fields, "description", ""),

        # Jira metadata fields for our analysis
        "issue_type": get_name(safe_getattr(fields, "issuetype", None)),
        "status": get_name(safe_getattr(fields, "status", None)),
        "resolution": get_name(safe_getattr(fields, "resolution", None)),
        "priority": get_name(safe_getattr(fields, "priority", None)),

        "labels": "; ".join(labels) if labels else "",
        "components": "; ".join([component.name for component in components]) if components else "",

        "created": safe_getattr(fields, "created", None),
        "updated": safe_getattr(fields, "updated", None),
        "resolved": safe_getattr(fields, "resolutiondate", None),

        "votes": safe_getattr(votes, "votes", None),
        "watchers": safe_getattr(watches, "watchCount", None),

        "parent_key": get_key(parent),
        "comment_count": safe_getattr(comments, "total", 0),
        "attachment_count": len(attachments) if attachments else 0,
    }

    return issue_data


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    issue_list = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(issue_list)} Tika issues.")
    print("Connecting to Apache Jira...")

    jira = JIRA(server=APACHE_JIRA_SERVER)

    all_issues = []

    fields_to_download = [
        "summary",
        "description",
        "issuetype",
        "status",
        "resolution",
        "priority",
        "labels",
        "components",
        "created",
        "updated",
        "resolutiondate",
        "votes",
        "watches",
        "parent",
        "comment",
        "attachment"
    ]

    for _, row in tqdm(issue_list.iterrows(), total=len(issue_list), desc="Downloading issues"):
        issue_key = row["issue_key"]

        try:
            issue = jira.issue(issue_key, fields=",".join(fields_to_download))
            issue_data = extract_issue_data(issue, row)
            all_issues.append(issue_data)

        except Exception as e:
            print(f"Failed to download {issue_key}: {e}")

            all_issues.append({
                "issue_key": issue_key,
                "existence": row["existence"],
                "property": row["property"],
                "executive": row["executive"],
                "summary": "",
                "description": "",
                "issue_type": "",
                "status": "",
                "resolution": "",
                "priority": "",
                "labels": "",
                "components": "",
                "created": "",
                "updated": "",
                "resolved": "",
                "votes": "",
                "watchers": "",
                "parent_key": "",
                "comment_count": "",
                "attachment_count": "",
                "download_error": str(e)
            })

    output_df = pd.DataFrame(all_issues)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_issues, f, indent=2, ensure_ascii=False)

    print("\nDownload complete.")
    print(f"Saved CSV file to: {OUTPUT_CSV}")
    print(f"Saved JSON file to: {OUTPUT_JSON}")

    print("\nFirst five downloaded rows:")
    print(output_df.head())


if __name__ == "__main__":
    main()
