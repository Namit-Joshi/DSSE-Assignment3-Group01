import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "data" / "raw" / "issues.xlsx"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "tika_issue_list.csv"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Read only the Tika sheet from the Excel file
    df = pd.read_excel(INPUT_FILE, sheet_name="Tika")

    print("Columns found in Excel:")
    print(df.columns.tolist())

    print("\nNumber of Tika issues:")
    print(len(df))

    print("\nFirst five rows:")
    print(df.head())

    df = df.rename(columns={
        "Issue ID": "issue_key",
        "Types of design decisions": "design_decisions"
    })

    # Split the three Boolean design-decision values
    decisions = df["design_decisions"].astype(str).str.split(expand=True)

    df["existence"] = decisions[0].map({"True": True, "False": False})
    df["property"] = decisions[1].map({"True": True, "False": False})
    df["executive"] = decisions[2].map({"True": True, "False": False})

    # Keeping only the columns needed for our analysis
    df = df[["issue_key", "existence", "property", "executive"]]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaned Tika issue list saved to:")
    print(OUTPUT_FILE)

    print("\nFirst five cleaned rows:")
    print(df.head())


if __name__ == "__main__":
    main()
