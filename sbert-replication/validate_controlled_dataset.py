"""Validate the controlled SBERT semantic-similarity dataset."""
import argparse
import pandas as pd

REQUIRED = ["id", "sentence1", "sentence2", "category", "human_score", "annotator"]
EXPECTED_CATEGORIES = {
    "paraphrase", "related", "negation", "argument_reversal",
    "numerical_change", "entity_change", "unrelated"
}

def main(path):
    df = pd.read_csv(path)
    problems = []

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns: {missing_cols}")

    if df["id"].duplicated().any():
        problems.append(f"Duplicate IDs: {df.loc[df['id'].duplicated(), 'id'].tolist()}")

    blank_text = df["sentence1"].isna() | df["sentence2"].isna()
    if blank_text.any():
        problems.append(f"Rows with missing sentence text: {df.loc[blank_text, 'id'].tolist()}")

    exact_dup = df.duplicated(subset=["sentence1", "sentence2"], keep=False)
    if exact_dup.any():
        problems.append(f"Duplicate sentence pairs found in IDs: {df.loc[exact_dup, 'id'].tolist()}")

    unexpected = sorted(set(df["category"].dropna()) - EXPECTED_CATEGORIES)
    if unexpected:
        problems.append(f"Unexpected categories: {unexpected}")

    scores = pd.to_numeric(df["human_score"], errors="coerce")
    filled = df["human_score"].notna() & (df["human_score"].astype(str).str.strip() != "")
    invalid_scores = filled & (scores.isna() | ~scores.between(0, 5) | (scores % 1 != 0))
    if invalid_scores.any():
        problems.append(f"Invalid human scores in IDs: {df.loc[invalid_scores, 'id'].tolist()}")

    print(f"Rows: {len(df)}")
    print("\nCategory counts:")
    print(df["category"].value_counts().sort_index().to_string())
    print(f"\nAnnotated rows: {filled.sum()} / {len(df)}")
    print(f"Rows with annotator ID: {df['annotator'].notna().sum()} / {len(df)}")

    if problems:
        print("\nVALIDATION WARNINGS")
        for p in problems:
            print("- " + p)
        raise SystemExit(1)
    print("\nValidation passed: structure, categories, duplicates, and entered scores are valid.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", nargs="?", default="sbert_controlled_semantic_dataset.csv")
    args = parser.parse_args()
    main(args.csv_path)

