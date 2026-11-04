import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer

def cosine_rows(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(a * b, axis=1) / (
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument(
        "--model",
        default="sentence-transformers/bert-base-nli-mean-tokens",
        help="Sentence Transformers model name/path",
    )
    parser.add_argument(
        "--output",
        default="controlled_sbert_results.csv",
        help="CSV containing human scores and SBERT predictions",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    required = {"id", "sentence1", "sentence2", "category", "human_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    scored = df[df["human_score"].notna()].copy()
    scored["human_score"] = pd.to_numeric(scored["human_score"], errors="raise")

    if len(scored) < 2:
        raise ValueError("At least two human-annotated rows are required.")

    print(f"Model: {args.model}")
    print(f"Total dataset rows: {len(df)}")
    print(f"Human-annotated rows used: {len(scored)}")
    print("\nAnnotated rows by category:")
    print(scored["category"].value_counts().sort_index().to_string())

    print("\nLoading SBERT model...")
    model = SentenceTransformer(args.model)

    print("Encoding sentence 1...")
    emb1 = model.encode(
        scored["sentence1"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print("Encoding sentence 2...")
    emb2 = model.encode(
        scored["sentence2"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    scored["sbert_cosine"] = cosine_rows(emb1, emb2)

    overall = spearmanr(scored["human_score"], scored["sbert_cosine"])
    print("\n=== OVERALL RESULT ===")
    print(f"N = {len(scored)}")
    print(f"Spearman rho = {overall.statistic:.6f}")
    print(f"Spearman x100 = {overall.statistic * 100:.3f}")
    print(f"p-value = {overall.pvalue:.6g}")

    print("\n=== CATEGORY RESULTS ===")
    category_rows = []
    for category, group in scored.groupby("category", sort=True):
        # Spearman is undefined if the human scores have no variation.
        human_unique = group["human_score"].nunique()
        model_unique = group["sbert_cosine"].nunique()

        if len(group) >= 2 and human_unique >= 2 and model_unique >= 2:
            r = spearmanr(group["human_score"], group["sbert_cosine"])
            rho = r.statistic
            p = r.pvalue
            rho_text = f"{rho:.6f}"
        else:
            rho = np.nan
            p = np.nan
            rho_text = "N/A (constant human scores)"

        category_rows.append({
            "category": category,
            "n": len(group),
            "human_mean": group["human_score"].mean(),
            "sbert_mean_cosine": group["sbert_cosine"].mean(),
            "spearman_rho": rho,
            "p_value": p,
        })

        print(
            f"{category:20s} N={len(group):2d} | "
            f"human mean={group['human_score'].mean():.3f} | "
            f"SBERT mean={group['sbert_cosine'].mean():.3f} | "
            f"rho={rho_text}"
        )

    category_df = pd.DataFrame(category_rows)

    # A simple discrepancy measure after mapping cosine [-1,1] to [0,5].
    scored["sbert_mapped_0_5"] = ((scored["sbert_cosine"] + 1) / 2) * 5
    scored["absolute_gap"] = (
        scored["human_score"] - scored["sbert_mapped_0_5"]
    ).abs()

    print("\n=== LARGEST HUMAN–SBERT DISAGREEMENTS ===")
    cols = [
        "id", "category", "sentence1", "sentence2",
        "human_score", "sbert_cosine", "absolute_gap"
    ]
    print(
        scored.sort_values("absolute_gap", ascending=False)[cols]
        .head(10)
        .to_string(index=False)
    )

    scored.to_csv(args.output, index=False)
    category_output = args.output.rsplit(".", 1)[0] + "_by_category.csv"
    category_df.to_csv(category_output, index=False)

    print(f"\nSaved pair-level results to: {args.output}")
    print(f"Saved category summary to: {category_output}")
    print(
        "\nNote: category-level Spearman is undefined when all human scores "
        "inside that category are identical. Mean SBERT cosine is still reported."
    )

if __name__ == "__main__":
    main()
