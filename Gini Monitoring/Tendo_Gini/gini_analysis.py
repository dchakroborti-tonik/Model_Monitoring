import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def calculate_gini(y_true, y_pred):
    """
    Calculate Gini coefficient from actual values and predicted scores.
    Gini = 2 * AUC - 1
    """
    auc = roc_auc_score(y_true, y_pred)
    gini = 2 * auc - 1
    return gini


def calculate_bad_metrics(df, target_col, maturity_flag_col):
    """
    Calculate count of bad and percentage of bad.
    Bad = target / target_maturity_flag
    """
    count_bad = (df[target_col] / df[maturity_flag_col]).sum()
    total_count = len(df)
    pct_bad = (count_bad / total_count * 100) if total_count > 0 else 0
    return count_bad, pct_bad


def combined_gini(df, score_cols, target_col):
    """
    Calculate combined Gini for multiple score columns.
    Uses average of normalized scores as combined score.
    """
    # Normalize scores to [0, 1] range
    normalized_scores = pd.DataFrame()
    for col in score_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val > min_val:
            normalized_scores[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            normalized_scores[col] = 0

    # Combined score is average of normalized scores
    combined_score = normalized_scores.mean(axis=1)
    combined_gini_val = calculate_gini(df[target_col], combined_score)

    return combined_gini_val


# Load your data
df = pd.read_csv("your_data.csv")  # Replace with your actual data file

print("=" * 90)
print("OVERALL GINI COEFFICIENT ANALYSIS (NO GROUPING)")
print("=" * 90)

# Dataset Information
print(f"\nDATASET INFORMATION:")
print(f"  Total Records: {len(df)}")
print(f"  Target Value Counts:\n{df['target'].value_counts()}")

# Overall Bad Metrics
overall_count_bad, overall_pct_bad = calculate_bad_metrics(
    df, "target", "target_maturity_flag"
)
print(f"\nOVERALL BAD METRICS:")
print(f"  Count of Bad: {overall_count_bad:.4f}")
print(f"  Percentage of Bad: {overall_pct_bad:.4f}%")

# Individual Score Gini
print(f"\nINDIVIDUAL SCORE GINI:")
print("-" * 90)
overall_gini_oop = calculate_gini(df["target"], df["oop_score"])
overall_gini_b2oop = calculate_gini(df["target"], df["b2oopscore"])

print(f"  OOP Score:")
print(f"    Gini: {overall_gini_oop:.6f}")
print(f"    AUC:  {(overall_gini_oop + 1) / 2:.6f}")

print(f"\n  B2OOP Score:")
print(f"    Gini: {overall_gini_b2oop:.6f}")
print(f"    AUC:  {(overall_gini_b2oop + 1) / 2:.6f}")

# Combined Gini
print(f"\nCOMBINED GINI:")
print("-" * 90)
overall_combined_gini = combined_gini(df, ["oop_score", "b2oopscore"], "target")

print(f"  Combined Score (Mean Normalized):")
print(f"    Gini: {overall_combined_gini:.6f}")
print(f"    AUC:  {(overall_combined_gini + 1) / 2:.6f}")

# Score Statistics
print(f"\nSCORE STATISTICS:")
print("-" * 90)
print(f"\n  OOP Score:")
print(f"    Min:  {df['oop_score'].min():.6f}")
print(f"    Max:  {df['oop_score'].max():.6f}")
print(f"    Mean: {df['oop_score'].mean():.6f}")
print(f"    Std:  {df['oop_score'].std():.6f}")

print(f"\n  B2OOP Score:")
print(f"    Min:  {df['b2oopscore'].min():.6f}")
print(f"    Max:  {df['b2oopscore'].max():.6f}")
print(f"    Mean: {df['b2oopscore'].mean():.6f}")
print(f"    Std:  {df['b2oopscore'].std():.6f}")

# Create Summary DataFrame
summary_overall = pd.DataFrame(
    {
        "Metric": [
            "OOP Score Gini",
            "OOP Score AUC",
            "B2OOP Score Gini",
            "B2OOP Score AUC",
            "Combined Gini",
            "Combined AUC",
            "Count of Bad",
            "Percentage of Bad",
            "Total Records",
            "Bad Count (target=1)",
            "Good Count (target=0)",
        ],
        "Value": [
            overall_gini_oop,
            (overall_gini_oop + 1) / 2,
            overall_gini_b2oop,
            (overall_gini_b2oop + 1) / 2,
            overall_combined_gini,
            (overall_combined_gini + 1) / 2,
            overall_count_bad,
            overall_pct_bad,
            len(df),
            (df["target"] == 1).sum(),
            (df["target"] == 0).sum(),
        ],
    }
)

print(f"\nSUMMARY TABLE:")
print("-" * 90)
print(summary_overall.to_string(index=False))

# Export to Excel
with pd.ExcelWriter("gini_overall_analysis.xlsx", engine="openpyxl") as writer:
    summary_overall.to_excel(writer, sheet_name="Overall Summary", index=False)

    # Detailed score statistics
    score_stats = pd.DataFrame(
        {
            "Score Column": ["OOP Score", "B2OOP Score"],
            "Min": [df["oop_score"].min(), df["b2oopscore"].min()],
            "Max": [df["oop_score"].max(), df["b2oopscore"].max()],
            "Mean": [df["oop_score"].mean(), df["b2oopscore"].mean()],
            "Median": [df["oop_score"].median(), df["b2oopscore"].median()],
            "Std Dev": [df["oop_score"].std(), df["b2oopscore"].std()],
            "Gini": [overall_gini_oop, overall_gini_b2oop],
            "AUC": [(overall_gini_oop + 1) / 2, (overall_gini_b2oop + 1) / 2],
        }
    )
    score_stats.to_excel(writer, sheet_name="Score Statistics", index=False)

print(f"\n✓ Results exported to 'gini_overall_analysis.xlsx'")
