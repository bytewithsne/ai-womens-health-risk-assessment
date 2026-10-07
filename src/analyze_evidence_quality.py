import pandas as pd
import numpy as np
from pathlib import Path

INDEX_FILE = Path("data/processed/risk_pattern_index.csv")
SENSITIVITY_FILE = Path("data/processed/sensitivity_extreme_cases.csv")
OUTPUT_FILE = Path("data/processed/evidence_quality_analysis.csv")

print("=" * 70)
print("EVIDENCE-QUALITY ANALYSIS")
print("=" * 70)

df = pd.read_csv(INDEX_FILE)

# Evidence coverage already calculated by the PSPI pipeline
coverage_col = "evidence_coverage_percent"

# ------------------------------------------------------------
# Categorize evidence quality
# ------------------------------------------------------------

def evidence_group(x):
    if pd.isna(x):
        return "Unknown"
    elif x < 30:
        return "Very low (<30%)"
    elif x < 50:
        return "Low (30-49%)"
    elif x < 70:
        return "Moderate (50-69%)"
    else:
        return "High (>=70%)"


df["evidence_quality"] = df[coverage_col].apply(evidence_group)

# ------------------------------------------------------------
# Basic distribution
# ------------------------------------------------------------

print()
print("=" * 70)
print("EVIDENCE QUALITY DISTRIBUTION")
print("=" * 70)

print(
    df["evidence_quality"]
    .value_counts()
    .to_string()
)

# ------------------------------------------------------------
# PSPI statistics by evidence group
# ------------------------------------------------------------

print()
print("=" * 70)
print("PSPI BY EVIDENCE QUALITY")
print("=" * 70)

summary = (
    df.groupby("evidence_quality", observed=True)
    .agg(
        cycles=("personalized_risk_pattern_index", "count"),
        mean_index=("personalized_risk_pattern_index", "mean"),
        median_index=("personalized_risk_pattern_index", "median"),
        std_index=("personalized_risk_pattern_index", "std"),
        minimum=("personalized_risk_pattern_index", "min"),
        maximum=("personalized_risk_pattern_index", "max"),
        mean_coverage=(coverage_col, "mean")
    )
    .round(3)
)
print(summary.to_string())

# ------------------------------------------------------------
# Sensitivity information
# ------------------------------------------------------------

sensitivity = pd.read_csv(SENSITIVITY_FILE)

# Merge sensitivity information
keys = ["User id", "Cycle number"]

if all(k in sensitivity.columns for k in keys):

    sensitivity_cols = [
        "User id",
        "Cycle number",
        "largest_sensitivity_change",
        "most_influential_feature"
    ]

    df = df.merge(
        sensitivity[sensitivity_cols],
        on=keys,
        how="left"
    )

    print()
    print("=" * 70)
    print("SENSITIVITY BY EVIDENCE QUALITY")
    print("=" * 70)

    sensitivity_summary = (
        df.groupby("evidence_quality", observed=True)
        ["largest_sensitivity_change"]
        .agg(
            mean_change="mean",
            median_change="median",
            maximum_change="max"
        )
        .round(3)
    )

    print(sensitivity_summary.to_string())

    # --------------------------------------------------------
    # Count extreme cases
    # --------------------------------------------------------

    df["extreme_sensitivity"] = (
        df["largest_sensitivity_change"] >= 15
    )

    print()
    print("=" * 70)
    print("EXTREME SENSITIVITY CASES BY EVIDENCE QUALITY")
    print("=" * 70)

    extreme_summary = (
        df.groupby("evidence_quality", observed=True)
        ["extreme_sensitivity"]
        .agg(
            total_cycles="count",
            extreme_cases="sum"
        )
    )

    extreme_summary["extreme_percentage"] = (
        extreme_summary["extreme_cases"]
        / extreme_summary["total_cycles"]
        * 100
    )

    print(
        extreme_summary
        .round(2)
        .to_string()
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)