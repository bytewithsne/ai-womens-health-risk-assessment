import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# VALIDATE PERSONALIZED RISK PATTERN INDEX
# ============================================================

INPUT_FILE = Path("data/processed/risk_pattern_index.csv")
OUTPUT_FILE = Path("data/processed/risk_pattern_validation.csv")

print("=" * 70)
print("VALIDATING PERSONALIZED RISK PATTERN INDEX")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows: {len(df)}")
print(f"Participants: {df['User id'].nunique()}")

# ------------------------------------------------------------
# 2. Basic distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. INDEX DISTRIBUTION")
print("=" * 70)

index_col = "personalized_risk_pattern_index"

print(df[index_col].describe())

print("\nCategory counts:")
print(df["risk_pattern_category"].value_counts(dropna=False))

# ------------------------------------------------------------
# 3. Check whether index changes across cycles
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. CYCLE-TO-CYCLE BEHAVIOR")
print("=" * 70)

df = df.sort_values(["User id", "Cycle number"])

df["index_change_from_previous_cycle"] = (
    df.groupby("User id")[index_col].diff()
)

change_stats = df["index_change_from_previous_cycle"].describe()

print(change_stats)

# Absolute change is easier to interpret
df["absolute_index_change"] = (
    df["index_change_from_previous_cycle"].abs()
)

print("\nAbsolute cycle-to-cycle change:")
print(df["absolute_index_change"].describe())

# ------------------------------------------------------------
# 4. Participant-level stability
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. PARTICIPANT-LEVEL STABILITY")
print("=" * 70)

participant_stats = (
    df.groupby("User id")[index_col]
    .agg(
        cycles="count",
        mean_index="mean",
        std_index="std",
        min_index="min",
        max_index="max"
    )
    .reset_index()
)

participant_stats["range"] = (
    participant_stats["max_index"]
    - participant_stats["min_index"]
)

print("\nParticipant-level index statistics:")
print(participant_stats.to_string(index=False))

print("\nAverage participant standard deviation:")
print(participant_stats["std_index"].mean())

print("\nAverage participant index range:")
print(participant_stats["range"].mean())

# ------------------------------------------------------------
# 5. Relationship with current burden
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. INDEX VS CURRENT BURDEN")
print("=" * 70)

current_col = "current_burden_score"

valid_current = df[[index_col, current_col]].dropna()

if len(valid_current) >= 3:

    correlation = valid_current[index_col].corr(
        valid_current[current_col]
    )

    print(
        f"Correlation between index and current burden: "
        f"{correlation:.4f}"
    )

else:
    print("Not enough observations for correlation.")

# ------------------------------------------------------------
# 6. Relationship with historical deviation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. INDEX VS HISTORICAL DEVIATION")
print("=" * 70)

deviation_col = "historical_deviation_score"

valid_deviation = df[[index_col, deviation_col]].dropna()

if len(valid_deviation) >= 3:

    correlation = valid_deviation[index_col].corr(
        valid_deviation[deviation_col]
    )

    print(
        f"Correlation between index and historical deviation: "
        f"{correlation:.4f}"
    )

else:
    print("Not enough observations for correlation.")

# ------------------------------------------------------------
# 7. Evidence coverage analysis
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. EVIDENCE COVERAGE")
print("=" * 70)

coverage_col = "evidence_coverage_percent"

print(df[coverage_col].describe())

print(
    "\nCorrelation between evidence coverage and index:"
)

valid_coverage = df[[index_col, coverage_col]].dropna()

if len(valid_coverage) >= 3:

    correlation = valid_coverage[index_col].corr(
        valid_coverage[coverage_col]
    )

    print(f"{correlation:.4f}")

# ------------------------------------------------------------
# 8. Category vs burden
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. CATEGORY BEHAVIOR")
print("=" * 70)

category_summary = (
    df.groupby("risk_pattern_category", observed=True)
    .agg(
        cycles=(index_col, "count"),
        mean_index=(index_col, "mean"),
        mean_current_burden=(current_col, "mean"),
        mean_historical_deviation=(deviation_col, "mean"),
        mean_coverage=(coverage_col, "mean")
    )
    .reset_index()
)

print(category_summary.to_string(index=False))

# ------------------------------------------------------------
# 9. Save validation dataset
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(f"\nSaved:")
print(OUTPUT_FILE)

print(
    "\nIMPORTANT:"
    "\n- This validation is exploratory."
    "\n- It does not establish clinical validity."
    "\n- The index is not a diagnosis."
    "\n- We will use these results to decide whether the"
    "\n  methodology needs modification."
)