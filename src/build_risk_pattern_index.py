import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# BUILD PERSONALIZED SYMPTOM-PATTERN INDEX
# ============================================================
#
# Research prototype only.
# NOT a diagnosis.
# NOT a clinically validated risk score.
#
# Design:
#   - Current symptom burden
#   - Personalized change from previous cycles
#   - Contextual variables are NOT treated as burden
#   - Missing observations remain missing
#   - Final index is bounded to 0-100
# ============================================================

INPUT_FILE = Path("data/processed/temporal_twin_features.csv")
OUTPUT_FILE = Path("data/processed/risk_pattern_index.csv")


# ------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------

# These variables are used to estimate current symptom burden.
CURRENT_BURDEN_FEATURES = [
    "bleeding_mean",
    "general_emotional_condition_mean",
    "general_physical_condition_mean",
    "type_of_stool_mean",
    "lower_back_pain_mean",
    "headache_mean",
]

# These variables represent change relative to the participant's
# own historical baseline.
DEVIATION_FEATURES = [
    "bleeding_historical_deviation",
    "general_emotional_condition_historical_deviation",
    "general_physical_condition_historical_deviation",
    "type_of_stool_historical_deviation",
    "ovulation_test_values_historical_deviation",
]

# Contextual variables.
# They are deliberately NOT included in the burden score.
CONTEXTUAL_FEATURES = [
    "ovulation_test_values_mean",
    "appetite_no_appetite_mean",
]


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def minmax_series(series):
    """
    Normalize a feature to 0-1 using the observed dataset range.

    Missing values remain missing.
    """
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)

    if pd.isna(minimum) or pd.isna(maximum):
        return pd.Series(np.nan, index=series.index)

    if maximum == minimum:
        return pd.Series(
            np.where(series.notna(), 0.0, np.nan),
            index=series.index
        )

    return (series - minimum) / (maximum - minimum)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("=" * 70)
print("BUILDING PERSONALIZED SYMPTOM-PATTERN INDEX")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"\nInput rows: {len(df)}")
print(f"Participants: {df['User id'].nunique()}")


# ------------------------------------------------------------
# Current burden
# ------------------------------------------------------------

print("\nCalculating current symptom burden...")

current_components = []

for feature in CURRENT_BURDEN_FEATURES:

    if feature not in df.columns:
        print(f"WARNING: Missing feature: {feature}")
        continue

    normalized = minmax_series(df[feature])

    # Higher value = greater burden
    if feature in [
        "bleeding_mean",
        "type_of_stool_mean",
        "lower_back_pain_mean",
        "headache_mean",
    ]:
        burden = normalized

    # Higher value = better condition,
    # therefore reverse it for burden.
    elif feature in [
        "general_emotional_condition_mean",
        "general_physical_condition_mean",
    ]:
        burden = 1 - normalized

    else:
        burden = normalized

    current_components.append(burden.rename(feature))


if current_components:

    current_component_df = pd.concat(
        current_components,
        axis=1
    )

    df["current_burden_score"] = (
        current_component_df.mean(axis=1, skipna=True)
    )

    df["current_burden_observed_features"] = (
        current_component_df.notna().sum(axis=1)
    )

else:

    df["current_burden_score"] = np.nan
    df["current_burden_observed_features"] = 0


# ------------------------------------------------------------
# Historical deviation
# ------------------------------------------------------------

print("Calculating personalized historical deviation...")

deviation_components = []

for feature in DEVIATION_FEATURES:

    if feature not in df.columns:
        print(f"WARNING: Missing feature: {feature}")
        continue

    # Absolute deviation means:
    # larger change from personal baseline = stronger change signal.
    absolute_deviation = df[feature].abs()

    # Bound each deviation feature independently to 0-1.
    normalized_deviation = minmax_series(
        absolute_deviation
    )

    deviation_components.append(
        normalized_deviation.rename(feature)
    )


if deviation_components:

    deviation_df = pd.concat(
        deviation_components,
        axis=1
    )

    df["historical_deviation_score"] = (
        deviation_df.mean(axis=1, skipna=True)
    )

    df["historical_deviation_observed_features"] = (
        deviation_df.notna().sum(axis=1)
    )

else:

    df["historical_deviation_score"] = np.nan
    df["historical_deviation_observed_features"] = 0


# ------------------------------------------------------------
# Combine components
# ------------------------------------------------------------

print("Combining current burden and personalized change...")

current = df["current_burden_score"]
historical = df["historical_deviation_score"]

df["personalized_risk_pattern_index"] = np.where(
    current.notna() & historical.notna(),
    (current * 0.50 + historical * 0.50) * 100,

    np.where(
        current.notna(),
        current * 100,

        np.where(
            historical.notna(),
            historical * 100,
            np.nan
        )
    )
)


# Explicit numerical safety bound.
df["personalized_risk_pattern_index"] = (
    df["personalized_risk_pattern_index"]
    .clip(0, 100)
)


# ------------------------------------------------------------
# Evidence coverage
# ------------------------------------------------------------

# IMPORTANT:
# Only 11 variables contribute to the evidence score:
#
# 6 current burden features
# + 5 historical deviation features
#
# Ovulation mean and appetite mean are contextual only.

TOTAL_EVIDENCE_FEATURES = (
    len(CURRENT_BURDEN_FEATURES)
    + len(DEVIATION_FEATURES)
)

df["evidence_observed_features"] = (
    df["current_burden_observed_features"]
    + df["historical_deviation_observed_features"]
)

df["evidence_coverage_percent"] = (
    df["evidence_observed_features"]
    / TOTAL_EVIDENCE_FEATURES
    * 100
)


# ------------------------------------------------------------
# Prototype categories
# ------------------------------------------------------------

def categorize(score):

    if pd.isna(score):
        return "Insufficient evidence"

    if score < 33:
        return "Lower pattern"

    elif score < 66:
        return "Intermediate pattern"

    else:
        return "Higher pattern"


df["risk_pattern_category"] = (
    df["personalized_risk_pattern_index"]
    .apply(categorize)
)


# ------------------------------------------------------------
# Select output columns
# ------------------------------------------------------------

output_columns = [
    "User id",
    "Cycle number",

    "current_burden_score",
    "historical_deviation_score",

    "personalized_risk_pattern_index",

    "current_burden_observed_features",
    "historical_deviation_observed_features",

    "evidence_observed_features",
    "evidence_coverage_percent",

    "risk_pattern_category",
]

output = df[output_columns].copy()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RISK PATTERN INDEX COMPLETE")
print("=" * 70)

print(f"Output rows: {len(output)}")
print(f"Output columns: {len(output.columns)}")

print("\nIndex range:")
print(
    f"Minimum: {output['personalized_risk_pattern_index'].min():.2f}"
)

print(
    f"Maximum: {output['personalized_risk_pattern_index'].max():.2f}"
)

print("\nRisk-pattern categories:")
print(
    output["risk_pattern_category"]
    .value_counts()
)

print("\nAverage evidence coverage:")
print(
    f"{output['evidence_coverage_percent'].mean():.2f}%"
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("- This is a research-derived prototype index.")
print("- It is NOT a diagnosis.")
print("- It is NOT a clinically validated risk score.")
print("- Contextual variables are not counted as burden evidence.")
print("- Missing observations are not treated as zero.")
print("- Historical deviation uses previous-cycle information.")
print("- The final index is bounded between 0 and 100.")