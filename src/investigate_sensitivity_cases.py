import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path("data/processed/temporal_twin_features.csv")
INDEX_FILE = Path("data/processed/risk_pattern_index.csv")
OUTPUT_FILE = Path("data/processed/sensitivity_extreme_cases.csv")

CURRENT_FEATURES = [
    "bleeding_mean",
    "general_emotional_condition_mean",
    "general_physical_condition_mean",
    "type_of_stool_mean",
    "lower_back_pain_mean",
    "headache_mean",
]

DEVIATION_FEATURES = [
    "bleeding_historical_deviation",
    "general_emotional_condition_historical_deviation",
    "general_physical_condition_historical_deviation",
    "type_of_stool_historical_deviation",
    "ovulation_test_values_historical_deviation",
]


def minmax_series(series):
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)

    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(np.nan, index=series.index)

    return (series - minimum) / (maximum - minimum)


def current_burden_score(df, features):

    components = []

    for feature in features:

        normalized = minmax_series(df[feature])

        if feature in [
            "general_emotional_condition_mean",
            "general_physical_condition_mean",
        ]:
            burden = 1 - normalized
        else:
            burden = normalized

        components.append(burden)

    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def historical_deviation_score(df, features):

    components = []

    for feature in features:
        normalized = minmax_series(df[feature].abs())
        components.append(normalized)

    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def calculate_index(df, current_features, deviation_features):

    current = current_burden_score(df, current_features)
    historical = historical_deviation_score(df, deviation_features)

    result = pd.Series(np.nan, index=df.index)

    both = current.notna() & historical.notna()
    only_current = current.notna() & historical.isna()
    only_historical = current.isna() & historical.notna()

    result.loc[both] = (
        0.5 * current.loc[both]
        + 0.5 * historical.loc[both]
    )

    result.loc[only_current] = current.loc[only_current]
    result.loc[only_historical] = historical.loc[only_historical]

    return result * 100


print("=" * 70)
print("INVESTIGATING EXTREME SENSITIVITY CASES")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
original = pd.read_csv(INDEX_FILE)

baseline = original[
    "personalized_risk_pattern_index"
].reset_index(drop=True)

# Keep useful identifiers
report = pd.DataFrame({
    "User id": df["User id"].values,
    "Cycle number": df["Cycle number"].values,
    "baseline_index": baseline.values,
})

# ------------------------------------------------------------
# Evidence information
# ------------------------------------------------------------

evidence_features = CURRENT_FEATURES + DEVIATION_FEATURES

report["available_evidence_count"] = df[
    evidence_features
].notna().sum(axis=1)

report["evidence_coverage_percent"] = (
    report["available_evidence_count"]
    / len(evidence_features)
    * 100
)

# ------------------------------------------------------------
# Investigate each feature removal
# ------------------------------------------------------------

for feature in CURRENT_FEATURES:

    remaining_current = [
        f for f in CURRENT_FEATURES if f != feature
    ]

    sensitivity = calculate_index(
        df,
        remaining_current,
        DEVIATION_FEATURES
    ).reset_index(drop=True)

    report[
        f"change_remove_{feature}"
    ] = (sensitivity - baseline).abs()


for feature in DEVIATION_FEATURES:

    remaining_deviation = [
        f for f in DEVIATION_FEATURES if f != feature
    ]

    sensitivity = calculate_index(
        df,
        CURRENT_FEATURES,
        remaining_deviation
    ).reset_index(drop=True)

    report[
        f"change_remove_{feature}"
    ] = (sensitivity - baseline).abs()


# ------------------------------------------------------------
# Find the largest sensitivity effect for each cycle
# ------------------------------------------------------------

change_columns = [
    c for c in report.columns
    if c.startswith("change_remove_")
]

report["largest_sensitivity_change"] = report[
    change_columns
].max(axis=1)

report["most_influential_feature"] = (
    report[change_columns]
    .fillna(-1)
    .idxmax(axis=1)
    .str.replace("change_remove_", "", regex=False)
)

# Rows where every sensitivity calculation was unavailable
all_missing = report[change_columns].isna().all(axis=1)

report.loc[
    all_missing,
    "most_influential_feature"
] = "Insufficient data"
# ------------------------------------------------------------
# Show extreme cases
# ------------------------------------------------------------

extreme = report[
    report["largest_sensitivity_change"] >= 15
].copy()

extreme = extreme.sort_values(
    "largest_sensitivity_change",
    ascending=False
)

print()
print("=" * 70)
print("EXTREME CASES (CHANGE >= 15 POINTS)")
print("=" * 70)

if len(extreme) == 0:

    print("No cycles exceeded a 15-point sensitivity change.")

else:

    print(
        extreme[
            [
                "User id",
                "Cycle number",
                "baseline_index",
                "largest_sensitivity_change",
                "most_influential_feature",
                "available_evidence_count",
                "evidence_coverage_percent",
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

report.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total cycles: {len(report)}")
print(f"Extreme cycles: {len(extreme)}")

if len(extreme) > 0:

    print(
        f"Mean coverage of extreme cycles: "
        f"{extreme['evidence_coverage_percent'].mean():.2f}%"
    )

    print(
        f"Mean baseline index of extreme cycles: "
        f"{extreme['baseline_index'].mean():.2f}"
    )

    print(
        f"Mean sensitivity change: "
        f"{extreme['largest_sensitivity_change'].mean():.2f}"
    )

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("This analysis identifies influential participant-cycles.")
print("It does not determine whether those observations are clinically abnormal.")