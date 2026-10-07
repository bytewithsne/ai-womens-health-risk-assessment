import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PERSONALIZED RISK PATTERN INDEX - SENSITIVITY ANALYSIS
# ============================================================

INPUT_FILE = Path("data/processed/temporal_twin_features.csv")
INDEX_FILE = Path("data/processed/risk_pattern_index.csv")
OUTPUT_FILE = Path("data/processed/sensitivity_analysis.csv")

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
    """Normalize a feature to 0-1 using its observed dataset range."""
    minimum = series.min(skipna=True)
    maximum = series.max(skipna=True)

    if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
        return pd.Series(np.nan, index=series.index)

    return (series - minimum) / (maximum - minimum)


def current_burden_score(df, features):
    """
    Calculate normalized current symptom burden.

    Higher burden:
        bleeding
        stool
        lower back pain
        headache

    Lower values indicate greater burden:
        emotional condition
        physical condition
    """

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

    if not components:
        return pd.Series(np.nan, index=df.index)

    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def historical_deviation_score(df, features):
    """
    Calculate normalized absolute deviation from historical baseline.
    """

    components = []

    for feature in features:
        normalized = minmax_series(df[feature].abs())
        components.append(normalized)

    if not components:
        return pd.Series(np.nan, index=df.index)

    return pd.concat(components, axis=1).mean(axis=1, skipna=True)


def calculate_index(df, current_features, deviation_features):

    current = current_burden_score(df, current_features)
    historical = historical_deviation_score(df, deviation_features)

    combined = pd.Series(np.nan, index=df.index)

    both = current.notna() & historical.notna()
    only_current = current.notna() & historical.isna()
    only_historical = current.isna() & historical.notna()

    combined.loc[both] = (
        0.5 * current.loc[both]
        + 0.5 * historical.loc[both]
    )

    combined.loc[only_current] = current.loc[only_current]
    combined.loc[only_historical] = historical.loc[only_historical]

    return combined * 100


print("=" * 70)
print("RUNNING SENSITIVITY ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
original = pd.read_csv(INDEX_FILE)

original_index = original["personalized_risk_pattern_index"]

results = []

# ------------------------------------------------------------
# Baseline
# ------------------------------------------------------------

results.append({
    "removed_feature": "NONE - baseline",
    "feature_group": "baseline",
    "mean_index": original_index.mean(),
    "std_index": original_index.std(),
    "mean_absolute_change": 0,
    "correlation_with_baseline": 1.0,
    "max_absolute_change": 0,
})

# ------------------------------------------------------------
# Remove one CURRENT feature at a time
# ------------------------------------------------------------

for feature in CURRENT_FEATURES:

    remaining_current = [
        f for f in CURRENT_FEATURES if f != feature
    ]

    sensitivity_index = calculate_index(
        df,
        remaining_current,
        DEVIATION_FEATURES
    )

    comparison = pd.concat(
        [
            original_index.rename("original"),
            sensitivity_index.rename("sensitivity")
        ],
        axis=1
    ).dropna()

    absolute_change = (
        comparison["sensitivity"]
        - comparison["original"]
    ).abs()

    correlation = comparison["original"].corr(
        comparison["sensitivity"]
    )

    results.append({
        "removed_feature": feature,
        "feature_group": "current_burden",
        "mean_index": sensitivity_index.mean(),
        "std_index": sensitivity_index.std(),
        "mean_absolute_change": absolute_change.mean(),
        "correlation_with_baseline": correlation,
        "max_absolute_change": absolute_change.max(),
    })


# ------------------------------------------------------------
# Remove one HISTORICAL DEVIATION feature at a time
# ------------------------------------------------------------

for feature in DEVIATION_FEATURES:

    remaining_deviation = [
        f for f in DEVIATION_FEATURES if f != feature
    ]

    sensitivity_index = calculate_index(
        df,
        CURRENT_FEATURES,
        remaining_deviation
    )

    comparison = pd.concat(
        [
            original_index.rename("original"),
            sensitivity_index.rename("sensitivity")
        ],
        axis=1
    ).dropna()

    absolute_change = (
        comparison["sensitivity"]
        - comparison["original"]
    ).abs()

    correlation = comparison["original"].corr(
        comparison["sensitivity"]
    )

    results.append({
        "removed_feature": feature,
        "feature_group": "historical_deviation",
        "mean_index": sensitivity_index.mean(),
        "std_index": sensitivity_index.std(),
        "mean_absolute_change": absolute_change.mean(),
        "correlation_with_baseline": correlation,
        "max_absolute_change": absolute_change.max(),
    })


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="mean_absolute_change",
    ascending=False
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("SENSITIVITY ANALYSIS COMPLETE")
print("=" * 70)

print()
print(results_df.to_string(index=False))

print()
print("Most influential features:")
print(
    results_df[
        results_df["removed_feature"] != "NONE - baseline"
    ][
        [
            "removed_feature",
            "feature_group",
            "mean_absolute_change",
            "correlation_with_baseline",
            "max_absolute_change",
        ]
    ].head(10).to_string(index=False)
)

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("Interpretation:")
print("- Higher correlation = more stable when the feature is removed.")
print("- Lower mean absolute change = less dependence on that feature.")
print("- A feature causing very large changes deserves investigation.")
print("- This is a robustness analysis, not clinical validation.")