import pandas as pd
import numpy as np
from pathlib import Path

INDEX_FILE = Path("data/processed/risk_pattern_index.csv")
FEATURE_FILE = Path("data/processed/temporal_twin_features.csv")
OUTPUT_FILE = Path("data/processed/pspi_explanations.csv")

print("=" * 70)
print("PSPI EXPLAINABILITY LAYER")
print("=" * 70)

# ---------------------------------------------------------------
# Load PSPI results and underlying temporal features
# ---------------------------------------------------------------

index_df = pd.read_csv(INDEX_FILE)
feature_df = pd.read_csv(FEATURE_FILE)

keys = ["User id", "Cycle number"]

df = index_df.merge(
    feature_df,
    on=keys,
    how="left",
    suffixes=("", "_feature")
)

# ---------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------

CURRENT_FEATURES = {
    "bleeding_mean": "Bleeding",
    "general_emotional_condition_mean": "Emotional condition",
    "general_physical_condition_mean": "Physical condition",
    "type_of_stool_mean": "Bowel/stool pattern",
    "lower_back_pain_mean": "Lower back pain",
    "headache_mean": "Headache",
}

DEVIATION_FEATURES = {
    "bleeding_historical_deviation": "Bleeding vs personal baseline",
    "general_emotional_condition_historical_deviation":
        "Emotional condition vs personal baseline",
    "general_physical_condition_historical_deviation":
        "Physical condition vs personal baseline",
    "type_of_stool_historical_deviation":
        "Bowel/stool pattern vs personal baseline",
    "ovulation_test_values_historical_deviation":
        "Ovulation pattern vs personal baseline",
}

# ---------------------------------------------------------------
# Current-state contributors
# ---------------------------------------------------------------

def get_current_contributors(row):

    contributors = []

    # Higher value = greater burden
    higher_burden = [
        ("bleeding_mean", "Bleeding"),
        ("type_of_stool_mean", "Bowel/stool pattern"),
        ("lower_back_pain_mean", "Lower back pain"),
        ("headache_mean", "Headache"),
    ]

    for feature, label in higher_burden:

        value = row.get(feature)

        if pd.notna(value):

            contributors.append(
                {
                    "label": label,
                    "strength": float(value)
                }
            )

    # Lower value = greater burden
    lower_burden = [
        ("general_emotional_condition_mean", "Emotional condition"),
        ("general_physical_condition_mean", "Physical condition"),
    ]

    for feature, label in lower_burden:

        value = row.get(feature)

        if pd.notna(value):

            contributors.append(
                {
                    "label": label,
                    "strength": float(1 - value)
                }
            )

    contributors.sort(
        key=lambda x: x["strength"],
        reverse=True
    )

    return contributors


# ---------------------------------------------------------------
# Personalized temporal contributors
# ---------------------------------------------------------------

def get_temporal_contributors(row):

    contributors = []

    for feature, label in DEVIATION_FEATURES.items():

        value = row.get(feature)

        if pd.notna(value):

            contributors.append(
                {
                    "label": label,
                    "strength": abs(float(value)),
                    "value": float(value)
                }
            )

    contributors.sort(
        key=lambda x: x["strength"],
        reverse=True
    )

    return contributors


# ---------------------------------------------------------------
# Generate explanation
# ---------------------------------------------------------------

def generate_explanation(row):

    index = row["personalized_risk_pattern_index"]

    # -----------------------------------------------------------
    # Insufficient evidence
    # -----------------------------------------------------------

    if pd.isna(index):

        return pd.Series({
            "pattern_category": "Insufficient evidence",
            "primary_current_contributor": "Insufficient evidence",
            "secondary_current_contributor": "Insufficient evidence",
            "primary_personalized_change": "Insufficient evidence",
            "explanation": (
                "No PSPI was generated because no core evidence "
                "was available for this cycle."
            )
        })

    # -----------------------------------------------------------
    # PSPI category
    # Must match build_risk_pattern_index.py exactly:
    # <33 = Lower
    # <66 = Intermediate
    # >=66 = Higher
    # -----------------------------------------------------------

    if index < 33:
        category = "Lower"

    elif index < 66:
        category = "Intermediate"

    else:
        category = "Higher"

    # -----------------------------------------------------------
    # Current contributors
    # -----------------------------------------------------------

    current = get_current_contributors(row)

    if len(current) >= 1:
        primary_current = current[0]["label"]
    else:
        primary_current = "No current-state contributor available"

    if len(current) >= 2:
        secondary_current = current[1]["label"]
    else:
        secondary_current = "No secondary contributor available"

    # -----------------------------------------------------------
    # Personalized change
    # -----------------------------------------------------------

    temporal = get_temporal_contributors(row)

    if temporal:

        strongest = temporal[0]

        if strongest["value"] > 0:

            personalized_change = (
                f"{strongest['label']} was above the "
                f"participant's historical pattern."
            )

        elif strongest["value"] < 0:

            personalized_change = (
                f"{strongest['label']} was below the "
                f"participant's historical pattern."
            )

        else:

            personalized_change = (
                f"{strongest['label']} was close to the "
                f"participant's historical pattern."
            )

    else:

        personalized_change = (
            "No historical comparison was available."
        )

    # -----------------------------------------------------------
    # Evidence quality
    # -----------------------------------------------------------

    coverage = row.get("evidence_coverage_percent")

    if pd.notna(coverage):

        evidence_text = (
            f"Evidence coverage was {coverage:.1f}%."
        )

    else:

        evidence_text = (
            "Evidence coverage was unavailable."
        )

    # -----------------------------------------------------------
    # Final explanation
    # -----------------------------------------------------------

    explanation = (
        f"PSPI pattern category: {category}. "
        f"The strongest current-state contributor was "
        f"{primary_current}, followed by "
        f"{secondary_current}. "
        f"{personalized_change} "
        f"{evidence_text} "
        f"This is a symptom-pattern representation and "
        f"not a clinical diagnosis."
    )

    return pd.Series({
        "pattern_category": category,
        "primary_current_contributor": primary_current,
        "secondary_current_contributor": secondary_current,
        "primary_personalized_change": personalized_change,
        "explanation": explanation
    })

# ---------------------------------------------------------------
# Generate explanations
# ---------------------------------------------------------------

explanations = df.apply(
    generate_explanation,
    axis=1
)

result = pd.concat(
    [df, explanations],
    axis=1
)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

# ---------------------------------------------------------------
# Output summary
# ---------------------------------------------------------------

print()
print("=" * 70)
print("EXPLANATION SUMMARY")
print("=" * 70)

print(
    result["pattern_category"]
    .value_counts(dropna=False)
    .to_string()
)

print()
print("=" * 70)
print("EXAMPLE EXPLANATIONS")
print("=" * 70)

example_columns = [
    "User id",
    "Cycle number",
    "personalized_risk_pattern_index",
    "pattern_category",
    "primary_current_contributor",
    "secondary_current_contributor",
    "primary_personalized_change",
]

available = [
    column
    for column in example_columns
    if column in result.columns
]

print(
    result[available]
    .head(10)
    .to_string(index=False)
)

print()
print("=" * 70)
print("FULL EXPLANATION EXAMPLE")
print("=" * 70)

valid_examples = result[
    result["personalized_risk_pattern_index"].notna()
]

if len(valid_examples) > 0:

    print(
        valid_examples.iloc[0]["explanation"]
    )

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)