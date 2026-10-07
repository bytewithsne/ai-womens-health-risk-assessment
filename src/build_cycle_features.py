import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/standardized_features.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "cycle_level_features.csv"


# ---------------------------------------------------------
# Load standardized daily data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("BUILDING CYCLE-LEVEL FEATURES")
print("=" * 70)

print(f"Daily observations: {len(df)}")
print(f"Participants: {df['User id'].nunique()}")


# ---------------------------------------------------------
# Features selected during standardization
# ---------------------------------------------------------

feature_columns = [
    "Ovulation test values (0.0-4.0) [normalized 0-1]",
    "Bleeding (0-no bleeding, 4-heaviest bleeding)",
    "Type of stool (0-no stool, 7-bowel blockage)",
    "General emotional condition (1-bad, 10-amazing)",
    "General physical condition (1-bad, 10-amazing)",
    "Sick day (0-no, 1-yes)",
    "Appetite / no appetite (0-no appetite, 10-most appetite)",
    "Lower back pain (0-no pain, 4-worst pain)",
    "Nausea (0-no nausea, 4-worst nausea)",
    "Breast tenderness (0-no pain, 4-worst pain)",
    "Headache (0-no pain, 4-worst pain)",
    "Period pain (0-no pain, 4-worst pain)",
    "Abdominal cramps (0-no pain, 4-worst pain)",
    "Endo belly (0-no endo belly , 4-worst endo belly)",
    "Ovaries pain (0-no pain, 4-worst pain)",
    "Pain before bowel movement (0-no pain, 4-worst pain)",
    "Muscle pain / flu like (0-no pain, 4-worst pain)",
    "Muscle weakness (0-no pain, 4-worst pain)",
    "Painful bowel movement (0-no pain, 4-worst pain)",
    "Right pelvic pain (0-no pain, 4-worst pain)",
]


# ---------------------------------------------------------
# Check features
# ---------------------------------------------------------

available_features = [
    column
    for column in feature_columns
    if column in df.columns
]

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]

print(f"\nSelected features: {len(available_features)}")

if missing_features:
    print("\nWARNING - Missing features:")
    for feature in missing_features:
        print(f"  - {feature}")


# ---------------------------------------------------------
# Convert core fields to numeric
# ---------------------------------------------------------

for column in [
    "Cycle day",
    "Cycle number",
    "Day in study",
    "First day of period",
]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ---------------------------------------------------------
# Create cycle groups
# ---------------------------------------------------------

cycle_groups = df.groupby(
    ["User id", "Cycle number"],
    dropna=False
)


# ---------------------------------------------------------
# Basic cycle information
# ---------------------------------------------------------

cycle_features = cycle_groups.size().reset_index(
    name="observed_days"
)

cycle_features["cycle_day_min"] = (
    cycle_groups["Cycle day"].min().values
)

cycle_features["cycle_day_max"] = (
    cycle_groups["Cycle day"].max().values
)

cycle_features["cycle_day_range"] = (
    cycle_features["cycle_day_max"]
    - cycle_features["cycle_day_min"]
    + 1
)


# ---------------------------------------------------------
# Aggregate symptom features
# ---------------------------------------------------------

for feature in available_features:

    # Remove scale description to create readable names
    short_name = feature.split(" (")[0]

    if "[normalized" in short_name:
        short_name = short_name.split(" [normalized")[0]

    safe_name = (
        short_name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(",", "")
        .replace("-", "_")
    )

    while "__" in safe_name:
        safe_name = safe_name.replace("__", "_")

    # Mean intensity
    cycle_features[f"{safe_name}_mean"] = (
        cycle_groups[feature].mean().values
    )

    # Maximum intensity
    cycle_features[f"{safe_name}_max"] = (
        cycle_groups[feature].max().values
    )

    # Variability
    cycle_features[f"{safe_name}_variability"] = (
        cycle_groups[feature].std().values
    )

    # Number of observations
    cycle_features[f"{safe_name}_observed_days"] = (
        cycle_groups[feature].count().values
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

cycle_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CYCLE FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Cycle-level rows: {len(cycle_features)}")
print(f"Cycle-level columns: {len(cycle_features.columns)}")

print("\nSaved:")
print(OUTPUT_FILE)

print("\nEach row represents:")
print("ONE PARTICIPANT × ONE MENSTRUAL CYCLE")

print("\nFor each selected feature we calculate:")
print("- Mean intensity")
print("- Maximum intensity")
print("- Variability")
print("- Number of observed days")

print("\nNo diagnosis or medical prediction is performed.")