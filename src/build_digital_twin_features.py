import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/cycle_level_features.csv")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "digital_twin_features.csv"


# ---------------------------------------------------------
# Load cycle-level data
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("BUILDING DIGITAL TWIN FEATURES")
print("=" * 70)

print(f"Input cycles: {len(df)}")
print(f"Participants: {df['User id'].nunique()}")


# ---------------------------------------------------------
# Sort chronologically within each participant
# ---------------------------------------------------------

df = df.sort_values(
    ["User id", "Cycle number"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Core cycle features
# ---------------------------------------------------------

twin = df[
    [
        "User id",
        "Cycle number",
        "observed_days",
        "cycle_day_min",
        "cycle_day_max",
        "cycle_day_range",
    ]
].copy()


# ---------------------------------------------------------
# Select symptom mean features
# ---------------------------------------------------------

symptom_mean_columns = [
    column
    for column in df.columns
    if column.endswith("_mean")
]


print(f"Symptom mean features: {len(symptom_mean_columns)}")


# ---------------------------------------------------------
# Add symptom intensity
# ---------------------------------------------------------

for column in symptom_mean_columns:
    twin[column] = df[column]


# ---------------------------------------------------------
# Add symptom variability
# ---------------------------------------------------------

variability_columns = [
    column
    for column in df.columns
    if column.endswith("_variability")
]

for column in variability_columns:
    twin[column] = df[column]


# ---------------------------------------------------------
# Add observation coverage
# ---------------------------------------------------------

observed_columns = [
    column
    for column in df.columns
    if column.endswith("_observed_days")
]

for column in observed_columns:
    twin[column] = df[column]


# ---------------------------------------------------------
# Calculate change from previous cycle
# ---------------------------------------------------------

for column in symptom_mean_columns:

    change_column = (
        column.replace("_mean", "_change_from_previous_cycle")
    )

    twin[change_column] = (
        df.groupby("User id")[column]
        .diff()
    )


# ---------------------------------------------------------
# Calculate participant baseline
# ---------------------------------------------------------

for column in symptom_mean_columns:

    baseline_column = (
        column.replace("_mean", "_participant_baseline")
    )

    twin[baseline_column] = (
        df.groupby("User id")[column]
        .transform("mean")
    )


# ---------------------------------------------------------
# Calculate deviation from personal baseline
# ---------------------------------------------------------

for column in symptom_mean_columns:

    baseline_column = (
        column.replace("_mean", "_participant_baseline")
    )

    deviation_column = (
        column.replace("_mean", "_deviation_from_baseline")
    )

    twin[deviation_column] = (
        twin[column]
        - twin[baseline_column]
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

twin.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("DIGITAL TWIN FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Output rows: {len(twin)}")
print(f"Output columns: {len(twin.columns)}")

print("\nSaved:")
print(OUTPUT_FILE)

print("\nDigital Twin now contains:")
print("- Cycle characteristics")
print("- Symptom intensity")
print("- Symptom variability")
print("- Observation coverage")
print("- Change from previous cycle")
print("- Participant-specific baseline")
print("- Deviation from personal baseline")

print("\nImportant:")
print("- Missing observations remain missing.")
print("- Missing symptoms are NOT treated as zero.")
print("- No diagnosis is produced.")