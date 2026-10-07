import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/selected_twin_features.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = OUTPUT_DIR / "temporal_twin_features.csv"


# ---------------------------------------------------------
# Load selected features
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("BUILDING TEMPORAL DIGITAL TWIN FEATURES")
print("=" * 70)

print(f"Input rows: {len(df)}")
print(f"Participants: {df['User id'].nunique()}")


# ---------------------------------------------------------
# Sort chronologically
# ---------------------------------------------------------

df = df.sort_values(
    ["User id", "Cycle number"]
).reset_index(drop=True)


# ---------------------------------------------------------
# Identify current-cycle symptom features
# ---------------------------------------------------------

mean_columns = [
    column
    for column in df.columns
    if column.endswith("_mean")
]


# ---------------------------------------------------------
# Create PREVIOUS-CYCLES baseline
# ---------------------------------------------------------

for column in mean_columns:

    baseline_name = (
        column.replace(
            "_mean",
            "_historical_baseline"
        )
    )

    df[baseline_name] = (
        df.groupby("User id")[column]
        .transform(
            lambda series:
            series.shift(1).expanding().mean()
        )
    )


# ---------------------------------------------------------
# Calculate deviation from historical baseline
# ---------------------------------------------------------

for column in mean_columns:

    baseline_name = (
        column.replace(
            "_mean",
            "_historical_baseline"
        )
    )

    deviation_name = (
        column.replace(
            "_mean",
            "_historical_deviation"
        )
    )

    df[deviation_name] = (
        df[column]
        - df[baseline_name]
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TEMPORAL FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Output rows: {len(df)}")
print(f"Output columns: {len(df.columns)}")

print("\nSaved:")
print(OUTPUT_FILE)

print("\nHistorical baseline means:")
print("- Cycle 1 has no previous-cycle baseline.")
print("- Cycle 2 uses Cycle 1.")
print("- Cycle 3 uses Cycles 1-2.")
print("- Later cycles use all preceding cycles.")

print("\nThis prevents the current cycle from influencing")
print("its own historical baseline.")