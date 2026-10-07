import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/digital_twin_features.csv")
OUTPUT_DIR = Path("data/processed")

OUTPUT_FILE = OUTPUT_DIR / "selected_twin_features.csv"
AUDIT_FILE = OUTPUT_DIR / "feature_selection_audit.csv"

MIN_COVERAGE = 0.40


# ---------------------------------------------------------
# Load Digital Twin features
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("DIGITAL TWIN FEATURE SELECTION")
print("=" * 70)

print(f"Input rows: {len(df)}")
print(f"Input columns: {len(df.columns)}")


# ---------------------------------------------------------
# Identify feature columns
# ---------------------------------------------------------

identifier_columns = [
    "User id",
    "Cycle number",
]

feature_columns = [
    column
    for column in df.columns
    if column not in identifier_columns
]


# ---------------------------------------------------------
# Calculate coverage
# ---------------------------------------------------------

coverage = df[feature_columns].notna().mean()


# ---------------------------------------------------------
# Classify features
# ---------------------------------------------------------

audit = pd.DataFrame({
    "feature": feature_columns,
    "observed_cycles": df[feature_columns].notna().sum().values,
    "total_cycles": len(df),
    "coverage_percentage": (
        coverage.values * 100
    ),
})

audit["selected"] = (
    audit["coverage_percentage"]
    >= MIN_COVERAGE * 100
)


# ---------------------------------------------------------
# Select features
# ---------------------------------------------------------

selected_features = audit.loc[
    audit["selected"],
    "feature"
].tolist()


selected_columns = (
    identifier_columns
    + selected_features
)

selected_df = df[selected_columns].copy()


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

selected_df.to_csv(
    OUTPUT_FILE,
    index=False
)

audit.to_csv(
    AUDIT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE SELECTION COMPLETE")
print("=" * 70)

print(f"Coverage threshold: {MIN_COVERAGE * 100:.0f}%")

print(f"Total candidate features: {len(feature_columns)}")
print(f"Selected features: {len(selected_features)}")
print(
    f"Removed features: "
    f"{len(feature_columns) - len(selected_features)}"
)

print("\nSaved:")
print(OUTPUT_FILE)
print(AUDIT_FILE)

print("\nSelected features:")

for feature in selected_features:
    print(f"  ✓ {feature}")

print("\nImportant:")
print("- Selection is based only on data coverage.")
print("- No outcome information is used.")
print("- Missing values are not converted to zero.")
print("- This is a feature-quality step, not a diagnosis model.")