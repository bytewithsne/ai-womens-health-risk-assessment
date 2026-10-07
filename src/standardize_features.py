import pandas as pd
from pathlib import Path
import re


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = Path("data/processed/combined_longitudinal_data.csv")

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "standardized_features.csv"
AUDIT_FILE = OUTPUT_DIR / "standardization_audit.csv"


# ---------------------------------------------------------
# Features selected from the dataset analysis
# ---------------------------------------------------------

FEATURES = [
    "Cycle day",
    "Cycle number",
    "Day in study",
    "First day of period",
    "Period (0-no, 1-yes)",
    "Ovulation test values (0.0-4.0)",
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
# Read combined dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STANDARDIZING RESEARCH FEATURES")
print("=" * 70)

print(f"Input rows: {len(df)}")
print(f"Input columns: {len(df.columns)}")


# ---------------------------------------------------------
# Keep participant and selected features
# ---------------------------------------------------------

available_features = [
    feature for feature in FEATURES
    if feature in df.columns
]

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

print(f"\nRequested features: {len(FEATURES)}")
print(f"Available features: {len(available_features)}")

if missing_features:
    print("\nFeatures not found:")
    for feature in missing_features:
        print(" -", feature)


# ---------------------------------------------------------
# Create standardized dataset
# ---------------------------------------------------------

standardized = pd.DataFrame()

# Preserve participant identity
if "User id" in df.columns:
    standardized["User id"] = df["User id"]

# Preserve important cycle information
for feature in [
    "Cycle day",
    "Cycle number",
    "Day in study",
    "First day of period"
]:
    if feature in df.columns:
        standardized[feature] = df[feature]


# ---------------------------------------------------------
# Standardization
# ---------------------------------------------------------

audit_records = []

for feature in available_features:

    # Skip variables already copied above
    if feature in standardized.columns:
        continue

    series = pd.to_numeric(df[feature], errors="coerce")

    # Determine observed range
    observed_min = series.min()
    observed_max = series.max()

    # Extract an explicit scale from the variable description
    match = re.search(
        r"\((?:[^,]*,\s*)?(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)",
        feature
    )

    if match:
        scale_min = float(match.group(1))
        scale_max = float(match.group(2))

        # Normalize to 0-1 using the documented scale
        if scale_max > scale_min:
            normalized = (
                (series - scale_min)
                / (scale_max - scale_min)
            )
        else:
            normalized = series

        standardized_name = feature + " [normalized 0-1]"

        standardized[standardized_name] = normalized

        method = (
            f"Min-max normalization using documented "
            f"scale {scale_min}-{scale_max}"
        )

    else:
        # If the scale cannot be safely determined,
        # preserve the original numeric value.
        standardized[feature] = series

        method = "Original numeric representation preserved"

        scale_min = None
        scale_max = None

    audit_records.append({
        "original_feature": feature,
        "standardized_feature": (
            standardized_name
            if match
            else feature
        ),
        "documented_scale_min": scale_min,
        "documented_scale_max": scale_max,
        "observed_min": observed_min,
        "observed_max": observed_max,
        "non_missing_values": int(series.notna().sum()),
        "missing_values": int(series.isna().sum()),
        "method": method
    })


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

standardized.to_csv(
    OUTPUT_FILE,
    index=False
)

audit_df = pd.DataFrame(audit_records)

audit_df.to_csv(
    AUDIT_FILE,
    index=False
)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("STANDARDIZATION COMPLETE")
print("=" * 70)

print(f"Output rows: {len(standardized)}")
print(f"Output columns: {len(standardized.columns)}")

print("\nSaved:")
print(OUTPUT_FILE)
print(AUDIT_FILE)

print("\nImportant:")
print("- Original KI EndoLIST files were not modified.")
print("- Original feature names/scales are preserved in the audit.")
print("- Explicit symptom scales were normalized to 0-1.")
print("- Missing values were NOT artificially filled.")
print("- No medical diagnosis is produced by this step.")