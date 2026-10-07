import pandas as pd
import json
from pathlib import Path


FEATURE_FILE = Path("data/processed/temporal_twin_features.csv")
OUTPUT_FILE = Path("data/processed/pspi_reference_parameters.json")


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


df = pd.read_csv(FEATURE_FILE)

parameters = {
    "current_features": {},
    "historical_deviation_features": {},
    "method": "Dataset-derived min-max normalization",
    "purpose": "Fixed reference parameters for reproducible PSPI calculation",
    "warning": "Research prototype only. Not clinically validated."
}


for feature in CURRENT_FEATURES:
    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    parameters["current_features"][feature] = {
        "min": float(values.min()),
        "max": float(values.max())
    }


for feature in DEVIATION_FEATURES:
    values = pd.to_numeric(df[feature], errors="coerce").dropna()

    parameters["historical_deviation_features"][feature] = {
        "min": float(values.min()),
        "max": float(values.max())
    }


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(parameters, f, indent=4)


print("=" * 70)
print("PSPI REFERENCE PARAMETERS SAVED")
print("=" * 70)

print(f"Current features: {len(CURRENT_FEATURES)}")
print(f"Historical deviation features: {len(DEVIATION_FEATURES)}")

print()
print("Saved:")
print(OUTPUT_FILE)

print()
print("These fixed parameters will be used for reproducible PSPI calculations.")