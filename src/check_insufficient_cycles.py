import pandas as pd
from pathlib import Path

INDEX_FILE = Path("data/processed/risk_pattern_index.csv")
OUTPUT_FILE = Path("data/processed/insufficient_cycles_check.csv")

print("=" * 70)
print("CHECKING INSUFFICIENT-EVIDENCE CYCLES")
print("=" * 70)

df = pd.read_csv(INDEX_FILE)

index_col = "personalized_risk_pattern_index"
coverage_col = "evidence_coverage_percent"

# Find rows that have evidence coverage but no PSPI
missing_index = df[
    df[index_col].isna()
].copy()

print()
print("Rows without PSPI:", len(missing_index))

print()
print("=" * 70)
print("ROWS WITHOUT PSPI")
print("=" * 70)

columns_to_show = [
    "User id",
    "Cycle number",
    "current_burden_component",
    "historical_deviation_component",
    index_col,
    coverage_col,
]

available_columns = [
    col for col in columns_to_show
    if col in missing_index.columns
]

print(
    missing_index[available_columns]
    .to_string(index=False)
)

print()
print("=" * 70)
print("EVIDENCE COVERAGE")
print("=" * 70)

print(
    missing_index[coverage_col]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

missing_index.to_csv(OUTPUT_FILE, index=False)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)
print(OUTPUT_FILE)