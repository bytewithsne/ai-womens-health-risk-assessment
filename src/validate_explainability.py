import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/processed/pspi_explanations.csv")
OUTPUT_FILE = Path("data/processed/explainability_validation.csv")

print("=" * 70)
print("EXPLAINABILITY VALIDATION")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

index_col = "personalized_risk_pattern_index"

# ---------------------------------------------------------------
# 1. Basic completeness
# ---------------------------------------------------------------

scored = df[df[index_col].notna()].copy()
insufficient = df[df[index_col].isna()].copy()

print()
print("=" * 70)
print("BASIC COUNTS")
print("=" * 70)

print("Total cycles:", len(df))
print("Scored cycles:", len(scored))
print("Insufficient cycles:", len(insufficient))

# ---------------------------------------------------------------
# 2. Contributor completeness
# ---------------------------------------------------------------

scored["has_current_contributor"] = (
    scored["primary_current_contributor"].notna()
    & ~scored["primary_current_contributor"].str.contains(
        "No current-state",
        na=False
    )
)

scored["has_personalized_change"] = (
    scored["primary_personalized_change"].notna()
    & ~scored["primary_personalized_change"].str.contains(
        "No historical comparison",
        na=False
    )
)

print()
print("=" * 70)
print("CONTRIBUTOR COMPLETENESS")
print("=" * 70)

print(
    "Scored cycles with current contributor:",
    scored["has_current_contributor"].sum(),
    f"({scored['has_current_contributor'].mean() * 100:.1f}%)"
)

print(
    "Scored cycles with personalized change:",
    scored["has_personalized_change"].sum(),
    f"({scored['has_personalized_change'].mean() * 100:.1f}%)"
)

# ---------------------------------------------------------------
# 3. Current contributor distribution
# ---------------------------------------------------------------

print()
print("=" * 70)
print("PRIMARY CURRENT CONTRIBUTOR DISTRIBUTION")
print("=" * 70)

print(
    scored["primary_current_contributor"]
    .value_counts(dropna=False)
    .to_string()
)

# ---------------------------------------------------------------
# 4. Personalized change availability
# ---------------------------------------------------------------

print()
print("=" * 70)
print("PERSONALIZED CHANGE AVAILABILITY")
print("=" * 70)

print(
    scored["has_personalized_change"]
    .value_counts()
    .rename(
        index={
            True: "Historical comparison available",
            False: "No historical comparison"
        }
    )
    .to_string()
)

# ---------------------------------------------------------------
# 5. Pattern category distribution
# ---------------------------------------------------------------

print()
print("=" * 70)
print("PATTERN CATEGORY")
print("=" * 70)

print(
    df["pattern_category"]
    .value_counts(dropna=False)
    .to_string()
)

# ---------------------------------------------------------------
# 6. Evidence coverage by category
# ---------------------------------------------------------------

print()
print("=" * 70)
print("EVIDENCE COVERAGE BY PATTERN CATEGORY")
print("=" * 70)

coverage_summary = (
    df.groupby("pattern_category", observed=True)
    ["evidence_coverage_percent"]
    .agg(
        cycles="count",
        mean_coverage="mean",
        minimum="min",
        maximum="max"
    )
    .round(2)
)

print(coverage_summary.to_string())

# ---------------------------------------------------------------
# 7. Check for contradictory explanations
# ---------------------------------------------------------------

print()
print("=" * 70)
print("EXPLANATION CONSISTENCY CHECK")
print("=" * 70)

# A scored cycle should not say "Insufficient evidence"
contradiction_1 = scored[
    scored["explanation"].str.contains(
        "No PSPI was generated",
        na=False
    )
]

# An insufficient cycle should not contain a normal contributor
contradiction_2 = insufficient[
    insufficient["primary_current_contributor"].fillna("").str.contains(
        "No current-state|Insufficient evidence",
        regex=True
    ) == False
]

print(
    "Scored cycles incorrectly marked insufficient:",
    len(contradiction_1)
)

print(
    "Insufficient cycles with unexpected contributor:",
    len(contradiction_2)
)

# ---------------------------------------------------------------
# 8. Save validation report
# ---------------------------------------------------------------

validation = pd.DataFrame({
    "metric": [
        "total_cycles",
        "scored_cycles",
        "insufficient_cycles",
        "scored_with_current_contributor",
        "scored_with_personalized_change",
        "contradictory_scored_explanations",
        "contradictory_insufficient_explanations"
    ],
    "value": [
        len(df),
        len(scored),
        len(insufficient),
        int(scored["has_current_contributor"].sum()),
        int(scored["has_personalized_change"].sum()),
        len(contradiction_1),
        len(contradiction_2)
    ]
})

validation.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 70)
print("SAVED")
print("=" * 70)

print(OUTPUT_FILE)