from pathlib import Path
from collections import Counter
import pandas as pd

DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "ki-endolist-endometriosis-longitudinal-individualized-symptoms-tracking-dataset-1.0.0"
    / "ki-endolist-endometriosis-longitudinal-individualized-symptoms-tracking-dataset-1.0.0"
)

user_files = sorted(DATASET_DIR.glob("User*.csv"))

column_counter = Counter()

for file in user_files:
    df = pd.read_csv(file)

    # Count each variable once per participant
    column_counter.update(set(df.columns))

results = pd.DataFrame(
    column_counter.items(),
    columns=["variable", "participants"]
)

results = results.sort_values(
    by=["participants", "variable"],
    ascending=[False, True]
)

print("=" * 80)
print("VARIABLE FREQUENCY ACROSS PARTICIPANTS")
print("=" * 80)

print(f"\nParticipants analyzed: {len(user_files)}")
print(f"Unique variables found: {len(results)}")

print("\nVariables present in at least 20 participants:\n")

common = results[results["participants"] >= 20]

for _, row in common.iterrows():
    print(
        f"{int(row['participants']):>2}/"
        f"{len(user_files)}  {row['variable']}"
    )

print("\n" + "=" * 80)
print("ALL VARIABLES")
print("=" * 80)

for _, row in results.iterrows():
    print(
        f"{int(row['participants']):>2}/"
        f"{len(user_files)}  {row['variable']}"
    )