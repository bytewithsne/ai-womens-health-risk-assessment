from pathlib import Path
import pandas as pd

# Location of the downloaded KI EndoLIST dataset
DATASET_DIR = (
    Path.home()
    / "Downloads"
    / "ki-endolist-endometriosis-longitudinal-individualized-symptoms-tracking-dataset-1.0.0"
    / "ki-endolist-endometriosis-longitudinal-individualized-symptoms-tracking-dataset-1.0.0"
)

print("=" * 70)
print("KI EndoLIST DATASET INSPECTION")
print("=" * 70)

# Find all participant files
user_files = sorted(DATASET_DIR.glob("User*.csv"))

print(f"\nDataset folder:")
print(DATASET_DIR)

print(f"\nParticipant files found: {len(user_files)}")

# Inspect every participant file
summary = []

for file in user_files:
    try:
        df = pd.read_csv(file)

        summary.append({
            "file": file.name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": int(df.isna().sum().sum())
        })

        print(
            f"{file.name:<12} "
            f"rows={len(df):<5} "
            f"columns={len(df.columns):<3} "
            f"missing={int(df.isna().sum().sum())}"
        )

    except Exception as e:
        print(f"{file.name}: ERROR - {e}")

# Create a summary table
summary_df = pd.DataFrame(summary)

print("\n" + "=" * 70)
print("OVERALL SUMMARY")
print("=" * 70)

if not summary_df.empty:
    print(f"Participants: {len(summary_df)}")
    print(f"Total rows: {summary_df['rows'].sum()}")
    print(f"Total columns across files: {summary_df['columns'].sum()}")
    print(f"Total missing values: {summary_df['missing_values'].sum()}")

print("\nInspection complete.")