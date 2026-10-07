import pandas as pd
from pathlib import Path
import re

# Original KI EndoLIST dataset location
DATA_DIR = Path(
    r"C:\Users\Iamsn\Downloads\ki-endolist-endometriosis-longitudinal-individualized-symptoms-tracking-dataset-1.0.0\ki-endolist-endometriosis-longitudinal-individualized-symptoms-tracking-dataset-1.0.0"
)

# Output location inside our project
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_column_name(column):
    """Make column names easier to work with without changing their meaning."""
    column = str(column).strip()
    column = re.sub(r"\s+", " ", column)
    return column


all_data = []
audit_records = []

participant_files = sorted(DATA_DIR.glob("User*.csv"))

print(f"Found {len(participant_files)} participant files.")


for file in participant_files:

    user_id = file.stem

    try:
        df = pd.read_csv(file)

        # Preserve the original column names, only removing accidental whitespace
        df.columns = [clean_column_name(c) for c in df.columns]

        # Add participant identifier
        if "User id" not in df.columns:
            df["User id"] = user_id

        # Record the structure for research auditability
        for column in df.columns:
            audit_records.append({
                "user_id": user_id,
                "original_column": column,
                "data_type": str(df[column].dtype),
                "non_null_values": int(df[column].notna().sum()),
                "missing_values": int(df[column].isna().sum()),
                "unique_values": int(df[column].nunique(dropna=True))
            })

        all_data.append(df)

        print(
            f"Loaded {user_id}: "
            f"{len(df)} rows, {len(df.columns)} columns"
        )

    except Exception as e:
        print(f"ERROR loading {file.name}: {e}")


# Combine all participant data
combined = pd.concat(
    all_data,
    ignore_index=True,
    sort=False
)

# Save processed dataset locally
combined_path = OUTPUT_DIR / "combined_longitudinal_data.csv"
combined.to_csv(combined_path, index=False)

# Save an audit file documenting the original variables
audit_df = pd.DataFrame(audit_records)

audit_path = OUTPUT_DIR / "variable_audit.csv"
audit_df.to_csv(audit_path, index=False)


print("\n----------------------------------------")
print("PREPROCESSING COMPLETE")
print("----------------------------------------")

print(f"Participants: {len(participant_files)}")
print(f"Combined rows: {len(combined)}")
print(f"Combined columns: {len(combined.columns)}")
print(f"Missing values: {int(combined.isna().sum().sum())}")

print(f"\nSaved:")
print(combined_path)
print(audit_path)

print("\nImportant:")
print("Original KI EndoLIST files were NOT modified.")
print("Processed health data remains local and is excluded from Git.")