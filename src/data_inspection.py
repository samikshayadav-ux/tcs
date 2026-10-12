"""
Data Inspection Module for AutoRisk AI.

Inspects the Kaggle Vehicle Insurance Claim Fraud Detection dataset (fraud_oracle.csv),
analyzes schema, data types, missing values, duplicates, class distribution,
and reviews columns for potential identifiers, leakage, and post-outcome risks.
"""

import argparse
import os
import sys
from typing import Dict, Any, List
import pandas as pd
import numpy as np


TARGET_COLUMN = "FraudFound_P"

# Columns identified as pure identifiers or operational metadata
IDENTIFIER_COLUMNS = [
    "PolicyNumber",  # Unique sequential integer 1..N (memorization risk)
    "RepNumber",     # Internal claim representative ID 1..16 (operational routing)
]

# Columns identified as leakage, post-outcome, or temporal artifacts
LEAKAGE_COLUMNS = [
    "Year",                 # Claim collection year (1994-1996) - causes temporal overfitting
    "MonthClaimed",         # Redundant with incident Month, contains corrupt '0' entries
    "DayOfWeekClaimed",     # Redundant with incident DayOfWeek, contains corrupt '0' entries
    "WeekOfMonthClaimed",   # Redundant with incident WeekOfMonth
]

# Operational / post-outcome claim processing attributes to flag for review
POST_OUTCOME_REVIEW_COLUMNS = [
    "NumberOfSuppliments",  # Additional repair estimates submitted during/after appraisal
    "AddressChange_Claim",  # Address changes logged during claim processing
    "Days_Policy_Accident", # Days between policy inception and accident
    "Days_Policy_Claim",    # Days between policy inception and claim filing
    "PoliceReportFiled",    # Formal police report indicator (completed post-incident)
    "WitnessPresent",       # Witness presence noted during accident investigation
    "AgentType",            # Internal vs External insurance agent
]


def load_dataset(file_path: str) -> pd.DataFrame:
    """Loads the insurance claim fraud dataset with UTF-8 BOM tolerance."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset not found at '{file_path}'. "
            f"Please ensure 'fraud_oracle.csv' is placed at the specified path."
        )
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV at '{file_path}': {exc}") from exc
    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data inspection on the insurance dataset.
    Returns a dictionary of inspection summaries.
    """
    total_rows, total_cols = df.shape

    # Data types and missing values
    col_summary = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        nunique = int(df[col].nunique())
        dtype = str(df[col].dtype)
        col_summary.append({
            "column": col,
            "dtype": dtype,
            "null_count": null_count,
            "null_pct": round(null_count / total_rows * 100, 2),
            "nunique": nunique
        })

    # Duplicates
    duplicate_count = int(df.duplicated().sum())

    # Target class distribution
    target_info = {}
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        pcts = (df[TARGET_COLUMN].value_counts(normalize=True, dropna=False) * 100).to_dict()
        target_info = {
            "target_column": TARGET_COLUMN,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "percentages": {str(k): round(float(v), 2) for k, v in pcts.items()},
            "imbalance_ratio": round(float(counts.get(0, 1) / max(counts.get(1, 1), 1)), 2)
        }
    else:
        target_info = {"error": f"Target column '{TARGET_COLUMN}' not found in dataset."}

    # Numeric summaries
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_summary = {}
    for col in numeric_df.columns:
        desc = numeric_df[col].describe().to_dict()
        # Check zero-age quirk
        zeros = int((numeric_df[col] == 0).sum())
        numeric_summary[col] = {
            "mean": round(float(desc.get("mean", 0)), 2),
            "std": round(float(desc.get("std", 0)), 2),
            "min": float(desc.get("min", 0)),
            "25%": float(desc.get("25%", 0)),
            "50%": float(desc.get("50%", 0)),
            "75%": float(desc.get("75%", 0)),
            "max": float(desc.get("max", 0)),
            "zeros_count": zeros
        }

    # Categorical summaries (top 5 categories)
    categorical_df = df.select_dtypes(exclude=[np.number])
    categorical_summary = {}
    for col in categorical_df.columns:
        top_cats = categorical_df[col].value_counts().head(5).to_dict()
        categorical_summary[col] = {
            "unique_count": int(categorical_df[col].nunique()),
            "top_categories": {str(k): int(v) for k, v in top_cats.items()}
        }

    # Review columns present
    present_identifiers = [c for c in IDENTIFIER_COLUMNS if c in df.columns]
    present_leakage = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    present_post_outcome = [c for c in POST_OUTCOME_REVIEW_COLUMNS if c in df.columns]

    return {
        "shape": {"rows": total_rows, "columns": total_cols},
        "column_names": list(df.columns),
        "columns_summary": col_summary,
        "duplicate_rows": duplicate_count,
        "target_distribution": target_info,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "review_identifiers": present_identifiers,
        "review_leakage": present_leakage,
        "review_post_outcome": present_post_outcome
    }


def print_inspection_report(report: Dict[str, Any]) -> None:
    """Formats and prints the inspection report to stdout."""
    print("=" * 80)
    print(" AUTORISK AI — VEHICLE INSURANCE DATA INSPECTION REPORT")
    print("=" * 80)

    rows = report["shape"]["rows"]
    cols = report["shape"]["columns"]
    print(f"\n1. DATASET DIMENSIONS: {rows:,} rows, {cols} columns")
    print(f"   Duplicate Rows: {report['duplicate_rows']}")

    print("\n2. TARGET DISTRIBUTION (FraudFound_P):")
    t_info = report["target_distribution"]
    if "counts" in t_info:
        for cls, count in t_info["counts"].items():
            pct = t_info["percentages"].get(cls, 0.0)
            label = "Fraudulent (1)" if str(cls) == "1" else "Non-Fraudulent / Legitimate (0)"
            print(f"   - Class {cls} [{label}]: {count:,} claims ({pct:.2f}%)")
        print(f"   Class Imbalance Ratio: ~{t_info['imbalance_ratio']} : 1 (Class 0 to Class 1)")
        print("   NOTE: The target measures historical claim fraud classification.")
        print("   It does NOT predict accident rates or vehicle mechanical breakdowns.")
    else:
        print(f"   {t_info.get('error')}")

    print("\n3. COLUMN TYPES & MISSING VALUES:")
    print(f"   {'Column Name':<25} {'Dtype':<10} {'Nulls':<8} {'Null%':<8} {'Unique':<8}")
    print("   " + "-" * 65)
    for c in report["columns_summary"]:
        print(f"   {c['column']:<25} {c['dtype']:<10} {c['null_count']:<8} {c['null_pct']:<8} {c['nunique']:<8}")

    print("\n4. IDENTIFIER, LEAKAGE & POST-OUTCOME REVIEW:")
    print(f"   Identifiers to Exclude: {report['review_identifiers']}")
    print("   - PolicyNumber: Sequential index 1..15420. Memorization/leakage risk.")
    print("   - RepNumber: Internal claim handler ID (operational staff routing).")
    print(f"   Temporal Leakage Columns: {report['review_leakage']}")
    print("   - Year: Historical collection window (1994-1996), ungeneralizable.")
    print("   - MonthClaimed / DayOfWeekClaimed: Redundant reporting timestamps with data artifacts ('0').")
    print(f"   Claim Adjudication Attributes Reviewed: {report['review_post_outcome']}")
    print("   - Note: Features like NumberOfSuppliments occur during post-claim repair appraisal.")

    print("\n5. NOTABLE NUMERIC SUMMARY:")
    for col, stat in report["numeric_summary"].items():
        if col not in [TARGET_COLUMN, "PolicyNumber", "RepNumber"]:
            print(f"   - {col}: range=[{stat['min']}, {stat['max']}], mean={stat['mean']}, zeros={stat['zeros_count']}")
            if col == "Age" and stat["zeros_count"] > 0:
                print(f"     * Warning: Age contains {stat['zeros_count']} zero values (missing/unrecorded driver age).")

    print("\n=" * 80)
    print(" END OF INSPECTION REPORT")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect Vehicle Insurance Fraud Dataset")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/fraud_oracle.csv",
        help="Path to fraud_oracle.csv file (default: data/fraud_oracle.csv)"
    )
    args = parser.parse_args()

    try:
        df = load_dataset(args.data_path)
        report = inspect_dataset(df)
        print_inspection_report(report)
    except Exception as err:
        print(f"\n[ERROR] Data inspection failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
