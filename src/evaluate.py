"""
Model Evaluation and Analysis Module for AutoRisk AI.

Loads the persisted fraud detection pipeline, evaluates performance on test data,
reports confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, and provides
insurance underwriting trade-off explanations.
"""

import argparse
import os
import sys
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "FraudFound_P"


def load_pipeline(model_path: str):
    """Loads fitted pipeline from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact not found at '{model_path}'. Run src/train.py first.")
    return joblib.load(model_path)


def evaluate_dataset(
    model_path: str,
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """Loads pipeline and dataset, splits into test set, and calculates performance metrics."""
    pipeline = load_pipeline(model_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'.")

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target '{TARGET_COLUMN}' not found in '{data_path}'.")

    y = df[TARGET_COLUMN].values.astype(int)

    # Use the same split logic as train.py
    _, X_test, _, y_test = train_test_split(
        df, y, test_size=test_size, stratify=y, random_state=random_state
    )

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    acc = float(accuracy_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None
    pr_auc = float(average_precision_score(y_test, y_proba)) if y_proba is not None else None

    total_test = len(y_test)
    total_fraud = int((y_test == 1).sum())
    total_legit = int((y_test == 0).sum())

    results = {
        "dataset": data_path,
        "model_file": model_path,
        "test_counts": {
            "total": total_test,
            "legitimate_claims": total_legit,
            "fraudulent_claims": total_fraud,
            "fraud_prevalence_pct": round(total_fraud / total_test * 100, 2),
        },
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "raw_matrix": [[tn, fp], [fn, tp]],
        },
        "metrics": {
            "precision_fraud": round(prec, 4),
            "recall_fraud": round(rec, 4),
            "f1_fraud": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 4) if pr_auc is not None else None,
            "accuracy": round(acc, 4),
        },
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Legitimate (0)", "Fraudulent (1)"], output_dict=True
        ),
    }

    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Prints a detailed underwriting-style evaluation report."""
    cm = results["confusion_matrix"]
    m = results["metrics"]
    counts = results["test_counts"]

    print("=" * 80)
    print(" AUTORISK AI — MODEL EVALUATION & UNDERWRITING REVIEW")
    print("=" * 80)
    print(f"\nModel File: {results['model_file']}")
    print(f"Data File:  {results['dataset']}")
    print(f"Test Claims: {counts['total']:,} (Fraud: {counts['fraudulent_claims']} [{counts['fraud_prevalence_pct']}%], Legit: {counts['legitimate_claims']})")

    print("\n--- CONFUSION MATRIX ---")
    print(f"  True Negatives  (TN): {cm['true_negatives']:<6} (Legitimate claims cleared without audit)")
    print(f"  False Positives (FP): {cm['false_positives']:<6} (Legitimate claims flagged for secondary review)")
    print(f"  False Negatives (FN): {cm['false_negatives']:<6} (Fraudulent claims missed - paid without audit)")
    print(f"  True Positives  (TP): {cm['true_positives']:<6} (Fraudulent claims correctly caught for review)")

    print("\n--- EVALUATION METRICS (Fraud Class = 1) ---")
    print(f"  Recall (Fraud Class):           {m['recall_fraud'] * 100:.2f}%  ({cm['true_positives']}/{counts['fraudulent_claims']} frauds flagged)")
    print(f"  Precision (Fraud Class):        {m['precision_fraud'] * 100:.2f}%  (Probability flagged claim is truly fraudulent)")
    print(f"  F1-Score (Fraud Class):         {m['f1_fraud']:.4f}")
    print(f"  Area Under ROC (ROC-AUC):       {m['roc_auc']:.4f}  (Ranking separation across thresholds)")
    print(f"  Area Under PR Curve (PR-AUC):   {m['pr_auc']:.4f}  (Average Precision under class imbalance)")
    print(f"  Accuracy (Secondary metric):    {m['accuracy'] * 100:.2f}%")

    print("\n--- INSURANCE UNDERWRITING TRADE-OFF ANALYSIS ---")
    print("1. FALSE NEGATIVES (Missed Fraud):")
    print(f"   The model produced {cm['false_negatives']} False Negatives out of {counts['fraudulent_claims']} total frauds.")
    print("   In insurance underwriting, False Negatives represent direct financial indemnity leakage,")
    print("   as fraudulent payouts cannot easily be recovered after settlement.")
    print("\n2. FALSE POSITIVES (Review Overhead):")
    print(f"   The model produced {cm['false_positives']} False Positives.")
    print("   In practice, this routes claims to the Special Investigation Unit (SIU) or manual desk audit.")
    print("   It incurs operational triage overhead (~1 investigator hour per claim) but protects capital.")
    print("   CRITICAL: A flag is a request for documentation verification, NOT proof of fraud.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoRisk AI Model")
    parser.add_argument("--data-path", type=str, default="data/fraud_oracle.csv", help="Dataset path")
    parser.add_argument("--model-path", type=str, default="models/fraud_model.joblib", help="Model joblib path")
    args = parser.parse_args()

    try:
        results = evaluate_dataset(args.model_path, args.data_path)
        print_evaluation_report(results)
    except Exception as err:
        print(f"\n[ERROR] Evaluation failed: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
