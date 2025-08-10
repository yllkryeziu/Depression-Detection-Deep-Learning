#!/usr/bin/env python3
"""
Calculate evaluation metrics for depression prediction results.

This script calculates UAR (Unweighted Average Recall), accuracy, F1 score,
precision, recall, and confusion matrix for both snippet-wise and patient-wise predictions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

def calculate_uar(y_true, y_pred):
    """Calculate Unweighted Average Recall (UAR)"""
    return balanced_accuracy_score(y_true, y_pred)

def calculate_all_metrics(y_true, y_pred, dataset_name=""):
    """Calculate all evaluation metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    uar = calculate_uar(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate specificity and sensitivity manually
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for class 1
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for class 0
    
    print(f"\n{'='*60}")
    print(f"{dataset_name.upper()} EVALUATION METRICS")
    print(f"{'='*60}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  UAR (Balanced Acc): {uar:.4f} ({uar*100:.2f}%)")
    print(f"  Weighted Precision: {precision:.4f}")
    print(f"  Weighted Recall:    {recall:.4f}")
    print(f"  Weighted F1-Score:  {f1:.4f}")
    
    print(f"\nPer-Class Metrics:")
    print(f"  Class 0 (Not Depressed):")
    print(f"    Precision: {precision_per_class[0]:.4f}")
    print(f"    Recall:    {recall_per_class[0]:.4f} (Specificity)")
    print(f"    F1-Score:  {f1_per_class[0]:.4f}")
    
    print(f"  Class 1 (Depressed):")
    print(f"    Precision: {precision_per_class[1]:.4f}")
    print(f"    Recall:    {recall_per_class[1]:.4f} (Sensitivity)")
    print(f"    F1-Score:  {f1_per_class[1]:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True\\Pred    0    1")
    print(f"      0    {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"      1    {cm[1,0]:4d} {cm[1,1]:4d}")
    
    print(f"\nDetailed Breakdown:")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP):  {tp}")
    print(f"  Sensitivity (TPR):    {sensitivity:.4f}")
    print(f"  Specificity (TNR):    {specificity:.4f}")
    
    # Return metrics as dictionary
    return {
        'accuracy': accuracy,
        'uar': uar,
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1': f1,
        'class0_precision': precision_per_class[0],
        'class0_recall': recall_per_class[0],
        'class0_f1': f1_per_class[0],
        'class1_precision': precision_per_class[1],
        'class1_recall': recall_per_class[1],
        'class1_f1': f1_per_class[1],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }

def load_ground_truth_for_snippets(snippet_df, patient_df):
    """Load ground truth labels for snippets by mapping from patient data"""
    # Create a mapping from patient_id to ground truth
    patient_gt_map = dict(zip(patient_df['patient_id'], patient_df['ground_truth_class']))
    
    # Map ground truth to snippets
    snippet_gt = []
    for _, row in snippet_df.iterrows():
        patient_id = row['patient_id']
        gt_class = patient_gt_map.get(patient_id, None)
        snippet_gt.append(gt_class)
    
    return snippet_gt

def main():
    # Load prediction files
    try:
        snippet_df = pd.read_csv('snippet_predictions.csv')
        patient_df = pd.read_csv('depression_predictions.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find prediction files. {e}")
        print("Please make sure 'snippet_predictions.csv' and 'depression_predictions.csv' exist.")
        return
    
    print("Loaded prediction files successfully!")
    print(f"Snippet predictions: {len(snippet_df)} samples")
    print(f"Patient predictions: {len(patient_df)} patients")
    
    # Calculate snippet-wise metrics
    print("\nCalculating snippet-wise metrics...")
    snippet_gt = load_ground_truth_for_snippets(snippet_df, patient_df)
    snippet_pred = snippet_df['predicted_class'].tolist()
    
    # Remove any None values (in case of missing ground truth)
    valid_indices = [i for i, gt in enumerate(snippet_gt) if gt is not None]
    snippet_gt_clean = [snippet_gt[i] for i in valid_indices]
    snippet_pred_clean = [snippet_pred[i] for i in valid_indices]
    
    snippet_metrics = calculate_all_metrics(
        snippet_gt_clean, snippet_pred_clean, "Snippet-wise"
    )
    
    # Calculate patient-wise metrics
    print("\nCalculating patient-wise metrics...")
    patient_gt = patient_df['ground_truth_class'].tolist()
    patient_pred = patient_df['predicted_class'].tolist()
    
    patient_metrics = calculate_all_metrics(
        patient_gt, patient_pred, "Patient-wise"
    )
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    
    print(f"\n{'Metric':<20} {'Snippet-wise':<15} {'Patient-wise':<15}")
    print(f"{'-'*50}")
    print(f"{'Accuracy':<20} {snippet_metrics['accuracy']:<15.4f} {patient_metrics['accuracy']:<15.4f}")
    print(f"{'UAR':<20} {snippet_metrics['uar']:<15.4f} {patient_metrics['uar']:<15.4f}")
    print(f"{'Weighted F1':<20} {snippet_metrics['weighted_f1']:<15.4f} {patient_metrics['weighted_f1']:<15.4f}")
    print(f"{'Sensitivity':<20} {snippet_metrics['sensitivity']:<15.4f} {patient_metrics['sensitivity']:<15.4f}")
    print(f"{'Specificity':<20} {snippet_metrics['specificity']:<15.4f} {patient_metrics['specificity']:<15.4f}")
    
    # Class distribution
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*60}")
    
    print(f"\nSnippet-wise:")
    snippet_class_counts = pd.Series(snippet_gt_clean).value_counts().sort_index()
    for class_label, count in snippet_class_counts.items():
        class_name = "Not Depressed" if class_label == 0 else "Depressed"
        percentage = count / len(snippet_gt_clean) * 100
        print(f"  Class {class_label} ({class_name}): {count} samples ({percentage:.1f}%)")
    
    print(f"\nPatient-wise:")
    patient_class_counts = pd.Series(patient_gt).value_counts().sort_index()
    for class_label, count in patient_class_counts.items():
        class_name = "Not Depressed" if class_label == 0 else "Depressed"
        percentage = count / len(patient_gt) * 100
        print(f"  Class {class_label} ({class_name}): {count} patients ({percentage:.1f}%)")
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 