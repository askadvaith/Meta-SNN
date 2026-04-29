"""
Metrics Utilities for Meta-Learning Experiments

Provides comprehensive classification metrics including:
- Accuracy
- Cross-entropy Loss
- Precision (macro, weighted, per-class)
- Recall (macro, weighted, per-class)
- F1 Score (macro, weighted, per-class)
- Confusion Matrix
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    y_true: Union[np.ndarray, torch.Tensor, List],
    y_pred: Union[np.ndarray, torch.Tensor, List],
    logits: Optional[torch.Tensor] = None,
    class_names: Optional[List[str]] = None,
    average: str = 'weighted'
) -> Dict:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (1D array of integers)
        y_pred: Predicted labels (1D array of integers)
        logits: Optional logits tensor for cross-entropy loss calculation
        class_names: Optional list of class names for reporting
        average: Averaging strategy for precision/recall/f1 ('micro', 'macro', 'weighted')
        
    Returns:
        Dictionary containing all metrics:
        - accuracy: Overall classification accuracy
        - loss: Cross-entropy loss (if logits provided)
        - precision_weighted: Weighted precision
        - precision_macro: Macro-averaged precision
        - recall_weighted: Weighted recall
        - recall_macro: Macro-averaged recall
        - f1_weighted: Weighted F1 score
        - f1_macro: Macro-averaged F1 score
        - precision_per_class: Per-class precision scores
        - recall_per_class: Per-class recall scores
        - f1_per_class: Per-class F1 scores
        - confusion_matrix: Confusion matrix as 2D array
        - classification_report: Detailed text report
    """
    # Convert to numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Get unique classes
    all_classes = np.unique(np.concatenate([y_true, y_pred]))
    num_classes = len(all_classes)
    
    # Basic accuracy
    accuracy = float(np.mean(y_true == y_pred))
    
    metrics = {
        'accuracy': accuracy,
        'num_samples': len(y_true),
        'num_classes': num_classes
    }
    
    # Cross-entropy loss (if logits provided)
    if logits is not None:
        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        if isinstance(y_true, np.ndarray):
            labels_tensor = torch.LongTensor(y_true)
        else:
            labels_tensor = y_true
        
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            loss = criterion(logits, labels_tensor).item()
        metrics['loss'] = loss
    
    # Precision scores
    metrics['precision_weighted'] = float(precision_score(
        y_true, y_pred, average='weighted', zero_division=0.0
    ))
    metrics['precision_macro'] = float(precision_score(
        y_true, y_pred, average='macro', zero_division=0.0
    ))
    metrics['precision_per_class'] = precision_score(
        y_true, y_pred, average=None, zero_division=0.0
    ).tolist()
    
    # Recall scores
    metrics['recall_weighted'] = float(recall_score(
        y_true, y_pred, average='weighted', zero_division=0.0
    ))
    metrics['recall_macro'] = float(recall_score(
        y_true, y_pred, average='macro', zero_division=0.0
    ))
    metrics['recall_per_class'] = recall_score(
        y_true, y_pred, average=None, zero_division=0.0
    ).tolist()
    
    # F1 scores
    metrics['f1_weighted'] = float(f1_score(
        y_true, y_pred, average='weighted', zero_division=0.0
    ))
    metrics['f1_macro'] = float(f1_score(
        y_true, y_pred, average='macro', zero_division=0.0
    ))
    metrics['f1_per_class'] = f1_score(
        y_true, y_pred, average=None, zero_division=0.0
    ).tolist()
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)
    metrics['confusion_matrix'] = cm.tolist()
    
    # Classification report (text)
    if class_names is not None and len(class_names) >= num_classes:
        target_names = [class_names[i] for i in all_classes]
    else:
        target_names = [f"Class_{i}" for i in all_classes]
    
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0.0
    )
    
    return metrics


def print_metrics_summary(metrics: Dict, title: str = "Evaluation Metrics"):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        title: Title for the summary
    """
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    print(f"\n📊 Overall Performance:")
    print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    if 'loss' in metrics:
        print(f"   Loss:      {metrics['loss']:.4f}")
    print(f"   Samples:   {metrics['num_samples']}")
    print(f"   Classes:   {metrics['num_classes']}")
    
    print(f"\n📈 Aggregated Metrics:")
    print(f"   {'Metric':<12} {'Weighted':>10} {'Macro':>10}")
    print(f"   {'-'*12} {'-'*10} {'-'*10}")
    print(f"   {'Precision':<12} {metrics['precision_weighted']:>10.4f} {metrics['precision_macro']:>10.4f}")
    print(f"   {'Recall':<12} {metrics['recall_weighted']:>10.4f} {metrics['recall_macro']:>10.4f}")
    print(f"   {'F1 Score':<12} {metrics['f1_weighted']:>10.4f} {metrics['f1_macro']:>10.4f}")
    
    print(f"\n📋 Per-Class Metrics:")
    num_classes = len(metrics['precision_per_class'])
    print(f"   {'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"   {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for i in range(num_classes):
        print(f"   {f'Class {i}':<10} {metrics['precision_per_class'][i]:>10.4f} "
              f"{metrics['recall_per_class'][i]:>10.4f} {metrics['f1_per_class'][i]:>10.4f}")
    
    print(f"\n🔢 Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    # Print header
    header = "   " + "".join([f"{f'P{i}':>6}" for i in range(cm.shape[1])])
    print(f"   {'':6}" + header[9:])
    print(f"   {'':6}" + "-" * (6 * cm.shape[1]))
    # Print rows
    for i, row in enumerate(cm):
        row_str = "".join([f"{v:>6}" for v in row])
        print(f"   {f'T{i}':<6}|{row_str}")
    print(f"   (T=True, P=Predicted)")
    
    print("\n" + "=" * 60)


def aggregate_task_metrics(task_metrics_list: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple tasks.
    
    Args:
        task_metrics_list: List of metric dictionaries from multiple tasks
        
    Returns:
        Aggregated metrics with mean and std for each metric
    """
    if not task_metrics_list:
        return {}
    
    # Keys to aggregate
    numeric_keys = [
        'accuracy', 'loss',
        'precision_weighted', 'precision_macro',
        'recall_weighted', 'recall_macro',
        'f1_weighted', 'f1_macro'
    ]
    
    aggregated = {}
    
    for key in numeric_keys:
        values = [m[key] for m in task_metrics_list if key in m]
        if values:
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
            aggregated[f'{key}_min'] = float(np.min(values))
            aggregated[f'{key}_max'] = float(np.max(values))
    
    aggregated['num_tasks'] = len(task_metrics_list)
    aggregated['total_samples'] = sum(m.get('num_samples', 0) for m in task_metrics_list)
    
    return aggregated


def print_aggregated_metrics(aggregated: Dict, title: str = "Aggregated Metrics"):
    """
    Print a formatted summary of aggregated metrics across tasks.
    
    Args:
        aggregated: Dictionary from aggregate_task_metrics()
        title: Title for the summary
    """
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    
    print(f"\n📊 Summary ({aggregated.get('num_tasks', 0)} tasks, "
          f"{aggregated.get('total_samples', 0)} total samples):")
    
    print(f"\n   {'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"   {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    metrics_to_show = [
        ('Accuracy', 'accuracy'),
        ('Loss', 'loss'),
        ('Precision (W)', 'precision_weighted'),
        ('Precision (M)', 'precision_macro'),
        ('Recall (W)', 'recall_weighted'),
        ('Recall (M)', 'recall_macro'),
        ('F1 Score (W)', 'f1_weighted'),
        ('F1 Score (M)', 'f1_macro'),
    ]
    
    for display_name, key in metrics_to_show:
        mean_key = f'{key}_mean'
        if mean_key in aggregated:
            print(f"   {display_name:<20} "
                  f"{aggregated[f'{key}_mean']:>10.4f} "
                  f"{aggregated[f'{key}_std']:>10.4f} "
                  f"{aggregated[f'{key}_min']:>10.4f} "
                  f"{aggregated[f'{key}_max']:>10.4f}")
    
    print("\n" + "=" * 70)
