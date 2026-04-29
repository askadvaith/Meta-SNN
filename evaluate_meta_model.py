#!/usr/bin/env python3
"""
Meta-Model Evaluation Script

Interactive script to:
1. Select a trained meta-model from the results folder
2. Select one or more datasets from the data folder
3. Adapt the meta-model to selected datasets
4. Report comprehensive metrics (accuracy, loss, precision, recall, F1, confusion matrix)

Usage:
    python evaluate_meta_model.py
    python evaluate_meta_model.py --experiment <experiment_name> --datasets beef wine
    python evaluate_meta_model.py --experiment <experiment_name> --all-datasets
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.meta_snn import create_meta_snn
from training.reptile import ReptileMetaLearner
from utils.data_loader import UnifiedOlfactoryDataLoader, TaskData, create_meta_learning_tasks
from utils.metrics import calculate_metrics, print_metrics_summary, aggregate_task_metrics, print_aggregated_metrics


def get_available_experiments(results_dir: str = "results") -> List[str]:
    """Get list of available experiment directories with best_meta_model.pth"""
    experiments = []
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return experiments
    
    for item in sorted(os.listdir(results_dir)):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            model_path = os.path.join(item_path, "models", "best_meta_model.pth")
            if os.path.exists(model_path):
                experiments.append(item)
    
    return experiments


def get_available_datasets(data_dir: str = "data") -> List[str]:
    """Get list of available dataset directories"""
    datasets = []
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return datasets
    
    for item in sorted(os.listdir(data_dir)):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            datasets.append(item)
    
    return datasets


def display_menu(title: str, options: List[str], allow_multiple: bool = False) -> List[int]:
    """Display interactive menu and get user selection(s)"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)
    
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    
    if allow_multiple:
        print(f"\n  [0] Select all")
        print(f"\nEnter choice(s) separated by commas (e.g., 1,3,4) or 0 for all:")
    else:
        print(f"\nEnter your choice:")
    
    while True:
        try:
            user_input = input("> ").strip()
            
            if allow_multiple and user_input == "0":
                return list(range(len(options)))
            
            indices = [int(x.strip()) - 1 for x in user_input.split(",")]
            
            # Validate indices
            for idx in indices:
                if idx < 0 or idx >= len(options):
                    raise ValueError(f"Invalid selection: {idx + 1}")
            
            if not allow_multiple and len(indices) > 1:
                print("Please select only one option.")
                continue
            
            return indices
            
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")


def load_experiment_config(experiment_dir: str) -> Dict:
    """Load experiment configuration"""
    config_path = os.path.join(experiment_dir, "config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Return default config if not found
    return {
        "model": {
            "input_size": 10,
            "hidden_sizes": [64, 32],
            "beta": 0.9
        },
        "meta_learning": {
            "inner_lr": 0.001,
            "inner_steps": 5
        },
        "data": {
            "sequence_length": 500,
            "downsample_factor": 2
        }
    }


def load_meta_model(
    experiment_dir: str,
    config: Dict,
    device: torch.device
) -> ReptileMetaLearner:
    """Load the best meta-model from an experiment directory"""
    model_path = os.path.join(experiment_dir, "models", "best_meta_model.pth")
    
    if not os.path.exists(model_path):
        # Try meta_model_final.pth as fallback
        model_path = os.path.join(experiment_dir, "models", "meta_model_final.pth")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found in {experiment_dir}/models/")
    
    # Load checkpoint first to determine output size
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Detect output size from saved weights
    output_size = 3  # default
    if 'model_state_dict' in checkpoint:
        head_weight_key = 'head_fc.weight'
        if head_weight_key in checkpoint['model_state_dict']:
            output_size = checkpoint['model_state_dict'][head_weight_key].shape[0]
    
    # Create model with config parameters
    model_config = config.get("model", {})
    meta_config = config.get("meta_learning", {})
    
    model = create_meta_snn(
        input_size=model_config.get("input_size", 10),
        hidden_sizes=model_config.get("hidden_sizes", [64, 32]),
        output_size=output_size,  # Use detected output size
        beta=model_config.get("beta", 0.9),
        device=device
    )
    
    learner = ReptileMetaLearner(
        model=model,
        meta_lr=meta_config.get("meta_lr", 0.1),
        inner_lr=meta_config.get("inner_lr", 0.001),
        inner_steps=meta_config.get("inner_steps", 5),
        device=device
    )
    
    # Load the trained weights (checkpoint already loaded)
    learner.model.load_state_dict(checkpoint['model_state_dict'])
    learner.meta_lr = checkpoint.get('meta_lr', learner.meta_lr)
    learner.inner_lr = checkpoint.get('inner_lr', learner.inner_lr)
    learner.inner_steps = checkpoint.get('inner_steps', learner.inner_steps)
    learner.history = checkpoint.get('history', learner.history)
    
    print(f"✓ Loaded model from: {model_path}")
    print(f"  Output size: {output_size}")
    
    return learner


def load_dataset_tasks(
    dataset_names: List[str],
    data_root: str,
    config: Dict
) -> Tuple[List[TaskData], Dict[str, List[TaskData]]]:
    """Load tasks for specified datasets"""
    data_config = config.get("data", {})
    
    loader = UnifiedOlfactoryDataLoader(
        data_root=data_root,
        sequence_length=data_config.get("sequence_length", 500),
        downsample_factor=data_config.get("downsample_factor", 2),
        max_features=10,
        normalize=True
    )
    
    # Load all tasks
    all_domain_tasks = loader.load_all_tasks()
    
    # Filter to requested datasets
    filtered_tasks = []
    tasks_by_domain = {}
    
    for domain, tasks in all_domain_tasks.items():
        if domain in dataset_names:
            filtered_tasks.extend(tasks)
            tasks_by_domain[domain] = tasks
    
    return filtered_tasks, tasks_by_domain


def evaluate_on_task(
    learner: ReptileMetaLearner,
    task: TaskData,
    adaptation_steps: int = 10,
    train_ratio: float = 0.8
) -> Dict:
    """Evaluate the meta-model on a single task with train/test split"""
    
    # Use adapt_and_evaluate which does train/test split
    adapted_model, metrics = learner.adapt_and_evaluate(
        task=task,
        train_ratio=train_ratio,
        adaptation_steps=adaptation_steps
    )
    
    return metrics


def save_evaluation_results(
    results: Dict,
    output_dir: str,
    experiment_name: str,
    datasets: List[str]
):
    """Save evaluation results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Make results JSON serializable
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return float(obj) if isinstance(obj, (np.float64, np.float32)) else int(obj)
        else:
            return obj
    
    serializable_results = make_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a meta-trained model on selected datasets"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Experiment name (subdirectory in results/)"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        type=str,
        help="Dataset names to evaluate on (subdirectories in data/)"
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Evaluate on all available datasets"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Results directory (default: results)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--adaptation-steps",
        type=int,
        default=10,
        help="Number of adaptation steps (default: 10)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training ratio for train/test split (default: 0.8)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'cpu', 'cuda', or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: experiment's metrics folder)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"\n🔧 Using device: {device}")
    
    # Get available experiments
    experiments = get_available_experiments(args.results_dir)
    if not experiments:
        print("❌ No experiments with saved models found in results directory.")
        return
    
    # Select experiment
    if args.experiment:
        if args.experiment in experiments:
            selected_experiment = args.experiment
        else:
            print(f"❌ Experiment '{args.experiment}' not found.")
            print(f"Available experiments: {experiments}")
            return
    else:
        indices = display_menu("Select Experiment", experiments, allow_multiple=False)
        selected_experiment = experiments[indices[0]]
    
    experiment_dir = os.path.join(args.results_dir, selected_experiment)
    print(f"\n📁 Selected experiment: {selected_experiment}")
    
    # Load experiment config
    config = load_experiment_config(experiment_dir)
    
    # Get available datasets
    datasets = get_available_datasets(args.data_dir)
    if not datasets:
        print("❌ No datasets found in data directory.")
        return
    
    # Select datasets
    if args.all_datasets:
        selected_datasets = datasets
    elif args.datasets:
        selected_datasets = [d for d in args.datasets if d in datasets]
        invalid = [d for d in args.datasets if d not in datasets]
        if invalid:
            print(f"⚠️ Warning: Invalid dataset names ignored: {invalid}")
        if not selected_datasets:
            print("❌ No valid datasets selected.")
            return
    else:
        indices = display_menu("Select Dataset(s)", datasets, allow_multiple=True)
        selected_datasets = [datasets[i] for i in indices]
    
    print(f"\n📊 Selected datasets: {selected_datasets}")
    
    # Load the meta-model
    print("\n⏳ Loading meta-model...")
    try:
        learner = load_meta_model(experiment_dir, config, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load tasks for selected datasets
    print("\n⏳ Loading datasets...")
    try:
        all_tasks, tasks_by_domain = load_dataset_tasks(
            selected_datasets, args.data_dir, config
        )
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        return
    
    if not all_tasks:
        print("❌ No tasks loaded from selected datasets.")
        return
    
    print(f"✓ Loaded {len(all_tasks)} tasks from {len(tasks_by_domain)} datasets")
    
    # Evaluate on each task
    print(f"\n{'='*70}")
    print(" EVALUATION")
    print('='*70)
    print(f"\nAdaptation Steps: {args.adaptation_steps}")
    print(f"Train/Test Split: {args.train_ratio:.0%} / {(1-args.train_ratio):.0%}")
    
    all_results = {
        "experiment": selected_experiment,
        "datasets": selected_datasets,
        "config": {
            "adaptation_steps": args.adaptation_steps,
            "train_ratio": args.train_ratio,
            "device": str(device)
        },
        "task_results": [],
        "domain_aggregated": {},
        "overall_aggregated": {}
    }
    
    all_task_metrics = []
    domain_metrics = {domain: [] for domain in tasks_by_domain.keys()}
    
    for domain, tasks in tasks_by_domain.items():
        print(f"\n{'─'*70}")
        print(f" Domain: {domain.upper()} ({len(tasks)} tasks)")
        print('─'*70)
        
        for task in tasks:
            print(f"\n📋 Task: {task.name}")
            print(f"   Samples: {len(task.X)}, Classes: {task.num_classes}")
            
            try:
                metrics = evaluate_on_task(
                    learner, task,
                    adaptation_steps=args.adaptation_steps,
                    train_ratio=args.train_ratio
                )
                
                # Print summary
                print(f"\n   📈 Results:")
                print(f"      Train - Loss: {metrics['train_loss']:.4f}, "
                      f"Acc: {metrics['train_accuracy']:.4f}")
                print(f"      Test  - Loss: {metrics['test_loss']:.4f}, "
                      f"Acc: {metrics['test_accuracy']:.4f}")
                print(f"      Test  - P: {metrics['test_precision']:.4f}, "
                      f"R: {metrics['test_recall']:.4f}, "
                      f"F1: {metrics['test_f1']:.4f}")
                
                # Store task result
                task_result = {
                    "task_name": task.name,
                    "domain": domain,
                    "num_samples": len(task.X),
                    "num_classes": task.num_classes,
                    "class_names": task.class_names,
                    "metrics": {
                        "train_loss": metrics['train_loss'],
                        "train_accuracy": metrics['train_accuracy'],
                        "train_precision": metrics['train_precision'],
                        "train_recall": metrics['train_recall'],
                        "train_f1": metrics['train_f1'],
                        "test_loss": metrics['test_loss'],
                        "test_accuracy": metrics['test_accuracy'],
                        "test_precision": metrics['test_precision'],
                        "test_recall": metrics['test_recall'],
                        "test_f1": metrics['test_f1'],
                        "test_precision_macro": metrics['test_precision_macro'],
                        "test_recall_macro": metrics['test_recall_macro'],
                        "test_f1_macro": metrics['test_f1_macro'],
                        "test_confusion_matrix": metrics['test_confusion_matrix']
                    }
                }
                all_results["task_results"].append(task_result)
                
                # Collect for aggregation
                test_metrics = {
                    'accuracy': metrics['test_accuracy'],
                    'loss': metrics['test_loss'],
                    'precision_weighted': metrics['test_precision'],
                    'recall_weighted': metrics['test_recall'],
                    'f1_weighted': metrics['test_f1'],
                    'precision_macro': metrics['test_precision_macro'],
                    'recall_macro': metrics['test_recall_macro'],
                    'f1_macro': metrics['test_f1_macro'],
                    'num_samples': len(task.X)
                }
                all_task_metrics.append(test_metrics)
                domain_metrics[domain].append(test_metrics)
                
            except Exception as e:
                print(f"   ❌ Evaluation failed: {e}")
                continue
    
    # Aggregate metrics per domain
    print(f"\n{'='*70}")
    print(" AGGREGATED RESULTS BY DOMAIN")
    print('='*70)
    
    for domain, metrics_list in domain_metrics.items():
        if metrics_list:
            aggregated = aggregate_task_metrics(metrics_list)
            all_results["domain_aggregated"][domain] = aggregated
            print_aggregated_metrics(aggregated, f"{domain.upper()} Domain Summary")
    
    # Overall aggregation
    if all_task_metrics:
        overall_aggregated = aggregate_task_metrics(all_task_metrics)
        all_results["overall_aggregated"] = overall_aggregated
        print_aggregated_metrics(overall_aggregated, "OVERALL SUMMARY")
    
    # Save results
    if not args.no_save:
        output_dir = args.output_dir or os.path.join(experiment_dir, "metrics")
        save_evaluation_results(
            all_results, output_dir, selected_experiment, selected_datasets
        )
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
