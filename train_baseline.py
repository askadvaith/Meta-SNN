"""
Baseline SNN Training Script

Train a standard (non-meta-learning) SNN on any single olfactory dataset
to establish baseline performance for comparison with meta-learning approaches.

Usage:
    # List available datasets
    python train_baseline.py --list
    
    # Train on a specific dataset (e.g., wine)
    python train_baseline.py --dataset wine --epochs 100
    
    # Train on a specific task within a domain
    python train_baseline.py --dataset beef --task beef_Brisket --epochs 100
    
    # Train with custom parameters
    python train_baseline.py --dataset vinegar --epochs 200 --lr 0.001 --batch_size 32
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import create_meta_learning_tasks, TaskData
from models.baselines.baseline_snn import BaselineSNN, create_baseline_snn


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Baseline SNN on Olfactory Data'
    )
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['beef', 'seafood', 'vinegar', 'wine', 'all'],
                        help='Dataset domain to train on')
    parser.add_argument('--task', type=str, default=None,
                        help='Specific task name to train on (e.g., beef_Brisket)')
    parser.add_argument('--list', action='store_true',
                        help='List all available datasets and tasks')
    
    # Data
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory containing datasets')
    parser.add_argument('--sequence_length', type=int, default=500,
                        help='Target sequence length after processing')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Downsampling factor for time series')
    
    # Model
    parser.add_argument('--input_size', type=int, default=10,
                        help='Number of input features')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 64],
                        help='Hidden layer sizes')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Membrane potential decay rate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Output
    parser.add_argument('--results_dir', type=str, default='results/baselines',
                        help='Directory for saving results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), auto-detected if None')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_available_tasks(data_root: str):
    """List all available datasets and tasks"""
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS AND TASKS")
    print("=" * 70)
    
    all_tasks, tasks_by_domain = create_meta_learning_tasks(
        data_root=data_root,
        sequence_length=500,
        downsample_factor=2
    )
    
    for domain, domain_tasks in tasks_by_domain.items():
        print(f"\n{domain.upper()} ({len(domain_tasks)} tasks):")
        print("-" * 50)
        for task in domain_tasks:
            print(f"  - {task.name}")
            print(f"      Samples: {task.X.shape[0]}, Classes: {task.num_classes}")
            print(f"      Classes: {task.class_names}")
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES:")
    print("=" * 70)
    print("  # Train on all wine tasks combined:")
    print("  python train_baseline.py --dataset wine --epochs 100")
    print("")
    print("  # Train on a specific beef task:")
    print("  python train_baseline.py --dataset beef --task beef_Brisket --epochs 100")
    print("")


def get_tasks_for_training(
    args,
    all_tasks: List[TaskData],
    tasks_by_domain: Dict[str, List[TaskData]]
) -> List[TaskData]:
    """Get tasks based on command line arguments"""
    
    if args.task:
        # Specific task requested
        matching_tasks = [t for t in all_tasks if t.name == args.task]
        if not matching_tasks:
            print(f"Error: Task '{args.task}' not found!")
            print("Use --list to see available tasks.")
            sys.exit(1)
        return matching_tasks
    
    elif args.dataset:
        if args.dataset == 'all':
            return all_tasks
        elif args.dataset in tasks_by_domain:
            return tasks_by_domain[args.dataset]
        else:
            print(f"Error: Dataset '{args.dataset}' not found!")
            sys.exit(1)
    
    else:
        print("Error: Please specify --dataset or --task, or use --list to see options.")
        sys.exit(1)


def combine_tasks(tasks: List[TaskData]) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """
    Combine multiple tasks into a single dataset.
    
    If tasks have different numbers of classes, we merge them and create
    a unified label space.
    
    Returns:
        X: Combined features
        y: Combined labels (remapped to unified space)
        class_names: Unified class names
        num_classes: Total number of classes
    """
    if len(tasks) == 1:
        task = tasks[0]
        return task.X, task.y, task.class_names, task.num_classes
    
    # For multiple tasks, we need to create a unified label space
    # Each task's classes become separate in the combined dataset
    all_X = []
    all_y = []
    class_names = []
    label_offset = 0
    
    for task in tasks:
        all_X.append(task.X)
        # Remap labels to unified space
        remapped_y = task.y + label_offset
        all_y.append(remapped_y)
        
        # Add class names with task prefix to avoid confusion
        for cn in task.class_names:
            class_names.append(f"{task.name}:{cn}")
        
        label_offset += task.num_classes
    
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    return X, y, class_names, label_offset


def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    val_split: float,
    seed: int
) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation data loaders"""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_split, random_state=seed, stratify=y
    )
    
    # Convert to tensors
    # Input format: [samples, timesteps, features] -> [timesteps, samples, features]
    X_train_t = torch.FloatTensor(X_train).permute(1, 0, 2)
    X_val_t = torch.FloatTensor(X_val).permute(1, 0, 2)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)
    
    # Create datasets - note we need to handle the time dimension
    # We'll create custom collate function
    train_dataset = TensorDataset(X_train_t.permute(1, 0, 2), y_train_t)
    val_dataset = TensorDataset(X_val_t.permute(1, 0, 2), y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def train_epoch(
    model: BaselineSNN,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        # Move to device and permute to [timesteps, batch, features]
        batch_X = batch_X.permute(1, 0, 2).to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        mem_out, spk_out, _ = model(batch_X, return_recordings=False)
        
        # Get logits from membrane potentials
        logits = model.get_output_logits(mem_out)
        
        # Compute loss
        loss = criterion(logits, batch_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(
    model: BaselineSNN,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            # Move to device and permute to [timesteps, batch, features]
            batch_X = batch_X.permute(1, 0, 2).to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            mem_out, spk_out, _ = model(batch_X, return_recordings=False)
            
            # Get logits
            logits = model.get_output_logits(mem_out)
            
            # Compute loss
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            
            # Track predictions
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)


def train_baseline(
    model: BaselineSNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    save_dir: str
) -> Dict:
    """
    Full training loop with early stopping.
    
    Returns:
        Dictionary with training history and final metrics
    """
    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'best_epoch': 0,
        'best_val_acc': 0.0
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("TRAINING BASELINE SNN")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch Size: {args.batch_size}")
    print("-" * 70)
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, os.path.join(save_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
                  f"Best: {best_val_acc:.4f}")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break
    
    print("-" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f} at epoch {history['best_epoch']+1}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(save_dir, 'final_model.pt'))
    
    return history


def compute_final_metrics(
    model: BaselineSNN,
    val_loader: DataLoader,
    device: torch.device,
    class_names: List[str]
) -> Dict:
    """Compute comprehensive final metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.permute(1, 0, 2).to(device)
            batch_y = batch_y.to(device)
            
            mem_out, spk_out, _ = model(batch_X, return_recordings=False)
            logits = model.get_output_logits(mem_out)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist(),
        'per_class': {}
    }
    
    # Add per-class metrics
    for i, cn in enumerate(class_names[:len(per_class_precision)]):
        metrics['per_class'][cn] = {
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'f1_score': float(per_class_f1[i]),
            'support': int(support[i])
        }
    
    return metrics


def save_training_plots(history: Dict, save_dir: str):
    """Save training history plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].axvline(x=history['best_epoch'], color='r', linestyle='--', label='Best Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150)
    plt.close()


def save_confusion_matrix_plot(conf_matrix: np.ndarray, class_names: List[str], save_dir: str):
    """Save confusion matrix visualization"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks
    ax.set(xticks=np.arange(conf_matrix.shape[1]),
           yticks=np.arange(conf_matrix.shape[0]),
           xticklabels=class_names[:conf_matrix.shape[1]],
           yticklabels=class_names[:conf_matrix.shape[0]],
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def main():
    """Main training function"""
    args = parse_args()
    
    # List tasks if requested
    if args.list:
        list_available_tasks(args.data_root)
        return
    
    # Validate arguments
    if not args.dataset and not args.task:
        print("Error: Please specify --dataset or --task, or use --list to see options.")
        sys.exit(1)
    
    # Set seed
    set_seed(args.seed)
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 70)
    print("BASELINE SNN TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading datasets...")
    all_tasks, tasks_by_domain = create_meta_learning_tasks(
        data_root=args.data_root,
        sequence_length=args.sequence_length,
        downsample_factor=args.downsample_factor
    )
    
    # Get tasks to train on
    tasks = get_tasks_for_training(args, all_tasks, tasks_by_domain)
    
    task_names = [t.name for t in tasks]
    print(f"\nTraining on {len(tasks)} task(s): {task_names}")
    
    # Combine tasks if multiple
    X, y, class_names, num_classes = combine_tasks(tasks)
    print(f"Combined dataset: {X.shape[0]} samples, {num_classes} classes")
    print(f"Classes: {class_names}")
    
    # Create data loaders
    train_loader, val_loader = prepare_data_loaders(
        X, y, args.batch_size, args.val_split, args.seed
    )
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.task if args.task else args.dataset
    exp_name = args.experiment_name or f"baseline_{dataset_name}_{timestamp}"
    save_dir = os.path.join(args.results_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    
    print(f"Saving results to: {save_dir}")
    
    # Create model
    model = create_baseline_snn(
        input_size=args.input_size,
        hidden_sizes=args.hidden_sizes,
        output_size=num_classes,
        beta=args.beta,
        dropout=args.dropout,
        device=device
    )
    
    print(f"\nModel Architecture:")
    print(f"  Input: {args.input_size}")
    print(f"  Hidden: {args.hidden_sizes}")
    print(f"  Output: {num_classes}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save config
    config = {
        'dataset': args.dataset,
        'task': args.task,
        'tasks_used': task_names,
        'num_classes': num_classes,
        'class_names': class_names,
        'total_samples': X.shape[0],
        'sequence_length': X.shape[1],
        'input_size': args.input_size,
        'hidden_sizes': args.hidden_sizes,
        'beta': args.beta,
        'dropout': args.dropout,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'val_split': args.val_split,
        'patience': args.patience,
        'seed': args.seed,
        'device': str(device),
        'timestamp': timestamp
    }
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train
    history = train_baseline(
        model, train_loader, val_loader, args, device,
        os.path.join(save_dir, 'models')
    )
    
    # Load best model for final evaluation
    best_checkpoint = torch.load(os.path.join(save_dir, 'models', 'best_model.pt'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Compute final metrics
    print("\nComputing final metrics...")
    final_metrics = compute_final_metrics(model, val_loader, device, class_names)
    
    # Save metrics
    with open(os.path.join(save_dir, 'metrics', 'final_metrics.json'), 'w') as f:
        json.dump(final_metrics, f, indent=4)
    
    with open(os.path.join(save_dir, 'metrics', 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save plots
    save_training_plots(history, os.path.join(save_dir, 'plots'))
    save_confusion_matrix_plot(
        np.array(final_metrics['confusion_matrix']),
        class_names,
        os.path.join(save_dir, 'plots')
    )
    
    # Print final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {history['best_val_acc']:.4f}")
    print(f"Final Metrics:")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1 Score:  {final_metrics['f1_score']:.4f}")
    print(f"\nResults saved to: {save_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
