"""
Reptile Meta-Learning Algorithm for SNNs

Implements the Reptile algorithm from "On First-Order Meta-Learning Algorithms" (Nichol et al., 2018)
with task interleaving for continual meta-learning.

Key concepts:
- Reptile: First-order meta-learning (no second derivatives)
- Task Interleaving: Sample from different domains to prevent catastrophic forgetting
- Outer loop: Update meta-parameters toward task-adapted parameters
- Inner loop: Standard SGD on task-specific data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from copy import deepcopy
from tqdm import tqdm
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from utils.data_loader import TaskData
from utils.metrics import calculate_metrics
from models.meta_snn import MetaSNN, create_meta_snn


class ReptileMetaLearner:
    """
    Reptile meta-learning algorithm for SNNs.
    
    Features:
    - Task interleaving across domains
    - Support for varying number of classes across tasks
    - Configurable inner/outer loop parameters
    - Class-weighted loss for imbalanced tasks
    """
    
    def __init__(
        self,
        model: MetaSNN,
        meta_lr: float = 0.1,
        inner_lr: float = 0.001,
        inner_steps: int = 5,
        device: torch.device = None,
        use_class_weights: bool = False
    ):
        """
        Initialize the Reptile meta-learner.
        
        Args:
            model: Base MetaSNN model (will be cloned for each task)
            meta_lr: Outer loop learning rate (Reptile step size)
            inner_lr: Inner loop learning rate for task adaptation
            inner_steps: Number of gradient steps per task
            device: Computation device
            use_class_weights: Whether to use class-weighted loss for imbalanced tasks
        """
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_class_weights = use_class_weights
        
        self.model = self.model.to(self.device)
        
        # Loss function (will be set per-task if using class weights)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'meta_loss': [],
            'task_losses': [],
            'task_accuracies': [],
            'task_precision': [],
            'task_recall': [],
            'task_f1': []
        }
    
    def _compute_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """
        Compute inverse frequency class weights for imbalanced data.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Tensor of class weights
        """
        class_counts = np.bincount(labels)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        # Inverse frequency weighting
        weights = 1.0 / class_counts
        # Normalize so weights sum to number of classes
        weights = weights / weights.sum() * len(class_counts)
        return torch.FloatTensor(weights).to(self.device)
    
    def _prepare_inputs(
        self,
        X: np.ndarray
    ) -> torch.Tensor:
        """
        Prepare inputs for the SNN (Direct Current Injection).
        
        Instead of binary thresholding, we pass continuous values (0-1)
        directly to the first layer. The first layer of LIF neurons will
        act as spike generators based on these input currents.
        
        Args:
            X: Input data [samples, timesteps, features]
            
        Returns:
            Input tensor [timesteps, samples, features]
        """
        # Convert to tensor
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        # Reshape to [timesteps, samples, features]
        # snntorch expects time-first
        inputs = X_tensor.permute(1, 0, 2)
        
        return inputs
    
    def _adapt_model_to_task(
        self,
        task: TaskData,
        num_samples: int = None
    ) -> Tuple[MetaSNN, float, float]:
        """
        Adapt the model to a specific task using inner loop optimization.
        
        Args:
            task: TaskData object containing the task
            num_samples: Number of samples to use (None = all)
            
        Returns:
            adapted_model: Model after task adaptation
            loss: Final loss on the task
            accuracy: Final accuracy on the task
        """
        # Clone the model
        adapted_model = deepcopy(self.model)
        adapted_model = adapted_model.to(self.device)
        
        # Adapt output head if necessary
        if task.num_classes != adapted_model.output_size:
            adapted_model.adapt_head(task.num_classes, self.device)
        
        # Prepare data
        X, y = task.X, task.y
        
        # Subsample if requested
        if num_samples is not None and num_samples < len(X):
            indices = np.random.choice(len(X), num_samples, replace=False)
            X, y = X[indices], y[indices]
        
        # Prepare inputs (Direct encoding)
        spikes = self._prepare_inputs(X)
        labels = torch.LongTensor(y).to(self.device)
        
        # Set up loss function with class weights if enabled
        if self.use_class_weights:
            class_weights = self._compute_class_weights(y)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = self.criterion
        
        # Inner loop optimizer
        optimizer = optim.Adam(adapted_model.parameters(), lr=self.inner_lr)
        
        # Inner loop training
        adapted_model.train()
        for step in range(self.inner_steps):
            optimizer.zero_grad()
            
            mem_out, _, _ = adapted_model(spikes, return_recordings=False)
            logits = adapted_model.get_output_logits(mem_out)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Calculate final metrics
        adapted_model.eval()
        with torch.no_grad():
            mem_out, _, _ = adapted_model(spikes, return_recordings=False)
            logits = adapted_model.get_output_logits(mem_out)
            
            final_loss = self.criterion(logits, labels).item()
            
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
            
            # Calculate comprehensive metrics
            metrics = calculate_metrics(
                y_true=labels.cpu().numpy(),
                y_pred=predictions.cpu().numpy(),
                logits=logits.cpu(),
                class_names=task.class_names if hasattr(task, 'class_names') else None
            )
            metrics['loss'] = final_loss
            metrics['accuracy'] = accuracy
        
        return adapted_model, metrics
    
    def _reptile_update(
        self,
        adapted_model: MetaSNN,
        epsilon: float = None
    ):
        """
        Perform Reptile update: move meta-parameters toward adapted parameters.
        
        Args:
            adapted_model: Model after task adaptation
            epsilon: Step size (defaults to meta_lr)
        """
        epsilon = epsilon or self.meta_lr
        
        # Get state dicts
        meta_state = self.model.state_dict()
        adapted_state = adapted_model.state_dict()
        
        # Reptile update: θ = θ + ε(θ' - θ)
        new_state = {}
        for key in meta_state.keys():
            # Only update encoder parameters (shared across tasks)
            if 'head' not in key:
                new_state[key] = meta_state[key] + epsilon * (adapted_state[key] - meta_state[key])
            else:
                new_state[key] = meta_state[key]
        
        self.model.load_state_dict(new_state)
    
    def train_meta_epoch(
        self,
        tasks: List[TaskData],
        tasks_per_epoch: int = None,
        samples_per_task: int = None,
        interleave: bool = True
    ) -> Dict[str, float]:
        """
        Train one meta-epoch with task interleaving.
        
        Args:
            tasks: List of all tasks
            tasks_per_epoch: Number of tasks to sample per epoch
            samples_per_task: Samples to use from each task
            interleave: Whether to interleave tasks across domains
            
        Returns:
            Dictionary with epoch metrics
        """
        tasks_per_epoch = tasks_per_epoch or len(tasks)
        
        # Task sampling strategy
        if interleave:
            # Group tasks by domain
            domain_tasks = {}
            for task in tasks:
                domain = task.domain
                if domain not in domain_tasks:
                    domain_tasks[domain] = []
                domain_tasks[domain].append(task)
            
            # Interleave: sample from each domain in round-robin
            sampled_tasks = []
            domains = list(domain_tasks.keys())
            domain_idx = 0
            
            while len(sampled_tasks) < tasks_per_epoch:
                domain = domains[domain_idx % len(domains)]
                if domain_tasks[domain]:
                    task = random.choice(domain_tasks[domain])
                    sampled_tasks.append(task)
                domain_idx += 1
        else:
            sampled_tasks = random.choices(tasks, k=tasks_per_epoch)
        
        epoch_losses = []
        epoch_accuracies = []
        epoch_precision = []
        epoch_recall = []
        epoch_f1 = []
        task_results = []
        
        for task in tqdm(sampled_tasks, desc="Meta-training"):
            # Adapt to task
            adapted_model, task_metrics = self._adapt_model_to_task(
                task,
                num_samples=samples_per_task
            )
            
            # Reptile update
            self._reptile_update(adapted_model)
            
            epoch_losses.append(task_metrics['loss'])
            epoch_accuracies.append(task_metrics['accuracy'])
            epoch_precision.append(task_metrics['precision_weighted'])
            epoch_recall.append(task_metrics['recall_weighted'])
            epoch_f1.append(task_metrics['f1_weighted'])
            task_results.append({
                'task': task.name,
                'domain': task.domain,
                'loss': task_metrics['loss'],
                'accuracy': task_metrics['accuracy'],
                'precision': task_metrics['precision_weighted'],
                'recall': task_metrics['recall_weighted'],
                'f1': task_metrics['f1_weighted']
            })
        
        # Record history
        mean_loss = np.mean(epoch_losses)
        mean_accuracy = np.mean(epoch_accuracies)
        mean_precision = np.mean(epoch_precision)
        mean_recall = np.mean(epoch_recall)
        mean_f1 = np.mean(epoch_f1)
        
        self.history['meta_loss'].append(mean_loss)
        self.history['task_losses'].append(epoch_losses)
        self.history['task_accuracies'].append(epoch_accuracies)
        self.history['task_precision'].append(epoch_precision)
        self.history['task_recall'].append(epoch_recall)
        self.history['task_f1'].append(epoch_f1)
        
        return {
            'loss': mean_loss,
            'accuracy': mean_accuracy,
            'precision': mean_precision,
            'recall': mean_recall,
            'f1': mean_f1,
            'task_results': task_results
        }
    
    def meta_train(
        self,
        tasks: List[TaskData],
        num_epochs: int = 100,
        tasks_per_epoch: int = None,
        samples_per_task: int = 32,
        interleave: bool = True,
        eval_interval: int = 10,
        eval_tasks: List[TaskData] = None,
        callback: Callable = None
    ) -> Dict:
        """
        Full meta-training loop.
        
        Args:
            tasks: Training tasks
            num_epochs: Number of meta-epochs
            tasks_per_epoch: Tasks per epoch
            samples_per_task: Samples from each task
            interleave: Use task interleaving
            eval_interval: Epochs between evaluations
            eval_tasks: Tasks for evaluation (optional)
            callback: Function called after each epoch
            
        Returns:
            Training history
        """
        print(f"Starting meta-training on {len(tasks)} tasks")
        print(f"  Meta LR: {self.meta_lr}")
        print(f"  Inner LR: {self.inner_lr}")
        print(f"  Inner Steps: {self.inner_steps}")
        print(f"  Interleaving: {interleave}")
        print("-" * 60)
        
        best_eval_acc = 0.0
        
        for epoch in range(num_epochs):
            # Train epoch
            metrics = self.train_meta_epoch(
                tasks,
                tasks_per_epoch=tasks_per_epoch,
                samples_per_task=samples_per_task,
                interleave=interleave
            )
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Loss: {metrics['loss']:.4f}, Acc: {metrics['accuracy']:.4f}, "
                  f"P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
            
            # Evaluation
            if eval_tasks and (epoch + 1) % eval_interval == 0:
                eval_metrics = self.evaluate(eval_tasks)
                print(f"  Eval - Loss: {eval_metrics['loss']:.4f}, "
                      f"Acc: {eval_metrics['accuracy']:.4f}, "
                      f"P: {eval_metrics['precision']:.4f}, R: {eval_metrics['recall']:.4f}, F1: {eval_metrics['f1']:.4f}")
                
                if eval_metrics['accuracy'] > best_eval_acc:
                    best_eval_acc = eval_metrics['accuracy']
            
            # Callback
            if callback:
                callback(epoch, metrics)
        
        return self.history
    
    def evaluate(
        self,
        tasks: List[TaskData],
        adaptation_steps: int = None
    ) -> Dict[str, float]:
        """
        Evaluate meta-learned model on tasks.
        
        Args:
            tasks: Tasks to evaluate on
            adaptation_steps: Steps for adaptation (None = use inner_steps)
            
        Returns:
            Evaluation metrics
        """
        original_steps = self.inner_steps
        if adaptation_steps is not None:
            self.inner_steps = adaptation_steps
        
        all_task_metrics = []
        
        for task in tasks:
            _, task_metrics = self._adapt_model_to_task(task)
            all_task_metrics.append(task_metrics)
        
        self.inner_steps = original_steps
        
        # Aggregate metrics across tasks
        return {
            'loss': np.mean([m['loss'] for m in all_task_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in all_task_metrics]),
            'precision': np.mean([m['precision_weighted'] for m in all_task_metrics]),
            'recall': np.mean([m['recall_weighted'] for m in all_task_metrics]),
            'f1': np.mean([m['f1_weighted'] for m in all_task_metrics]),
            'losses': [m['loss'] for m in all_task_metrics],
            'accuracies': [m['accuracy'] for m in all_task_metrics],
            'task_metrics': all_task_metrics
        }
    
    def adapt_and_evaluate(
        self,
        task: TaskData,
        train_ratio: float = 0.8,
        adaptation_steps: int = None,
        min_test_samples: int = 3
    ) -> Tuple[MetaSNN, Dict]:
        """
        Adapt to a task and evaluate on held-out data.
        
        Args:
            task: Task to adapt to
            train_ratio: Ratio of data for adaptation
            adaptation_steps: Override inner_steps
            min_test_samples: Minimum number of test samples to ensure
            
        Returns:
            adapted_model: Model after adaptation
            metrics: Evaluation metrics
        """
        original_steps = self.inner_steps
        if adaptation_steps is not None:
            self.inner_steps = adaptation_steps
        
        # Split data - ensure minimum test samples for small datasets
        total_samples = len(task.X)
        n_test = max(min_test_samples, int(total_samples * (1 - train_ratio)))
        n_train = total_samples - n_test
        
        # Handle edge case where we can't have enough test samples
        if n_train < 1:
            n_train = 1
            n_test = total_samples - 1
        
        indices = np.random.permutation(total_samples)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        
        # Create train task
        train_task = TaskData(
            name=task.name + "_train",
            domain=task.domain,
            X=task.X[train_idx],
            y=task.y[train_idx],
            class_names=task.class_names,
            feature_names=task.feature_names,
            num_classes=task.num_classes,
            metadata=task.metadata
        )
        
        # Adapt
        adapted_model, train_metrics = self._adapt_model_to_task(train_task)
        
        # Evaluate on test
        X_test = task.X[test_idx]
        y_test = task.y[test_idx]
        
        spikes = self._prepare_inputs(X_test)
        labels = torch.LongTensor(y_test).to(self.device)
        
        adapted_model.eval()
        with torch.no_grad():
            mem_out, _, recordings = adapted_model(spikes, return_recordings=True)
            logits = adapted_model.get_output_logits(mem_out)
            
            predictions = torch.argmax(logits, dim=1)
            
            # Calculate comprehensive test metrics
            test_metrics = calculate_metrics(
                y_true=labels.cpu().numpy(),
                y_pred=predictions.cpu().numpy(),
                logits=logits.cpu(),
                class_names=task.class_names if hasattr(task, 'class_names') else None
            )
        
        self.inner_steps = original_steps
        
        metrics = {
            # Train metrics
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision_weighted'],
            'train_recall': train_metrics['recall_weighted'],
            'train_f1': train_metrics['f1_weighted'],
            # Test metrics
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision_weighted'],
            'test_recall': test_metrics['recall_weighted'],
            'test_f1': test_metrics['f1_weighted'],
            'test_precision_macro': test_metrics['precision_macro'],
            'test_recall_macro': test_metrics['recall_macro'],
            'test_f1_macro': test_metrics['f1_macro'],
            'test_precision_per_class': test_metrics['precision_per_class'],
            'test_recall_per_class': test_metrics['recall_per_class'],
            'test_f1_per_class': test_metrics['f1_per_class'],
            'test_confusion_matrix': test_metrics['confusion_matrix'],
            'test_classification_report': test_metrics['classification_report'],
            # Raw data
            'predictions': predictions.cpu().numpy(),
            'true_labels': y_test,
            'recordings': recordings
        }
        
        return adapted_model, metrics
    
    def save_model(self, path: str):
        """Save meta-learned model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps,
            'history': self.history
        }, path)
    
    def load_model(self, path: str):
        """Load meta-learned model"""
        # Use weights_only=False for trusted local checkpoints (contains numpy arrays in history)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_lr = checkpoint['meta_lr']
        self.inner_lr = checkpoint['inner_lr']
        self.inner_steps = checkpoint['inner_steps']
        self.history = checkpoint.get('history', self.history)


def create_reptile_learner(
    input_size: int = 10,
    hidden_sizes: List[int] = [64, 32],
    output_size: int = 3,
    beta: float = 0.9,
    meta_lr: float = 0.1,
    inner_lr: float = 0.001,
    inner_steps: int = 5,
    device: torch.device = None,
    use_class_weights: bool = False
) -> ReptileMetaLearner:
    """
    Factory function to create a Reptile meta-learner.
    
    Args:
        input_size: Number of input features
        hidden_sizes: Hidden layer sizes
        output_size: Default output size
        beta: Membrane decay rate
        meta_lr: Outer loop learning rate
        inner_lr: Inner loop learning rate
        inner_steps: Inner loop steps
        device: Computation device
        use_class_weights: Whether to use class-weighted loss for imbalanced tasks
        
    Returns:
        Configured ReptileMetaLearner
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_meta_snn(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        beta=beta,
        device=device
    )
    
    learner = ReptileMetaLearner(
        model=model,
        meta_lr=meta_lr,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        device=device,
        use_class_weights=use_class_weights
    )
    
    return learner


if __name__ == "__main__":
    # Test the Reptile meta-learner
    from utils.data_loader import create_meta_learning_tasks
    
    # Load tasks
    print("Loading tasks...")
    all_tasks, tasks_by_domain = create_meta_learning_tasks(
        data_root="data",
        sequence_length=250,
        downsample_factor=2
    )
    
    if len(all_tasks) == 0:
        print("No tasks loaded. Exiting.")
        exit(1)
    
    # Create learner
    print("\nCreating Reptile meta-learner...")
    learner = create_reptile_learner(
        input_size=10,
        hidden_sizes=[64, 32],
        output_size=3,
        beta=0.9,
        meta_lr=0.1,
        inner_lr=0.001,
        inner_steps=5
    )
    
    # Quick test: adapt to first task
    print("\nTesting adaptation on first task...")
    task = all_tasks[0]
    print(f"Task: {task.name}, Classes: {task.num_classes}, Samples: {len(task.X)}")
    
    adapted_model, loss, accuracy = learner._adapt_model_to_task(task)
    print(f"After adaptation - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Short meta-training test
    print("\nRunning short meta-training test...")
    history = learner.meta_train(
        tasks=all_tasks,
        num_epochs=3,
        tasks_per_epoch=2,
        samples_per_task=16,
        interleave=True
    )
    
    print("\nMeta-training complete!")
    print(f"Final meta-loss: {history['meta_loss'][-1]:.4f}")
