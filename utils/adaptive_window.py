"""
Adaptive Window Selection for Meta-Learning

Instead of using a fixed sequence_length for all tasks, this module learns
optimal window sizes per task during preprocessing.

Key features:
- Grid search over window candidates
- Multi-objective scoring: accuracy, computational cost, transfer ability
- Spectral analysis for information retention (optional)
- Per-task and per-domain window optimization

Scoring function:
    score = α × adaptation_accuracy - β × computational_cost + γ × transfer_score

Where:
    - adaptation_accuracy: How well model adapts to this task with k inner steps
    - computational_cost: Normalized window size (longer = slower)
    - transfer_score: How well this window helps other tasks (prevents overfitting)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from copy import deepcopy
from tqdm import tqdm
import warnings
from scipy import fft

warnings.filterwarnings('ignore')


@dataclass
class WindowSearchResult:
    """Container for window search results"""
    task_name: str
    domain: str
    original_length: int
    optimal_window: int
    best_score: float
    adaptation_accuracy: float
    transfer_score: float
    computational_cost: float
    info_retention: Optional[float]  # FFT-based metric
    all_scores: Dict[int, Dict[str, float]]  # window -> {acc, transfer, cost, score}


@dataclass
class WindowSearchConfig:
    """Configuration for window search"""
    # Window candidates (or range parameters)
    window_candidates: List[int] = None  # If None, use range parameters
    window_min: int = 100
    window_max: int = 2000
    window_step: int = 100
    
    # Scoring weights
    alpha: float = 1.0   # Weight for adaptation accuracy
    beta: float = 0.1    # Weight for computational cost (penalty)
    gamma: float = 0.3   # Weight for transfer score
    
    # Search parameters
    inner_steps: int = 5       # Quick adaptation steps
    samples_per_task: int = 32  # Samples for quick evaluation
    transfer_tasks: int = 3     # Number of tasks for transfer eval
    
    # Information retention (spectral analysis)
    use_spectral_analysis: bool = False
    min_info_retention: float = 0.7  # Minimum required info retention
    
    # Efficiency
    parallel_eval: bool = False  # Not implemented yet
    cache_results: bool = True
    
    def __post_init__(self):
        if self.window_candidates is None:
            self.window_candidates = list(range(
                self.window_min, 
                self.window_max + 1, 
                self.window_step
            ))


class AdaptiveWindowSelector:
    """
    Learns optimal window sizes for each task in the meta-learning setup.
    
    Usage:
        selector = AdaptiveWindowSelector(config, device)
        optimal_windows = selector.find_optimal_windows(all_tasks, meta_model)
        # Returns: {task_name: optimal_window_size}
    """
    
    def __init__(
        self,
        config: WindowSearchConfig = None,
        device: torch.device = None,
        verbose: bool = True
    ):
        """
        Initialize the adaptive window selector.
        
        Args:
            config: Window search configuration
            device: Computation device
            verbose: Whether to print progress
        """
        self.config = config or WindowSearchConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        
        # Cache for processed data
        self._cache = {}
        
        # Results storage
        self.results: Dict[str, WindowSearchResult] = {}
        
    def find_optimal_windows(
        self,
        all_tasks: List[Any],
        meta_model: nn.Module,
        inner_lr: float = 0.001,
        validation_tasks: List[str] = None
    ) -> Dict[str, int]:
        """
        Find optimal window sizes for all tasks.
        
        Args:
            all_tasks: List of TaskData objects
            meta_model: The meta-learned SNN model
            inner_lr: Learning rate for quick adaptation
            validation_tasks: Task names to hold out for validation (prevents overfitting)
            
        Returns:
            Dictionary mapping task names to optimal window sizes
        """
        optimal_windows = {}
        
        # Separate validation tasks if provided
        train_tasks = all_tasks
        if validation_tasks:
            train_tasks = [t for t in all_tasks if t.name not in validation_tasks]
            if self.verbose:
                print(f"Holding out {len(validation_tasks)} tasks for validation")
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("ADAPTIVE WINDOW SELECTION")
            print("=" * 60)
            print(f"Window candidates: {self.config.window_candidates}")
            print(f"Scoring weights: α={self.config.alpha}, β={self.config.beta}, γ={self.config.gamma}")
            print(f"Quick adapt: {self.config.inner_steps} steps, {self.config.samples_per_task} samples")
            print()
        
        # Process each task
        for task in tqdm(train_tasks, desc="Finding optimal windows", disable=not self.verbose):
            result = self._find_optimal_window_for_task(
                task=task,
                all_tasks=train_tasks,
                meta_model=meta_model,
                inner_lr=inner_lr
            )
            
            optimal_windows[task.name] = result.optimal_window
            self.results[task.name] = result
            
            if self.verbose:
                print(f"\n  {task.name}: optimal={result.optimal_window} "
                      f"(acc={result.adaptation_accuracy:.3f}, "
                      f"transfer={result.transfer_score:.3f}, "
                      f"score={result.best_score:.3f})")
        
        # Print summary
        if self.verbose:
            self._print_summary(optimal_windows)
        
        return optimal_windows
    
    def _find_optimal_window_for_task(
        self,
        task: Any,
        all_tasks: List[Any],
        meta_model: nn.Module,
        inner_lr: float
    ) -> WindowSearchResult:
        """
        Find optimal window for a single task.
        
        Args:
            task: TaskData object
            all_tasks: All tasks (for transfer evaluation)
            meta_model: Meta-learned model
            inner_lr: Inner loop learning rate
            
        Returns:
            WindowSearchResult with optimal window and metrics
        """
        # Get original data length
        original_length = task.metadata.get('original_length', task.X.shape[1])
        
        # Filter window candidates based on original length
        valid_windows = [
            w for w in self.config.window_candidates 
            if w <= original_length
        ]
        
        if len(valid_windows) == 0:
            # If all windows are too large, use the original length
            valid_windows = [original_length]
        
        best_score = float('-inf')
        best_window = valid_windows[0]
        best_acc = 0.0
        best_transfer = 0.0
        best_cost = 0.0
        best_info = None
        all_scores = {}
        
        # Get other tasks for transfer evaluation
        other_tasks = [t for t in all_tasks if t.name != task.name]
        if len(other_tasks) > self.config.transfer_tasks:
            # Sample diverse tasks from different domains
            other_tasks = self._sample_diverse_tasks(other_tasks, self.config.transfer_tasks)
        
        for window_size in valid_windows:
            # Apply window to task data
            task_windowed = self._apply_window(task, window_size)
            
            # Quick adaptation
            adapted_model, acc = self._quick_adapt(
                meta_model=meta_model,
                task=task_windowed,
                inner_lr=inner_lr
            )
            
            # Transfer score (how well adapted model works on other tasks)
            transfer_acc = self._compute_transfer_score(
                adapted_model=adapted_model,
                other_tasks=other_tasks,
                window_size=window_size
            )
            
            # Computational cost (normalized)
            max_window = max(self.config.window_candidates)
            cost = window_size / max_window
            
            # Information retention (optional spectral analysis)
            info_retention = None
            if self.config.use_spectral_analysis:
                info_retention = self._compute_info_retention(task, window_size)
                # Skip if below threshold
                if info_retention < self.config.min_info_retention:
                    all_scores[window_size] = {
                        'accuracy': acc,
                        'transfer': transfer_acc,
                        'cost': cost,
                        'info_retention': info_retention,
                        'score': float('-inf'),
                        'reason': 'info_retention_too_low'
                    }
                    continue
            
            # Compute final score
            score = (
                self.config.alpha * acc 
                - self.config.beta * cost 
                + self.config.gamma * transfer_acc
            )
            
            all_scores[window_size] = {
                'accuracy': acc,
                'transfer': transfer_acc,
                'cost': cost,
                'info_retention': info_retention,
                'score': score
            }
            
            if score > best_score:
                best_score = score
                best_window = window_size
                best_acc = acc
                best_transfer = transfer_acc
                best_cost = cost
                best_info = info_retention
        
        return WindowSearchResult(
            task_name=task.name,
            domain=task.domain,
            original_length=original_length,
            optimal_window=best_window,
            best_score=best_score,
            adaptation_accuracy=best_acc,
            transfer_score=best_transfer,
            computational_cost=best_cost,
            info_retention=best_info,
            all_scores=all_scores
        )
    
    def _apply_window(self, task: Any, window_size: int) -> Any:
        """
        Apply a specific window size to task data.
        
        This creates a view of the task with truncated/padded sequences.
        
        Args:
            task: Original TaskData
            window_size: Target window size
            
        Returns:
            Modified TaskData with new sequence length
        """
        # Check cache
        cache_key = (task.name, window_size)
        if self.config.cache_results and cache_key in self._cache:
            return self._cache[cache_key]
        
        X = task.X
        current_length = X.shape[1]
        
        if window_size == current_length:
            return task
        
        if window_size < current_length:
            # Truncate
            X_new = X[:, :window_size, :]
        else:
            # Pad by repeating last value
            padding_length = window_size - current_length
            padding = np.repeat(X[:, -1:, :], padding_length, axis=1)
            X_new = np.concatenate([X, padding], axis=1)
        
        # Create new task with modified data
        # Using SimpleNamespace for lightweight copy
        from types import SimpleNamespace
        task_windowed = SimpleNamespace(
            name=task.name,
            domain=task.domain,
            X=X_new,
            y=task.y,
            class_names=task.class_names,
            feature_names=task.feature_names,
            num_classes=task.num_classes,
            metadata={**task.metadata, 'window_size': window_size}
        )
        
        if self.config.cache_results:
            self._cache[cache_key] = task_windowed
        
        return task_windowed
    
    def _quick_adapt(
        self,
        meta_model: nn.Module,
        task: Any,
        inner_lr: float
    ) -> Tuple[nn.Module, float]:
        """
        Perform quick adaptation to evaluate a window size.
        
        Args:
            meta_model: Meta-learned model to adapt
            task: Task with specific window size
            inner_lr: Learning rate
            
        Returns:
            Tuple of (adapted_model, accuracy)
        """
        # Clone model
        adapted_model = deepcopy(meta_model).to(self.device)
        
        # Adapt output head if needed
        if hasattr(adapted_model, 'adapt_head') and task.num_classes != adapted_model.output_size:
            adapted_model.adapt_head(task.num_classes, self.device)
        
        # Subsample data
        X, y = task.X, task.y
        n_samples = min(self.config.samples_per_task, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]
        
        # Convert to spikes (threshold-based)
        X_tensor = torch.FloatTensor(X).to(self.device)
        spikes = (X_tensor > 0.5).float().permute(1, 0, 2)  # [T, B, F]
        labels = torch.LongTensor(y).to(self.device)
        
        # Quick inner loop
        optimizer = optim.Adam(adapted_model.parameters(), lr=inner_lr)
        criterion = nn.CrossEntropyLoss()
        
        adapted_model.train()
        for _ in range(self.config.inner_steps):
            optimizer.zero_grad()
            
            mem_out, _, _ = adapted_model(spikes, return_recordings=False)
            logits = adapted_model.get_output_logits(mem_out)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        adapted_model.eval()
        with torch.no_grad():
            mem_out, _, _ = adapted_model(spikes, return_recordings=False)
            logits = adapted_model.get_output_logits(mem_out)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
        
        return adapted_model, accuracy
    
    def _compute_transfer_score(
        self,
        adapted_model: nn.Module,
        other_tasks: List[Any],
        window_size: int
    ) -> float:
        """
        Compute how well the adapted model transfers to other tasks.
        
        This helps prevent overfitting to a specific window size that
        only works for one task.
        
        Args:
            adapted_model: Model adapted with specific window
            other_tasks: Other tasks to evaluate on
            window_size: Window size to apply to other tasks
            
        Returns:
            Mean accuracy on other tasks
        """
        if len(other_tasks) == 0:
            return 0.0
        
        accuracies = []
        
        adapted_model.eval()
        with torch.no_grad():
            for task in other_tasks:
                # Apply same window to this task
                task_windowed = self._apply_window(task, window_size)
                
                # Subsample
                X, y = task_windowed.X, task_windowed.y
                n_samples = min(self.config.samples_per_task // 2, len(X))
                if n_samples < 2:
                    continue
                    
                indices = np.random.choice(len(X), n_samples, replace=False)
                X, y = X[indices], y[indices]
                
                # Encode and evaluate
                X_tensor = torch.FloatTensor(X).to(self.device)
                spikes = (X_tensor > 0.5).float().permute(1, 0, 2)
                labels = torch.LongTensor(y).to(self.device)
                
                # Need to adapt head for this task's classes
                model_copy = deepcopy(adapted_model)
                if hasattr(model_copy, 'adapt_head') and task.num_classes != model_copy.output_size:
                    model_copy.adapt_head(task.num_classes, self.device)
                
                mem_out, _, _ = model_copy(spikes, return_recordings=False)
                logits = model_copy.get_output_logits(mem_out)
                predictions = torch.argmax(logits, dim=1)
                acc = (predictions == labels).float().mean().item()
                accuracies.append(acc)
        
        return np.mean(accuracies) if accuracies else 0.0
    
    def _compute_info_retention(self, task: Any, window_size: int) -> float:
        """
        Compute information retention using spectral analysis (FFT).
        
        Measures how much of the original signal's frequency content
        is preserved after windowing.
        
        Args:
            task: Original task data
            window_size: Window size to evaluate
            
        Returns:
            Information retention ratio (0-1)
        """
        X = task.X
        original_length = X.shape[1]
        
        if window_size >= original_length:
            return 1.0  # No information loss
        
        # Compute FFT of original and windowed signals
        # Average across samples and features for efficiency
        original_avg = np.mean(X, axis=(0, 2))  # [timesteps]
        windowed_avg = original_avg[:window_size]
        
        # FFT
        fft_original = np.abs(fft.fft(original_avg))
        fft_windowed = np.abs(fft.fft(windowed_avg, n=original_length))  # Pad to same length
        
        # Compute energy retention
        energy_original = np.sum(fft_original ** 2)
        energy_windowed = np.sum(fft_windowed ** 2)
        
        retention = energy_windowed / (energy_original + 1e-10)
        return min(retention, 1.0)  # Cap at 1.0
    
    def _sample_diverse_tasks(self, tasks: List[Any], n: int) -> List[Any]:
        """
        Sample diverse tasks from different domains.
        
        Args:
            tasks: List of tasks
            n: Number of tasks to sample
            
        Returns:
            Sampled tasks
        """
        # Group by domain
        domains = {}
        for task in tasks:
            domain = task.domain
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(task)
        
        # Sample from each domain
        sampled = []
        domain_list = list(domains.keys())
        
        while len(sampled) < n and any(domains.values()):
            for domain in domain_list:
                if domains[domain] and len(sampled) < n:
                    # Random sample from this domain
                    idx = np.random.randint(len(domains[domain]))
                    sampled.append(domains[domain].pop(idx))
        
        return sampled
    
    def _print_summary(self, optimal_windows: Dict[str, int]):
        """Print summary of window selection results."""
        print("\n" + "=" * 60)
        print("WINDOW SELECTION SUMMARY")
        print("=" * 60)
        
        # Group by domain
        domains = {}
        for task_name, window in optimal_windows.items():
            result = self.results.get(task_name)
            if result:
                domain = result.domain
                if domain not in domains:
                    domains[domain] = []
                domains[domain].append((task_name, window, result))
        
        for domain, domain_results in domains.items():
            print(f"\n{domain.upper()}:")
            windows = [w for _, w, _ in domain_results]
            print(f"  Windows: min={min(windows)}, max={max(windows)}, mean={np.mean(windows):.0f}")
            
            for task_name, window, result in domain_results:
                print(f"    {task_name}: {window} "
                      f"(orig={result.original_length}, "
                      f"acc={result.adaptation_accuracy:.3f})")
        
        # Overall statistics
        all_windows = list(optimal_windows.values())
        print(f"\nOverall: {len(all_windows)} tasks")
        print(f"  Window range: {min(all_windows)} - {max(all_windows)}")
        print(f"  Mean window: {np.mean(all_windows):.0f}")
        print(f"  Std window: {np.std(all_windows):.0f}")
    
    def save_results(self, path: str):
        """Save window search results to JSON."""
        results_dict = {
            'config': asdict(self.config),
            'results': {
                name: {
                    'task_name': r.task_name,
                    'domain': r.domain,
                    'original_length': r.original_length,
                    'optimal_window': r.optimal_window,
                    'best_score': r.best_score,
                    'adaptation_accuracy': r.adaptation_accuracy,
                    'transfer_score': r.transfer_score,
                    'computational_cost': r.computational_cost,
                    'info_retention': r.info_retention,
                    'all_scores': {str(k): v for k, v in r.all_scores.items()}
                }
                for name, r in self.results.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        if self.verbose:
            print(f"Results saved to {path}")
    
    def load_results(self, path: str) -> Dict[str, int]:
        """
        Load previously computed optimal windows.
        
        Args:
            path: Path to JSON results file
            
        Returns:
            Dictionary mapping task names to optimal windows
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        optimal_windows = {}
        for name, result in data['results'].items():
            optimal_windows[name] = result['optimal_window']
            
            # Reconstruct WindowSearchResult
            self.results[name] = WindowSearchResult(
                task_name=result['task_name'],
                domain=result['domain'],
                original_length=result['original_length'],
                optimal_window=result['optimal_window'],
                best_score=result['best_score'],
                adaptation_accuracy=result['adaptation_accuracy'],
                transfer_score=result['transfer_score'],
                computational_cost=result['computational_cost'],
                info_retention=result['info_retention'],
                all_scores={int(k): v for k, v in result['all_scores'].items()}
            )
        
        if self.verbose:
            print(f"Loaded optimal windows for {len(optimal_windows)} tasks from {path}")
        
        return optimal_windows


def visualize_window_search(
    results: Dict[str, WindowSearchResult],
    save_path: str = None
):
    """
    Visualize window search results.
    
    Creates plots showing:
    1. Window size vs accuracy per task
    2. Domain-wise window distributions
    3. Score components breakdown
    
    Args:
        results: Dictionary of WindowSearchResult objects
        save_path: Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Window vs Accuracy for each task
    ax1 = axes[0, 0]
    for task_name, result in results.items():
        windows = list(result.all_scores.keys())
        accuracies = [result.all_scores[w]['accuracy'] for w in windows]
        ax1.plot(windows, accuracies, 'o-', alpha=0.7, label=task_name[:15])
        # Mark optimal
        ax1.axvline(result.optimal_window, color='gray', alpha=0.3, linestyle='--')
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('Adaptation Accuracy')
    ax1.set_title('Window Size vs Accuracy')
    ax1.legend(fontsize=6, loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. Domain-wise optimal window distribution
    ax2 = axes[0, 1]
    domains = {}
    for result in results.values():
        if result.domain not in domains:
            domains[result.domain] = []
        domains[result.domain].append(result.optimal_window)
    
    domain_names = list(domains.keys())
    domain_windows = [domains[d] for d in domain_names]
    ax2.boxplot(domain_windows, labels=domain_names)
    ax2.set_ylabel('Optimal Window Size')
    ax2.set_title('Optimal Windows by Domain')
    ax2.grid(True, alpha=0.3)
    
    # 3. Score components for best windows
    ax3 = axes[1, 0]
    task_names = list(results.keys())
    x = np.arange(len(task_names))
    width = 0.25
    
    accs = [results[t].adaptation_accuracy for t in task_names]
    transfers = [results[t].transfer_score for t in task_names]
    costs = [results[t].computational_cost for t in task_names]
    
    ax3.bar(x - width, accs, width, label='Accuracy', alpha=0.8)
    ax3.bar(x, transfers, width, label='Transfer', alpha=0.8)
    ax3.bar(x + width, costs, width, label='Cost', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([t[:10] for t in task_names], rotation=45, ha='right')
    ax3.set_ylabel('Score Component')
    ax3.set_title('Score Components at Optimal Windows')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Window size vs total score heatmap-style
    ax4 = axes[1, 1]
    for task_name, result in results.items():
        windows = list(result.all_scores.keys())
        scores = [result.all_scores[w]['score'] for w in windows]
        ax4.plot(windows, scores, 'o-', alpha=0.7, label=task_name[:15])
    ax4.set_xlabel('Window Size')
    ax4.set_ylabel('Total Score')
    ax4.set_title('Window Size vs Total Score')
    ax4.legend(fontsize=6, loc='best')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


# Convenience function for integration with main pipeline
def find_optimal_windows_for_tasks(
    all_tasks: List[Any],
    meta_model: nn.Module,
    config: Dict[str, Any] = None,
    device: torch.device = None,
    save_path: str = None
) -> Dict[str, int]:
    """
    Convenience function to find optimal windows for all tasks.
    
    Args:
        all_tasks: List of TaskData objects
        meta_model: Meta-learned SNN model
        config: Configuration dictionary (optional)
        device: Computation device
        save_path: Path to save results (optional)
        
    Returns:
        Dictionary mapping task names to optimal window sizes
    """
    # Build config
    search_config = WindowSearchConfig()
    if config:
        for key, value in config.items():
            if hasattr(search_config, key):
                setattr(search_config, key, value)
    
    # Create selector
    selector = AdaptiveWindowSelector(
        config=search_config,
        device=device,
        verbose=True
    )
    
    # Find optimal windows
    inner_lr = config.get('inner_lr', 0.001) if config else 0.001
    optimal_windows = selector.find_optimal_windows(
        all_tasks=all_tasks,
        meta_model=meta_model,
        inner_lr=inner_lr
    )
    
    # Save results if path provided
    if save_path:
        selector.save_results(save_path)
    
    return optimal_windows


if __name__ == "__main__":
    # Test the module
    print("Adaptive Window Selection Module")
    print("=" * 40)
    
    # Create default config
    config = WindowSearchConfig(
        window_candidates=[100, 200, 300, 400, 500, 750, 1000],
        alpha=1.0,
        beta=0.1,
        gamma=0.3
    )
    
    print(f"Window candidates: {config.window_candidates}")
    print(f"Scoring: {config.alpha}*acc - {config.beta}*cost + {config.gamma}*transfer")
