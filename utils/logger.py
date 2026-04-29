"""
Structured Logging System for Meta-Learning Experiments

Provides comprehensive logging with timestamped directories containing:
- Configuration files (YAML/JSON)
- Training logs (CSV and text)
- Model checkpoints
- Evaluation metrics
- Explainability visualizations
- Summary reports
"""

import os
import json
import csv
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil
import numpy as np
import torch


class ExperimentLogger:
    """
    Comprehensive experiment logging for meta-learning pipeline.
    
    Directory structure:
    results/
    └── experiment_YYYYMMDD_HHMMSS/
        ├── config.json
        ├── logs/
        │   ├── training.log
        │   ├── meta_training.csv
        │   └── task_evaluations.csv
        ├── models/
        │   ├── meta_model_epoch_X.pth
        │   └── best_model.pth
        ├── metrics/
        │   ├── task_metrics.json
        │   └── summary.json
        └── explanations/
            ├── task_name/
            │   ├── sample_0.png
            │   └── feature_importance.json
            └── ...
    """
    
    def __init__(
        self,
        base_dir: str = "results",
        experiment_name: str = None,
        config: Dict = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            base_dir: Base directory for all experiments
            experiment_name: Custom experiment name (auto-generated if None)
            config: Configuration dictionary to save
        """
        # Create timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name:
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = f"experiment_{timestamp}"
        
        self.base_dir = base_dir
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        
        # Create directory structure
        self._create_directories()
        
        # Initialize file logger
        self._setup_file_logger()
        
        # Save config if provided
        if config:
            self.save_config(config)
        
        # Initialize metric tracking
        self.metrics = {
            'meta_training': [],
            'task_evaluations': []
        }
        
        self.logger.info(f"Experiment initialized: {self.experiment_name}")
        self.logger.info(f"Results will be saved to: {self.experiment_dir}")
    
    def _create_directories(self):
        """Create experiment directory structure"""
        dirs = [
            self.experiment_dir,
            os.path.join(self.experiment_dir, 'logs'),
            os.path.join(self.experiment_dir, 'models'),
            os.path.join(self.experiment_dir, 'metrics'),
            os.path.join(self.experiment_dir, 'explanations')
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _setup_file_logger(self):
        """Setup Python logging to file"""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        log_path = os.path.join(self.experiment_dir, 'logs', 'training.log')
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def save_config(self, config: Dict):
        """Save experiment configuration"""
        config_path = os.path.join(self.experiment_dir, 'config.json')
        
        # Convert any non-serializable objects
        serializable_config = self._make_serializable(config)
        
        with open(config_path, 'w') as f:
            json.dump(serializable_config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {config_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, torch.device):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def log_meta_training_step(
        self,
        epoch: int,
        metrics: Dict[str, float],
        task_results: List[Dict] = None
    ):
        """
        Log a meta-training epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary with 'loss', 'accuracy', etc.
            task_results: Optional list of per-task results
        """
        # Log to file
        self.logger.info(
            f"Epoch {epoch}: Loss={metrics.get('loss', 0):.4f}, "
            f"Accuracy={metrics.get('accuracy', 0):.4f}"
        )
        
        # Store metrics
        record = {'epoch': epoch, **metrics}
        self.metrics['meta_training'].append(record)
        
        # Append to CSV
        csv_path = os.path.join(self.experiment_dir, 'logs', 'meta_training.csv')
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'accuracy'])
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'epoch': epoch,
                'loss': metrics.get('loss', 0),
                'accuracy': metrics.get('accuracy', 0)
            })
        
        # Log task results if provided
        if task_results:
            tasks_csv = os.path.join(self.experiment_dir, 'logs', f'task_results_epoch_{epoch}.csv')
            with open(tasks_csv, 'w', newline='') as f:
                if task_results:
                    writer = csv.DictWriter(f, fieldnames=task_results[0].keys())
                    writer.writeheader()
                    writer.writerows(task_results)
    
    def log_task_evaluation(
        self,
        task_name: str,
        metrics: Dict[str, float],
        predictions: np.ndarray = None,
        true_labels: np.ndarray = None
    ):
        """
        Log evaluation results for a specific task.
        
        Args:
            task_name: Name of the task
            metrics: Evaluation metrics
            predictions: Model predictions
            true_labels: Ground truth labels
        """
        self.logger.info(
            f"Task {task_name}: Test Accuracy={metrics.get('test_accuracy', 0):.4f}"
        )
        
        # Store in metrics
        record = {'task': task_name, **metrics}
        self.metrics['task_evaluations'].append(record)
        
        # Append to CSV
        csv_path = os.path.join(self.experiment_dir, 'logs', 'task_evaluations.csv')
        file_exists = os.path.exists(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            fieldnames = ['task', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'task': task_name,
                'train_loss': metrics.get('train_loss', 0),
                'train_accuracy': metrics.get('train_accuracy', 0),
                'test_loss': metrics.get('test_loss', 0),
                'test_accuracy': metrics.get('test_accuracy', 0)
            })
        
        # Save detailed predictions
        if predictions is not None and true_labels is not None:
            pred_path = os.path.join(
                self.experiment_dir, 'metrics', 
                f'{task_name}_predictions.json'
            )
            with open(pred_path, 'w') as f:
                json.dump({
                    'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                    'true_labels': true_labels.tolist() if isinstance(true_labels, np.ndarray) else true_labels
                }, f)
    
    def save_model(
        self,
        model: torch.nn.Module,
        filename: str,
        optimizer: torch.optim.Optimizer = None,
        epoch: int = None,
        metrics: Dict = None
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            filename: Filename (without path)
            optimizer: Optional optimizer state
            epoch: Optional epoch number
            metrics: Optional metrics dictionary
        """
        model_path = os.path.join(self.experiment_dir, 'models', filename)
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def save_adapted_model(
        self,
        model: torch.nn.Module,
        task_name: str,
        metrics: Dict = None
    ):
        """
        Save task-adapted model.
        
        Args:
            model: Adapted model
            task_name: Task name for filename
            metrics: Adaptation metrics
        """
        filename = f"adapted_{task_name}.pth"
        self.save_model(model, filename, metrics=metrics)
    
    def get_explanation_dir(self, task_name: str) -> str:
        """
        Get directory for task explanations, creating if needed.
        
        Args:
            task_name: Task name
            
        Returns:
            Path to explanation directory
        """
        exp_dir = os.path.join(self.experiment_dir, 'explanations', task_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir
    
    def save_explanation(
        self,
        task_name: str,
        sample_idx: int,
        explanation: Dict,
        figure = None
    ):
        """
        Save explanation results.
        
        Args:
            task_name: Task name
            sample_idx: Sample index
            explanation: Explanation dictionary
            figure: Optional matplotlib figure
        """
        exp_dir = self.get_explanation_dir(task_name)
        
        # Save figure
        if figure is not None:
            fig_path = os.path.join(exp_dir, f'sample_{sample_idx}.png')
            figure.savefig(fig_path, dpi=150, bbox_inches='tight')
            self.logger.debug(f"Explanation figure saved to {fig_path}")
        
        # Save JSON data (excluding non-serializable items)
        # Convert numpy types to Python native types for JSON serialization
        prediction = explanation.get('prediction')
        if hasattr(prediction, 'item'):
            prediction = prediction.item()
        
        true_label = explanation.get('true_label')
        if hasattr(true_label, 'item'):
            true_label = true_label.item()
        
        probs = explanation.get('probabilities')
        if isinstance(probs, np.ndarray):
            probs = probs.tolist()
        
        json_data = {
            'prediction': int(prediction) if prediction is not None else None,
            'true_label': int(true_label) if true_label is not None else None,
            'probabilities': probs,
            'sample_idx': int(sample_idx)
        }
        
        json_path = os.path.join(exp_dir, f'sample_{sample_idx}.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    def save_feature_importance(
        self,
        task_name: str,
        importance: np.ndarray,
        feature_names: List[str]
    ):
        """
        Save feature importance scores.
        
        Args:
            task_name: Task name
            importance: Importance scores
            feature_names: Feature names
        """
        exp_dir = self.get_explanation_dir(task_name)
        
        data = {
            'feature_names': feature_names,
            'importance': importance.tolist() if isinstance(importance, np.ndarray) else importance
        }
        
        path = os.path.join(exp_dir, 'feature_importance.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_summary(self, summary: Dict = None):
        """
        Save experiment summary.
        
        Args:
            summary: Optional custom summary, otherwise auto-generated
        """
        if summary is None:
            # Auto-generate summary
            summary = {
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'meta_training_epochs': len(self.metrics['meta_training']),
                'tasks_evaluated': len(self.metrics['task_evaluations'])
            }
            
            if self.metrics['meta_training']:
                summary['final_meta_loss'] = self.metrics['meta_training'][-1].get('loss')
                summary['final_meta_accuracy'] = self.metrics['meta_training'][-1].get('accuracy')
            
            if self.metrics['task_evaluations']:
                accs = [m.get('test_accuracy', 0) for m in self.metrics['task_evaluations']]
                summary['mean_task_accuracy'] = float(np.mean(accs))
                summary['std_task_accuracy'] = float(np.std(accs))
        
        summary_path = os.path.join(self.experiment_dir, 'metrics', 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to {summary_path}")
        
        # Also log summary
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 60)
        for key, value in summary.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_message(self, message: str, level: str = 'info'):
        """
        Log a custom message.
        
        Args:
            message: Message to log
            level: Log level ('debug', 'info', 'warning', 'error')
        """
        log_func = getattr(self.logger, level, self.logger.info)
        log_func(message)
    
    def close(self):
        """Cleanup and save final summary"""
        self.save_summary()
        self.logger.info(f"Experiment {self.experiment_name} completed")
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def create_experiment_logger(
    base_dir: str = "results",
    experiment_name: str = None,
    config: Dict = None
) -> ExperimentLogger:
    """
    Factory function to create experiment logger.
    
    Args:
        base_dir: Base results directory
        experiment_name: Custom experiment name
        config: Configuration to save
        
    Returns:
        Configured ExperimentLogger
    """
    return ExperimentLogger(
        base_dir=base_dir,
        experiment_name=experiment_name,
        config=config
    )


if __name__ == "__main__":
    # Test the logger
    print("Testing ExperimentLogger...")
    
    config = {
        'model': {
            'input_size': 10,
            'hidden_sizes': [64, 32],
            'output_size': 3
        },
        'training': {
            'meta_lr': 0.1,
            'inner_lr': 0.001,
            'num_epochs': 100
        }
    }
    
    logger = create_experiment_logger(
        base_dir="test_results",
        experiment_name="test_experiment",
        config=config
    )
    
    # Test logging
    logger.log_message("Starting test experiment")
    
    # Test meta-training logging
    for epoch in range(3):
        metrics = {'loss': 1.0 - epoch * 0.3, 'accuracy': 0.5 + epoch * 0.15}
        logger.log_meta_training_step(epoch, metrics)
    
    # Test task evaluation logging
    for task in ['task_1', 'task_2']:
        logger.log_task_evaluation(
            task,
            {'train_loss': 0.5, 'train_accuracy': 0.8, 'test_loss': 0.6, 'test_accuracy': 0.75},
            predictions=np.array([0, 1, 2, 1]),
            true_labels=np.array([0, 1, 2, 0])
        )
    
    # Test summary
    logger.close()
    
    print(f"\nTest results saved to: {logger.experiment_dir}")
