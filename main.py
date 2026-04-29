"""
Meta-Learning SNN Pipeline for Olfactory Data

Main orchestration script that runs the complete pipeline:
1. Load and preprocess heterogeneous olfactory datasets
2. Meta-train SNN using Reptile with task interleaving
3. Adapt and evaluate on each task
4. Generate TSA explanations for predictions
5. Save all results to timestamped directories

Usage:
    python main.py [--config config.json]
    python main.py --data_root data --epochs 100 --meta_lr 0.1

Based on "Robust Spike-Based Continual Meta-Learning" concepts.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import create_meta_learning_tasks, TaskData, apply_adaptive_windows
from utils.logger import create_experiment_logger, ExperimentLogger
from models.meta_snn import create_meta_snn, MetaSNN
from training.reptile import create_reptile_learner, ReptileMetaLearner
from explainability.tsa import TSAExplainer, TSAVisualizer, generate_explanations_for_task
from utils.adaptive_window import (
    AdaptiveWindowSelector, 
    WindowSearchConfig,
    find_optimal_windows_for_tasks,
    visualize_window_search
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Meta-Learning SNN for Olfactory Data Classification'
    )
    
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
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[64, 32],
                        help='Hidden layer sizes')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Membrane potential decay rate')
    
    # Meta-Learning
    parser.add_argument('--meta_lr', type=float, default=0.1,
                        help='Meta (outer) learning rate for Reptile')
    parser.add_argument('--inner_lr', type=float, default=0.001,
                        help='Inner loop learning rate')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='Number of inner loop gradient steps')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of meta-training epochs')
    parser.add_argument('--tasks_per_epoch', type=int, default=None,
                        help='Tasks to sample per epoch (None = all)')
    parser.add_argument('--samples_per_task', type=int, default=32,
                        help='Samples to use from each task')
    parser.add_argument('--interleave', action='store_true', default=True,
                        help='Use task interleaving across domains')
    
    # Evaluation
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Epochs between evaluations')
    parser.add_argument('--adaptation_steps', type=int, default=10,
                        help='Adaptation steps for final evaluation')
    
    # Explainability
    parser.add_argument('--explain_samples', type=int, default=5,
                        help='Number of samples to explain per task')
    parser.add_argument('--no_explanations', action='store_true',
                        help='Skip generating explanations')
    
    # Adaptive Window Selection
    parser.add_argument('--adaptive_windows', action='store_true',
                        help='Enable adaptive window selection per task')
    parser.add_argument('--window_candidates', type=int, nargs='+', 
                        default=[100, 200, 300, 400, 500, 750, 1000, 1500],
                        help='Window sizes to try during search')
    parser.add_argument('--window_alpha', type=float, default=1.0,
                        help='Weight for adaptation accuracy in window scoring')
    parser.add_argument('--window_beta', type=float, default=0.1,
                        help='Weight for computational cost in window scoring')
    parser.add_argument('--window_gamma', type=float, default=0.3,
                        help='Weight for transfer score in window scoring')
    parser.add_argument('--load_windows', type=str, default=None,
                        help='Path to pre-computed optimal windows JSON')
    
    # Output
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for saving results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Custom experiment name')
    
    # Config file
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to JSON config file (overrides other args)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu), auto-detected if None')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Update args with config values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return args


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_gpu_setup(device: torch.device) -> dict:
    """
    Verify GPU setup and return diagnostic information.
    
    Args:
        device: The torch device being used
        
    Returns:
        Dictionary with GPU diagnostic info
    """
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_used': str(device),
        'cuda_version': None,
        'gpu_name': None,
        'gpu_memory_total': None,
        'gpu_memory_allocated': None,
    }
    
    if torch.cuda.is_available():
        gpu_info['cuda_version'] = torch.version.cuda
        if device.type == 'cuda':
            gpu_info['gpu_name'] = torch.cuda.get_device_name(device)
            gpu_info['gpu_memory_total'] = f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
            gpu_info['gpu_memory_allocated'] = f"{torch.cuda.memory_allocated(device) / 1e6:.2f} MB"
    
    return gpu_info


def run_pipeline(args):
    """
    Run the complete meta-learning pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set seed
    set_seed(args.seed)
    
    # Device setup with explicit verification
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verify GPU setup
    gpu_info = verify_gpu_setup(device)
    
    print("=" * 70)
    print("META-LEARNING SNN FOR OLFACTORY DATA")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Random seed: {args.seed}")
    
    # Print GPU diagnostics
    print("\n--- GPU Diagnostics ---")
    print(f"CUDA Available: {gpu_info['cuda_available']}")
    if gpu_info['cuda_available']:
        print(f"CUDA Version: {gpu_info['cuda_version']}")
        if device.type == 'cuda':
            print(f"GPU Name: {gpu_info['gpu_name']}")
            print(f"GPU Memory Total: {gpu_info['gpu_memory_total']}")
            print(f"GPU Memory Allocated: {gpu_info['gpu_memory_allocated']}")
        else:
            print("WARNING: CUDA is available but CPU device is being used!")
            print("         Consider using --device cuda for faster training.")
    else:
        print("WARNING: CUDA not available. Training will be slow on CPU.")
    print("-" * 25)
    print()
    
    # Create configuration dictionary
    config = {
        'data': {
            'data_root': args.data_root,
            'sequence_length': args.sequence_length,
            'downsample_factor': args.downsample_factor
        },
        'model': {
            'input_size': args.input_size,
            'hidden_sizes': args.hidden_sizes,
            'beta': args.beta
        },
        'meta_learning': {
            'meta_lr': args.meta_lr,
            'inner_lr': args.inner_lr,
            'inner_steps': args.inner_steps,
            'epochs': args.epochs,
            'tasks_per_epoch': args.tasks_per_epoch,
            'samples_per_task': args.samples_per_task,
            'interleave': args.interleave
        },
        'evaluation': {
            'eval_interval': args.eval_interval,
            'adaptation_steps': args.adaptation_steps,
            'explain_samples': args.explain_samples
        },
        'adaptive_window': {
            'enabled': getattr(args, 'adaptive_windows', False),
            'window_candidates': getattr(args, 'window_candidates', [100, 200, 300, 400, 500, 750, 1000, 1500]),
            'alpha': getattr(args, 'window_alpha', 1.0),
            'beta': getattr(args, 'window_beta', 0.1),
            'gamma': getattr(args, 'window_gamma', 0.3),
            'load_cached': getattr(args, 'load_windows', None)
        },
        'device': str(device),
        'gpu_info': gpu_info,
        'seed': args.seed
    }
    
    # Initialize logger
    logger = create_experiment_logger(
        base_dir=args.results_dir,
        experiment_name=args.experiment_name,
        config=config
    )
    
    try:
        # ====================================================================
        # STEP 1: Load Data
        # ====================================================================
        logger.log_message("=" * 60)
        logger.log_message("STEP 1: Loading Olfactory Datasets")
        logger.log_message("=" * 60)
        
        all_tasks, tasks_by_domain = create_meta_learning_tasks(
            data_root=args.data_root,
            sequence_length=args.sequence_length,
            downsample_factor=args.downsample_factor,
            max_features=args.input_size
        )
        
        if len(all_tasks) == 0:
            logger.log_message("ERROR: No tasks loaded. Check data path.", level='error')
            return
        
        logger.log_message(f"Loaded {len(all_tasks)} tasks from {len(tasks_by_domain)} domains")
        
        for domain, domain_tasks in tasks_by_domain.items():
            total_samples = sum(t.X.shape[0] for t in domain_tasks)
            logger.log_message(f"  {domain}: {len(domain_tasks)} tasks, {total_samples} samples")
        
        # ====================================================================
        # STEP 1.5: Adaptive Window Selection (Optional)
        # ====================================================================
        optimal_windows = None
        
        # Check config for adaptive window settings
        adaptive_config = config.get('adaptive_window', {}) if isinstance(config, dict) else {}
        use_adaptive = getattr(args, 'adaptive_windows', False) or adaptive_config.get('enabled', False)
        
        if use_adaptive:
            logger.log_message("=" * 60)
            logger.log_message("STEP 1.5: Adaptive Window Selection")
            logger.log_message("=" * 60)
            
            # Check if we should load pre-computed windows
            load_path = getattr(args, 'load_windows', None) or adaptive_config.get('load_cached', None)
            
            if load_path and os.path.exists(load_path):
                logger.log_message(f"Loading pre-computed windows from: {load_path}")
                selector = AdaptiveWindowSelector(device=device)
                optimal_windows = selector.load_results(load_path)
            else:
                # Need to create a temporary model for window search
                logger.log_message("Running window search (this may take a while)...")
                
                # Create temporary learner for window optimization
                temp_max_classes = max(t.num_classes for t in all_tasks)
                temp_learner = create_reptile_learner(
                    input_size=args.input_size,
                    hidden_sizes=args.hidden_sizes,
                    output_size=temp_max_classes,
                    beta=args.beta,
                    meta_lr=args.meta_lr,
                    inner_lr=args.inner_lr,
                    inner_steps=args.inner_steps,
                    device=device
                )
                
                # Build window search config
                window_config = WindowSearchConfig(
                    window_candidates=getattr(args, 'window_candidates', None) or 
                                     adaptive_config.get('window_candidates', [100, 200, 300, 400, 500, 750, 1000, 1500]),
                    alpha=getattr(args, 'window_alpha', None) or adaptive_config.get('alpha', 1.0),
                    beta=getattr(args, 'window_beta', None) or adaptive_config.get('beta', 0.1),
                    gamma=getattr(args, 'window_gamma', None) or adaptive_config.get('gamma', 0.3),
                    inner_steps=adaptive_config.get('inner_steps', 5),
                    samples_per_task=adaptive_config.get('samples_per_task', 32),
                    transfer_tasks=adaptive_config.get('transfer_tasks', 3),
                    use_spectral_analysis=adaptive_config.get('use_spectral_analysis', False),
                    min_info_retention=adaptive_config.get('min_info_retention', 0.7)
                )
                
                selector = AdaptiveWindowSelector(
                    config=window_config,
                    device=device,
                    verbose=True
                )
                
                optimal_windows = selector.find_optimal_windows(
                    all_tasks=all_tasks,
                    meta_model=temp_learner.model,
                    inner_lr=args.inner_lr
                )
                
                # Save results
                windows_path = os.path.join(logger.experiment_dir, 'optimal_windows.json')
                selector.save_results(windows_path)
                logger.log_message(f"Optimal windows saved to: {windows_path}")
                
                # Generate visualization
                try:
                    vis_path = os.path.join(logger.experiment_dir, 'window_analysis.png')
                    visualize_window_search(selector.results, save_path=vis_path)
                except Exception as e:
                    logger.log_message(f"Warning: Could not generate window visualization: {e}")
            
            # Apply optimal windows to tasks
            if optimal_windows:
                logger.log_message("Applying optimal windows to tasks...")
                all_tasks = apply_adaptive_windows(all_tasks, optimal_windows)
                
                # Update tasks_by_domain
                tasks_by_domain = {}
                for task in all_tasks:
                    if task.domain not in tasks_by_domain:
                        tasks_by_domain[task.domain] = []
                    tasks_by_domain[task.domain].append(task)
                
                logger.log_message(f"Applied adaptive windows to {len(optimal_windows)} tasks")
        
        # ====================================================================
        # STEP 2: Create Meta-Learner
        # ====================================================================
        logger.log_message("=" * 60)
        logger.log_message("STEP 2: Creating Meta-Learning SNN")
        logger.log_message("=" * 60)
        
        # Determine default output size (max classes across tasks)
        max_classes = max(t.num_classes for t in all_tasks)
        
        learner = create_reptile_learner(
            input_size=args.input_size,
            hidden_sizes=args.hidden_sizes,
            output_size=max_classes,
            beta=args.beta,
            meta_lr=args.meta_lr,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            device=device
        )
        
        total_params = sum(p.numel() for p in learner.model.parameters())
        logger.log_message(f"Model created with {total_params:,} parameters")
        logger.log_message(f"Architecture: {args.input_size} -> {args.hidden_sizes} -> {max_classes}")
        
        # ====================================================================
        # STEP 3: Meta-Training with Task Interleaving
        # ====================================================================
        logger.log_message("=" * 60)
        logger.log_message("STEP 3: Meta-Training with Reptile")
        logger.log_message("=" * 60)
        
        # Track best model for checkpoint saving
        best_accuracy = 0.0
        best_epoch = 0
        
        def training_callback(epoch, metrics):
            """Callback to log training progress and save best model"""
            nonlocal best_accuracy, best_epoch
            
            logger.log_meta_training_step(
                epoch=epoch + 1,
                metrics=metrics,
                task_results=metrics.get('task_results')
            )
            
            # Save model only when accuracy improves (best model strategy)
            current_accuracy = metrics['accuracy']
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_epoch = epoch + 1
                logger.save_model(
                    learner.model,
                    'best_meta_model.pth',
                    epoch=epoch + 1,
                    metrics={'loss': metrics['loss'], 'accuracy': metrics['accuracy']}
                )
                logger.log_message(
                    f"  New best model saved! Accuracy: {best_accuracy:.4f} at epoch {best_epoch}"
                )
        
        # Run meta-training
        history = learner.meta_train(
            tasks=all_tasks,
            num_epochs=args.epochs,
            tasks_per_epoch=args.tasks_per_epoch,
            samples_per_task=args.samples_per_task,
            interleave=args.interleave,
            eval_interval=args.eval_interval,
            callback=training_callback
        )
        
        # Save final model and log best model info
        logger.save_model(learner.model, 'meta_model_final.pth', metrics={
            'final_loss': history['meta_loss'][-1],
            'final_accuracy': np.mean(history['task_accuracies'][-1])
        })
        logger.log_message(f"Best model was at epoch {best_epoch} with accuracy {best_accuracy:.4f}")
        
        # ====================================================================
        # STEP 4: Task-Specific Adaptation and Evaluation
        # ====================================================================
        logger.log_message("=" * 60)
        logger.log_message("STEP 4: Task-Specific Adaptation & Evaluation")
        logger.log_message("=" * 60)
        
        task_results = []
        adapted_models = {}
        
        for task in all_tasks:
            logger.log_message(f"Adapting to task: {task.name}")
            
            # Adapt and evaluate
            adapted_model, metrics = learner.adapt_and_evaluate(
                task,
                train_ratio=0.8,
                adaptation_steps=args.adaptation_steps
            )
            
            # Log results
            logger.log_task_evaluation(
                task_name=task.name,
                metrics=metrics,
                predictions=metrics['predictions'],
                true_labels=metrics['true_labels']
            )
            
            # Store adapted model
            adapted_models[task.name] = adapted_model
            
            # Save adapted model
            logger.save_adapted_model(adapted_model, task.name, metrics)
            
            task_results.append({
                'task': task.name,
                'domain': task.domain,
                'num_classes': task.num_classes,
                'test_accuracy': metrics['test_accuracy'],
                'test_loss': metrics['test_loss']
            })
            
            logger.log_message(
                f"  {task.name}: Test Acc = {metrics['test_accuracy']:.4f}"
            )
        
        # Print summary
        logger.log_message("\nTask Evaluation Summary:")
        for domain in tasks_by_domain.keys():
            domain_results = [r for r in task_results if r['domain'] == domain]
            if domain_results:
                mean_acc = np.mean([r['test_accuracy'] for r in domain_results])
                logger.log_message(f"  {domain}: Mean Accuracy = {mean_acc:.4f}")
        
        overall_acc = np.mean([r['test_accuracy'] for r in task_results])
        logger.log_message(f"\nOverall Mean Test Accuracy: {overall_acc:.4f}")
        
        # ====================================================================
        # STEP 5: Explainability with TSA
        # ====================================================================
        if not args.no_explanations:
            logger.log_message("=" * 60)
            logger.log_message("STEP 5: Generating TSA Explanations")
            logger.log_message("=" * 60)
            
            for task in all_tasks:
                logger.log_message(f"Generating explanations for: {task.name}")
                
                adapted_model = adapted_models.get(task.name)
                if adapted_model is None:
                    continue
                
                # Get explanation directory
                exp_dir = logger.get_explanation_dir(task.name)
                
                # Prepare test data for explanations
                n_samples = min(args.explain_samples, len(task.X))
                indices = np.random.choice(len(task.X), n_samples, replace=False)
                
                # Create explainer and visualizer
                explainer = TSAExplainer(adapted_model, device)
                visualizer = TSAVisualizer(
                    feature_names=task.feature_names[:task.X.shape[2]] if task.feature_names else None,
                    class_names=task.class_names
                )
                
                all_importances = []
                explanations_generated = 0
                
                for idx in indices:
                    try:
                        # Get sample and convert to spike tensor [T, F]
                        sample_X = task.X[idx]  # [T, F]
                        sample_y = int(task.y[idx])
                        
                        # Convert to spikes via threshold
                        x_tensor = torch.FloatTensor(sample_X)
                        spikes = (x_tensor > 0.5).float()
                        
                        # Generate explanation using refactored TSA
                        explanation = explainer.explain_prediction(spikes, true_label=sample_y)
                        
                        # Create visualization
                        fig = visualizer.create_explanation_figure(
                            explanation,
                            feature_names=task.feature_names[:task.X.shape[2]] if task.feature_names else None,
                            class_names=task.class_names,
                            title=f'{task.name} - Sample {idx}'
                        )
                        
                        # Save figure
                        from pathlib import Path
                        fig_path = Path(exp_dir) / f'sample_{idx}.png'
                        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        
                        # Get feature importance
                        feat_imp = explainer.get_feature_importance(
                            explanation['tsa_map'],
                            explanation['prediction']
                        )
                        all_importances.append(feat_imp)
                        
                        # Perform perturbation analysis
                        perturbation_data = explainer.perturbation_analysis(
                            spikes, explanation['tsa_predicted']
                        )
                        
                        # Save perturbation curve if valid
                        if 'error' not in perturbation_data:
                            fig_perturb = visualizer.plot_perturbation_curve(
                                perturbation_data,
                                title=f'{task.name} - Sample {idx} Perturbation'
                            )
                            perturb_path = Path(exp_dir) / f'sample_{idx}_perturbation.png'
                            fig_perturb.savefig(perturb_path, dpi=150, bbox_inches='tight')
                            plt.close(fig_perturb)
                        
                        # Save JSON data
                        logger.save_explanation(task.name, idx, explanation, None)
                        
                        explanations_generated += 1
                        
                    except Exception as e:
                        logger.log_message(f"  Warning: Could not explain sample {idx}: {e}", level='warning')
                        continue
                
                # Save aggregated feature importance
                if all_importances:
                    mean_importance = np.mean(all_importances, axis=0)
                    logger.save_feature_importance(
                        task.name,
                        mean_importance,
                        task.feature_names[:task.X.shape[2]] if task.feature_names else [f"f{i}" for i in range(len(mean_importance))]
                    )
                
                logger.log_message(f"  Generated {explanations_generated} explanations")
        
        # ====================================================================
        # STEP 6: Save Final Summary
        # ====================================================================
        logger.log_message("=" * 60)
        logger.log_message("STEP 6: Saving Final Summary")
        logger.log_message("=" * 60)
        
        summary = {
            'experiment_name': logger.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'seed': args.seed,
            'num_tasks': len(all_tasks),
            'num_domains': len(tasks_by_domain),
            'domains': list(tasks_by_domain.keys()),
            'meta_training': {
                'epochs': args.epochs,
                'final_loss': float(history['meta_loss'][-1]),
                'final_accuracy': float(np.mean(history['task_accuracies'][-1]))
            },
            'evaluation': {
                'mean_test_accuracy': float(overall_acc),
                'std_test_accuracy': float(np.std([r['test_accuracy'] for r in task_results])),
                'per_domain_accuracy': {
                    domain: float(np.mean([r['test_accuracy'] for r in task_results if r['domain'] == domain]))
                    for domain in tasks_by_domain.keys()
                }
            },
            'model': {
                'input_size': args.input_size,
                'hidden_sizes': args.hidden_sizes,
                'total_parameters': total_params
            },
            'adaptive_windows': {
                'enabled': optimal_windows is not None,
                'windows': optimal_windows
            } if optimal_windows else {'enabled': False}
        }
        
        logger.save_summary(summary)
        
        logger.log_message("=" * 60)
        logger.log_message("PIPELINE COMPLETED SUCCESSFULLY")
        logger.log_message(f"Results saved to: {logger.experiment_dir}")
        logger.log_message("=" * 60)
        
    except Exception as e:
        logger.log_message(f"Pipeline failed with error: {str(e)}", level='error')
        import traceback
        logger.log_message(traceback.format_exc(), level='error')
        raise
    
    finally:
        logger.close()


def main():
    """Main entry point"""
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
