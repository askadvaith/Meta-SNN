"""
Temporal Spike Attribution (TSA) for Explainable SNNs

Refactored to match the reference implementation from 
`v2_tsa_singletask_classification_wo_th.py`.

Key concepts:
- TSA-S (Spike): Contribution when neuron spikes at time t_k: β^(t' - t_k)
- TSA-NS (Non-Spike): Penalty when neuron doesn't spike: -(1/fan_in) * β^(t' - t_k)
- Combined N value for each neuron propagated through weight matrices
- Full attribution chain: diag(N0) @ W1.T @ diag(N1) @ W2.T @ ... @ diag(P)
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json


def compute_N_from_spike_times(
    spike_times_list: List[int], 
    t_prime: int, 
    beta: float, 
    fan_in: int
) -> float:
    """
    Compute the N value for a neuron at time t_prime.
    
    This combines TSA-S (spike contribution) and TSA-NS (non-spike penalty):
    - TSAS: For each spike at t_k <= t', add β^(t' - t_k)
    - TSANS: For each non-spike timestep t_k <= t', subtract (1/fan_in) * β^(t' - t_k)
    
    Args:
        spike_times_list: List of timesteps where the neuron spiked
        t_prime: Current observation timestep
        beta: Membrane decay rate
        fan_in: Number of input connections to this layer
        
    Returns:
        N value (combined spike/non-spike contribution)
    """
    N = 0.0
    
    # TSAS part - contribution from spikes
    for t_k in spike_times_list:
        if t_k <= t_prime:
            N += beta ** (t_prime - t_k)
    
    # TSANS part - penalty for non-spikes
    spike_set = set(spike_times_list)
    for t_k in range(t_prime + 1):
        if t_k not in spike_set:
            N -= (1.0 / fan_in) * (beta ** (t_prime - t_k))
    
    return N


class TSAExplainer:
    """
    Temporal Spike Attribution explainer for MetaSNN models.
    
    Computes attribution maps showing how input spikes across
    time contribute to the model's predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None
    ):
        """
        Initialize the TSA explainer.
        
        Args:
            model: SNN model to explain (MetaSNN or similar)
            device: Computation device
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_tsa(
        self,
        x: torch.Tensor,
        beta: float = None
    ) -> torch.Tensor:
        """
        Compute full TSA attributions for a single sample.
        
        This follows the reference implementation:
        CI_t = diag(N0) @ W1.T @ diag(N1) @ W2.T @ diag(N2) @ W3.T @ diag(P)
        
        Args:
            x: Input spike tensor [timesteps, features] (single sample)
            beta: Membrane decay rate (uses model's beta if None)
            
        Returns:
            TSA attributions [timesteps, features, classes]
        """
        if beta is None:
            beta = self.model.beta
        
        x = x.to(self.device)
        
        # Ensure x is 2D: [timesteps, features]
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Run model forward pass to get spike recordings
        x_batch = x.unsqueeze(1)  # [timesteps, 1, features]
        mem_rec, _, recordings = self.model(x_batch, return_recordings=True)
        
        T = mem_rec.shape[0]  # timesteps
        F = x.shape[1]  # input features
        C = mem_rec.shape[2]  # output classes
        
        # Get hidden layer spike recordings
        h_spk_list = recordings['encoder_spk']  # List of [T, batch, hidden_size]
        
        # Extract weight matrices (transposed for attribution flow)
        weight_matrices = []
        for fc in self.model.encoder_fc:
            weight_matrices.append(fc.weight.t().to(self.device))  # [in, out]
        weight_matrices.append(self.model.head_fc.weight.t().to(self.device))  # [hidden_last, classes]
        
        # Extract spike times for input
        inp_spk = [torch.where(x[:, i] > 0)[0].tolist() for i in range(F)]
        
        # Extract spike times for each hidden layer
        hidden_spk_times = []
        for h_spk in h_spk_list:
            H = h_spk.shape[2]
            layer_spk_times = [torch.where(h_spk[:, 0, i] > 0)[0].tolist() for i in range(H)]
            hidden_spk_times.append(layer_spk_times)
        
        # Layer sizes for fan_in calculation
        layer_sizes = [F] + [h.shape[2] for h in h_spk_list]
        
        # Output membrane potentials
        out_mem = mem_rec.squeeze(1)  # [T, classes]
        
        # Compute attribution for each timestep
        attributions = []
        
        for t in range(T):
            # Softmax of membrane potential at time t
            P_t = torch.softmax(out_mem[t], dim=0)
            P_diag = torch.diag(P_t)
            
            # Compute N values for input layer
            fan1 = F
            N0 = torch.tensor(
                [compute_N_from_spike_times(inp_spk[i], t, beta, fan1) for i in range(F)],
                device=self.device,
                dtype=torch.float32
            )
            
            # Compute N values for hidden layers
            N_hidden = []
            for layer_idx, layer_spk_times in enumerate(hidden_spk_times):
                fan_in = layer_sizes[layer_idx]
                H = len(layer_spk_times)
                N_layer = torch.tensor(
                    [compute_N_from_spike_times(layer_spk_times[i], t, beta, fan_in) for i in range(H)],
                    device=self.device,
                    dtype=torch.float32
                )
                N_hidden.append(N_layer)
            
            # Build the attribution chain: diag(N0) @ W1.T @ diag(N1) @ W2.T @ ... @ diag(P)
            # Start with input layer
            CI_t = torch.diag(N0) @ weight_matrices[0]
            
            # Chain through hidden layers
            for layer_idx, N_layer in enumerate(N_hidden):
                CI_t = CI_t @ torch.diag(N_layer) @ weight_matrices[layer_idx + 1]
            
            # Multiply with class probabilities
            CI_t = CI_t @ P_diag
            
            attributions.append(CI_t)
        
        return torch.stack(attributions)  # [T, F, C]
    
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor
    ) -> Tuple[int, torch.Tensor]:
        """
        Get prediction and membrane potentials for a sample.
        
        Args:
            x: Input spike tensor [timesteps, features]
            
        Returns:
            Tuple of (predicted_class, membrane_sums)
        """
        x = x.to(self.device)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [T, 1, F]
        
        mem_rec, _, _ = self.model(x, return_recordings=True)
        membrane_sums = mem_rec.sum(dim=0)[0].cpu()  # Sum over time
        pred = int(torch.argmax(membrane_sums))
        
        return pred, membrane_sums
    
    def explain_prediction(
        self,
        x: torch.Tensor,
        true_label: int = None
    ) -> Dict[str, Any]:
        """
        Generate complete explanation for a prediction.
        
        Args:
            x: Input spike tensor [timesteps, features]
            true_label: Optional ground truth label
            
        Returns:
            Dictionary with attribution map, predictions, probabilities, etc.
        """
        # Compute TSA
        att_full = self.compute_tsa(x)  # [T, F, C]
        
        # Get prediction
        pred, mem = self.predict(x)
        
        # Get attribution for predicted class
        att_pred = att_full[:, :, pred]  # [T, F]
        
        # Compute probabilities from membrane potentials
        probs = torch.softmax(mem, dim=0).numpy()
        
        return {
            'tsa_map': att_full,  # [T, F, C]
            'tsa_predicted': att_pred,  # [T, F]
            'prediction': pred,
            'true_label': true_label,
            'probabilities': probs,
            'membrane_potentials': mem.numpy(),
            'input_spikes': x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        }
    
    def get_feature_importance(
        self,
        tsa_map: torch.Tensor,
        predicted_class: int = None
    ) -> np.ndarray:
        """
        Aggregate TSA map to get feature importance scores.
        
        Args:
            tsa_map: TSA attribution [T, F, C]
            predicted_class: Class to analyze (None = all classes)
            
        Returns:
            Feature importance scores [F]
        """
        if predicted_class is None:
            # Sum absolute attributions across time and all classes
            importance = tsa_map.abs().sum(dim=(0, 2))
        else:
            # Attribution for specific class, summed over time
            importance = tsa_map[:, :, predicted_class].abs().sum(dim=0)
        
        return importance.detach().cpu().numpy()
    
    def get_temporal_importance(
        self,
        tsa_map: torch.Tensor,
        predicted_class: int = None
    ) -> np.ndarray:
        """
        Aggregate TSA map to get temporal importance pattern.
        
        Args:
            tsa_map: TSA attribution [T, F, C]
            predicted_class: Class to analyze
            
        Returns:
            Temporal importance scores [T]
        """
        if predicted_class is None:
            importance = tsa_map.abs().sum(dim=(1, 2))
        else:
            importance = tsa_map[:, :, predicted_class].abs().sum(dim=1)
        
        return importance.detach().cpu().numpy()
    
    @torch.no_grad()
    def perturbation_analysis(
        self,
        x: torch.Tensor,
        attribution: torch.Tensor,
        k_values: List[int] = None
    ) -> Dict:
        """
        Perform perturbation analysis by deleting top-k vs random-k spikes.
        
        This validates that TSA correctly identifies important spikes by showing
        that removing high-attribution spikes has more impact than random removal.
        
        Args:
            x: Input spike tensor [T, F]
            attribution: TSA attribution for predicted class [T, F]
            k_values: List of k values to test (default: [1, 6, 11, ...])
            
        Returns:
            Dictionary with perturbation analysis results
        """
        x = x.to(self.device)
        T, F = attribution.shape
        
        # Flatten attribution and find spike positions
        flat_imp = attribution.abs().flatten().cpu()
        x_cpu = x.cpu()
        spike_mask = (x_cpu > 0.5).flatten()
        spike_indices = torch.nonzero(spike_mask).flatten()
        
        if spike_indices.numel() == 0:
            return {'error': 'No spikes to test perturbation'}
        
        if k_values is None:
            k_values = list(range(1, min(50, spike_indices.numel()), 5))
        
        # Baseline prediction
        base_pred, base_mem = self.predict(x)
        base_val = base_mem[base_pred].item()
        
        perturbation_data = {
            'baseline': {
                'predicted_class': int(base_pred),
                'membrane_potential': float(base_val)
            },
            'k_values': k_values,
            'top_k_effects': [],
            'random_k_effects': []
        }
        
        for k in k_values:
            if k > spike_indices.numel():
                break
            
            # Top-K deletion (most important spikes)
            imp_values = flat_imp[spike_indices]
            top_order = torch.topk(imp_values, k).indices
            top_idx = spike_indices[top_order]
            
            x_edit = x.clone()
            for idx in top_idx:
                t = (idx // F).item()
                f = (idx % F).item()
                x_edit[t, f] = 0.0
            
            pred_edit, mem_edit = self.predict(x_edit)
            top_k_effect = base_val - mem_edit[base_pred].item()
            perturbation_data['top_k_effects'].append(float(top_k_effect))
            
            # Random-K deletion
            rand_perm = spike_indices[torch.randperm(spike_indices.numel())[:k]]
            x_rand = x.clone()
            for idx in rand_perm:
                t = (idx // F).item()
                f = (idx % F).item()
                x_rand[t, f] = 0.0
            
            pred_rand, mem_rand = self.predict(x_rand)
            rand_k_effect = base_val - mem_rand[base_pred].item()
            perturbation_data['random_k_effects'].append(float(rand_k_effect))
        
        # Summary statistics
        if perturbation_data['top_k_effects']:
            perturbation_data['summary'] = {
                'top_k_avg_drop': float(np.mean(perturbation_data['top_k_effects'])),
                'random_k_avg_drop': float(np.mean(perturbation_data['random_k_effects'])),
                'top_k_more_effective': bool(
                    np.mean(perturbation_data['top_k_effects']) > 
                    np.mean(perturbation_data['random_k_effects'])
                )
            }
        
        return perturbation_data


class TSAVisualizer:
    """
    Visualization utilities for TSA explanations.
    """
    
    def __init__(
        self,
        feature_names: List[str] = None,
        class_names: List[str] = None
    ):
        """
        Initialize visualizer.
        
        Args:
            feature_names: Names of input features
            class_names: Names of output classes
        """
        self.feature_names = feature_names
        self.class_names = class_names
    
    def plot_heatmap(
        self,
        att2d: torch.Tensor,
        feature_names: List[str] = None,
        title: str = "TSA Attribution",
        save_path: str = None,
        figsize: Tuple[int, int] = (14, 6),
        show: bool = False
    ) -> plt.Figure:
        """
        Plot attribution heatmap.
        
        Args:
            att2d: Attribution matrix [T, F]
            feature_names: Feature names for y-axis
            title: Plot title
            save_path: Path to save figure
            figsize: Figure size
            show: Whether to display plot
            
        Returns:
            Matplotlib figure
        """
        feature_names = feature_names or self.feature_names
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(att2d.shape[1])]
        
        arr = att2d.cpu().numpy() if isinstance(att2d, torch.Tensor) else att2d
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            arr.T,  # Transpose so features are on y-axis
            cmap="coolwarm",
            center=0,
            xticklabels=50,
            yticklabels=feature_names,
            ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_perturbation_curve(
        self,
        perturbation_data: Dict,
        title: str = "Perturbation Analysis",
        save_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot perturbation analysis results.
        
        Args:
            perturbation_data: Results from TSAExplainer.perturbation_analysis()
            title: Plot title
            save_path: Path to save figure
            show: Whether to display plot
            
        Returns:
            Matplotlib figure
        """
        k_values = perturbation_data['k_values'][:len(perturbation_data['top_k_effects'])]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(k_values, perturbation_data['top_k_effects'], 
                label="Top-K deletion", linewidth=2, marker='o')
        ax.plot(k_values, perturbation_data['random_k_effects'], 
                label="Random-K deletion", linewidth=2, marker='s')
        ax.set_xlabel("K (number of deleted spikes)")
        ax.set_ylabel("Reduction in membrane potential")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_feature_importance(
        self,
        importance: np.ndarray,
        feature_names: List[str] = None,
        title: str = "Feature Importance",
        save_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Plot feature importance bar chart.
        
        Args:
            importance: Importance scores [F]
            feature_names: Feature names
            title: Plot title
            save_path: Path to save figure
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        feature_names = feature_names or self.feature_names
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(sorted_importance)), sorted_importance)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel("Importance Score")
        ax.set_title(title)
        ax.invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_explanation_figure(
        self,
        explanation: Dict,
        feature_names: List[str] = None,
        class_names: List[str] = None,
        title: str = None,
        save_path: str = None,
        show: bool = False
    ) -> plt.Figure:
        """
        Create comprehensive explanation figure with multiple panels.
        
        Args:
            explanation: Output from TSAExplainer.explain_prediction()
            feature_names: Feature names
            class_names: Class names
            title: Overall title
            save_path: Path to save
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        feature_names = feature_names or self.feature_names
        class_names = class_names or self.class_names
        
        pred = explanation['prediction']
        true_label = explanation.get('true_label')
        probs = explanation['probabilities']
        tsa_pred = explanation['tsa_predicted']
        
        # Get class name for predicted class
        pred_name = class_names[pred] if class_names else f"Class {pred}"
        true_name = class_names[true_label] if (class_names and true_label is not None) else None
        
        fig = plt.figure(figsize=(16, 10))
        
        # TSA Heatmap (main panel)
        ax1 = fig.add_subplot(2, 2, (1, 2))
        arr = tsa_pred.cpu().numpy() if isinstance(tsa_pred, torch.Tensor) else tsa_pred
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(arr.shape[1])]
        sns.heatmap(
            arr.T,
            cmap="coolwarm",
            center=0,
            xticklabels=50,
            yticklabels=feature_names,
            ax=ax1
        )
        title_str = f"TSA Attribution (Predicted: {pred_name})"
        if true_name:
            title_str += f" | True: {true_name}"
        ax1.set_title(title_str)
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Feature")
        
        # Feature importance
        ax2 = fig.add_subplot(2, 2, 3)
        importance = np.abs(arr).sum(axis=0)
        sorted_idx = np.argsort(importance)[::-1]
        ax2.barh(range(len(importance)), importance[sorted_idx])
        ax2.set_yticks(range(len(feature_names)))
        ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax2.set_xlabel("Cumulative |Attribution|")
        ax2.set_title("Feature Importance")
        ax2.invert_yaxis()
        
        # Class probabilities
        ax3 = fig.add_subplot(2, 2, 4)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(probs))]
        colors = ['green' if i == pred else 'steelblue' for i in range(len(probs))]
        ax3.bar(class_names, probs, color=colors)
        ax3.set_ylabel("Probability")
        ax3.set_title("Class Probabilities")
        ax3.set_ylim(0, 1)
        
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig


def generate_explanations_for_task(
    model: nn.Module,
    task_data: Dict,
    task_name: str,
    save_dir: Path,
    num_samples: int = 5,
    feature_names: List[str] = None,
    class_names: List[str] = None,
    device: torch.device = None
) -> List[Dict]:
    """
    Generate TSA explanations for multiple samples from a task.
    
    Args:
        model: Trained SNN model
        task_data: Dictionary with 'X_test', 'y_test' tensors
        task_name: Name of the task for saving
        save_dir: Directory to save explanations
        num_samples: Number of samples to explain
        feature_names: Feature names
        class_names: Class names
        device: Computation device
        
    Returns:
        List of explanation dictionaries
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize explainer and visualizer
    explainer = TSAExplainer(model, device)
    visualizer = TSAVisualizer(feature_names, class_names)
    
    # Get test data
    X_test = task_data['X_test']
    y_test = task_data['y_test']
    
    # Select samples to explain
    num_samples = min(num_samples, len(X_test))
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    explanations = []
    
    for i, idx in enumerate(indices):
        # Get sample (handle both [T, F] and [T, B, F] formats)
        x = X_test[idx]
        if x.dim() == 3:
            x = x.squeeze(1)
        true_label = int(y_test[idx])
        
        # Generate explanation
        explanation = explainer.explain_prediction(x, true_label)
        
        # Create visualization
        fig = visualizer.create_explanation_figure(
            explanation,
            feature_names=feature_names,
            class_names=class_names,
            title=f"{task_name} - Sample {idx}"
        )
        
        # Save figure
        fig_path = save_dir / f"sample_{idx}_explanation.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Compute and save feature importance
        importance = explainer.get_feature_importance(
            explanation['tsa_map'],
            explanation['prediction']
        )
        
        # Perform perturbation analysis
        perturbation_data = explainer.perturbation_analysis(
            x, explanation['tsa_predicted']
        )
        
        # Save perturbation curve
        if 'error' not in perturbation_data:
            fig_perturb = visualizer.plot_perturbation_curve(
                perturbation_data,
                title=f"{task_name} - Sample {idx} Perturbation Analysis"
            )
            perturb_path = save_dir / f"sample_{idx}_perturbation.png"
            fig_perturb.savefig(perturb_path, dpi=150, bbox_inches='tight')
            plt.close(fig_perturb)
        
        # Store results
        result = {
            'sample_idx': int(idx),
            'true_label': true_label,
            'prediction': explanation['prediction'],
            'correct': explanation['prediction'] == true_label,
            'probabilities': explanation['probabilities'].tolist(),
            'feature_importance': importance.tolist(),
            'perturbation': perturbation_data
        }
        explanations.append(result)
        
        # Save individual JSON
        json_path = save_dir / f"sample_{idx}_data.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save aggregated feature importance
    if explanations:
        avg_importance = np.mean([e['feature_importance'] for e in explanations], axis=0)
        fig_imp = visualizer.plot_feature_importance(
            avg_importance,
            feature_names=feature_names,
            title=f"{task_name} - Average Feature Importance"
        )
        fig_imp.savefig(save_dir / "feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close(fig_imp)
        
        # Save feature importance JSON
        with open(save_dir / "feature_importance.json", 'w') as f:
            json.dump({
                'feature_names': feature_names or [f"f{i}" for i in range(len(avg_importance))],
                'average_importance': avg_importance.tolist(),
                'per_sample_importance': [e['feature_importance'] for e in explanations]
            }, f, indent=2)
    
    return explanations
