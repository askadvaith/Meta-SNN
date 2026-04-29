"""
Baseline Spiking Neural Network for Standard Training

A simple 128-64 SNN architecture for traditional (non-meta-learning) training.
Used to establish baseline performance for comparison with meta-learning approaches.

Architecture:
- Input Layer -> 128 hidden neurons (LIF) -> 64 hidden neurons (LIF) -> Output
- Uses same LIF neurons and surrogate gradients as meta-learning model
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Dict, List, Tuple, Optional
import copy


class BaselineSNN(nn.Module):
    """
    Standard Spiking Neural Network for baseline comparison.
    
    This is a simple feedforward SNN without meta-learning components.
    Uses the same architecture (128-64) and LIF dynamics as the meta-learning
    model to ensure fair comparison.
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_sizes: List[int] = [128, 64],
        output_size: int = 4,
        beta: float = 0.9,
        spike_grad: str = 'fast_sigmoid',
        dropout: float = 0.1
    ):
        """
        Initialize the Baseline SNN.
        
        Args:
            input_size: Number of input features (sensors)
            hidden_sizes: List of hidden layer sizes [128, 64]
            output_size: Number of output classes
            beta: Membrane potential decay rate (0 < beta < 1)
            spike_grad: Surrogate gradient function ('fast_sigmoid', 'atan', 'straight_through')
            dropout: Dropout probability for regularization
        """
        super(BaselineSNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.beta = beta
        
        # Store layer sizes for compatibility
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.nb_layers = len(hidden_sizes) + 1  # hidden + output
        
        # Surrogate gradient
        spike_grad_fn = self._get_spike_grad(spike_grad)
        
        # Build hidden layers
        self.fc_layers = nn.ModuleList()
        self.lif_layers = nn.ModuleList()
        
        prev_size = input_size
        for h_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(prev_size, h_size))
            self.lif_layers.append(snn.Leaky(
                beta=beta,
                spike_grad=spike_grad_fn,
                learn_beta=True  # Enable learnable beta for adaptive temporal dynamics
            ))
            prev_size = h_size
        
        # Output layer
        self.fc_out = nn.Linear(prev_size, output_size)
        self.lif_out = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad_fn,
            learn_beta=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        
        # Initialize weights
        self._init_weights()
    
    def _get_spike_grad(self, name: str):
        """Get surrogate gradient function by name"""
        if name == 'fast_sigmoid':
            return surrogate.fast_sigmoid(slope=25)
        elif name == 'atan':
            return surrogate.atan(alpha=2.0)
        elif name == 'straight_through':
            return surrogate.straight_through_estimator()
        else:
            return surrogate.fast_sigmoid(slope=25)
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for layer in self.fc_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize membrane potentials for all layers"""
        mems = []
        for lif in self.lif_layers:
            mems.append(lif.init_leaky())
        mems.append(self.lif_out.init_leaky())
        return mems
    
    def forward(
        self,
        x: torch.Tensor,
        return_recordings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through the SNN.
        
        Args:
            x: Input spike train [timesteps, batch_size, input_size]
            return_recordings: Whether to return spike/membrane recordings
            
        Returns:
            mem_out: Output membrane potentials [timesteps, batch, output_size]
            spk_out: Output spikes [timesteps, batch, output_size]
            recordings: Dict with spike/membrane recordings per layer (if return_recordings)
        """
        batch_size = x.size(1)
        device = x.device
        num_steps = x.size(0)
        
        # Initialize membrane potentials
        mems = []
        for lif in self.lif_layers:
            mems.append(lif.init_leaky())
        mems.append(self.lif_out.init_leaky())
        
        # Recording containers
        if return_recordings:
            recordings = {
                'input': x,
                'hidden_spk': [[] for _ in range(len(self.fc_layers))],
                'hidden_mem': [[] for _ in range(len(self.fc_layers))],
                'output_spk': [],
                'output_mem': []
            }
        
        # Process timesteps
        output_spk_rec = []
        output_mem_rec = []
        
        for step in range(num_steps):
            x_step = x[step]
            
            # Hidden layers
            for i, (fc, lif) in enumerate(zip(self.fc_layers, self.lif_layers)):
                cur = fc(x_step)
                if self.dropout is not None:
                    cur = self.dropout(cur)
                spk, mems[i] = lif(cur, mems[i])
                x_step = spk
                
                if return_recordings:
                    recordings['hidden_spk'][i].append(spk)
                    recordings['hidden_mem'][i].append(mems[i])
            
            # Output layer
            cur = self.fc_out(x_step)
            spk, mems[-1] = self.lif_out(cur, mems[-1])
            
            output_spk_rec.append(spk)
            output_mem_rec.append(mems[-1])
            
            if return_recordings:
                recordings['output_spk'].append(spk)
                recordings['output_mem'].append(mems[-1])
        
        # Stack recordings
        output_spk = torch.stack(output_spk_rec)
        output_mem = torch.stack(output_mem_rec)
        
        if return_recordings:
            for i in range(len(self.fc_layers)):
                recordings['hidden_spk'][i] = torch.stack(recordings['hidden_spk'][i])
                recordings['hidden_mem'][i] = torch.stack(recordings['hidden_mem'][i])
            recordings['output_spk'] = output_spk
            recordings['output_mem'] = output_mem
            return output_mem, output_spk, recordings
        
        return output_mem, output_spk, None
    
    def get_output_logits(self, mem_rec: torch.Tensor) -> torch.Tensor:
        """
        Convert membrane potential recordings to classification logits.
        
        Args:
            mem_rec: Membrane recordings [timesteps, batch, output_size]
            
        Returns:
            logits: Classification logits [batch, output_size]
        """
        # Sum membrane potentials over time
        return mem_rec.sum(dim=0)
    
    def clone(self) -> 'BaselineSNN':
        """Create a deep copy of the model"""
        return copy.deepcopy(self)
    
    def get_weights(self) -> List[torch.Tensor]:
        """Get weight matrices"""
        weights = []
        for fc in self.fc_layers:
            weights.append(fc.weight.data.T)  # Transpose for forward direction
        weights.append(self.fc_out.weight.data.T)
        return weights
    
    @property
    def weights(self) -> List[torch.Tensor]:
        """Property to get weights"""
        return self.get_weights()


def create_baseline_snn(
    input_size: int = 10,
    hidden_sizes: List[int] = [128, 64],
    output_size: int = 4,
    beta: float = 0.9,
    dropout: float = 0.1,
    device: torch.device = None
) -> BaselineSNN:
    """
    Factory function to create a baseline SNN.
    
    Args:
        input_size: Number of input features
        hidden_sizes: Hidden layer sizes [128, 64]
        output_size: Number of output classes
        beta: Membrane decay rate
        dropout: Dropout probability
        device: Target device
        
    Returns:
        Initialized BaselineSNN model
    """
    model = BaselineSNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        beta=beta,
        dropout=dropout
    )
    
    if device is not None:
        model = model.to(device)
    
    return model
