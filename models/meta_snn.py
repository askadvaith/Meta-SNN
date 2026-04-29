"""
Meta-Learning Spiking Neural Network Architecture

Based on "Robust Spike-Based Continual Meta-Learning" paper concepts:
- LIF (Leaky Integrate-and-Fire) neurons with surrogate gradients
- Flexible architecture supporting different task head sizes
- Compatible with Reptile/MAML meta-learning algorithms
- Records spike trains for TSA explainability

Architecture:
- Shared encoder: Input -> Hidden1 -> Hidden2 (transferable representations)
- Task-specific head: Hidden2 -> Output (adaptable per task)
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from typing import Dict, List, Tuple, Optional
import copy


class MetaSNN(nn.Module):
    """
    Meta-Learning compatible Spiking Neural Network.
    
    Features:
    - Shared encoder layers that learn task-agnostic representations
    - Flexible output head for different number of classes
    - Records all spike/membrane activity for explainability
    - Compatible with gradient-based meta-learning (MAML, Reptile)
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_sizes: List[int] = [64, 32],
        output_size: int = 3,
        beta: float = 0.9,
        spike_grad: str = 'fast_sigmoid',
        dropout: float = 0.0
    ):
        """
        Initialize the Meta-SNN.
        
        Args:
            input_size: Number of input features (sensors)
            hidden_sizes: List of hidden layer sizes [hidden1, hidden2, ...]
            output_size: Number of output classes
            beta: Membrane potential decay rate (0 < beta < 1)
            spike_grad: Surrogate gradient function ('fast_sigmoid', 'atan', 'straight_through')
            dropout: Dropout probability for regularization
        """
        super(MetaSNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.beta = beta
        
        # Store layer sizes for TSA
        self.layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.nb_layers = len(hidden_sizes) + 1  # hidden + output
        
        # Surrogate gradient
        spike_grad_fn = self._get_spike_grad(spike_grad)
        
        # Build encoder layers
        self.encoder_fc = nn.ModuleList()
        self.encoder_lif = nn.ModuleList()
        
        prev_size = input_size
        for h_size in hidden_sizes:
            self.encoder_fc.append(nn.Linear(prev_size, h_size))
            self.encoder_lif.append(snn.Leaky(
                beta=beta,
                spike_grad=spike_grad_fn,
                learn_beta=True  # Enable learnable beta for adaptive temporal dynamics
            ))
            prev_size = h_size
        
        # Output head (task-specific)
        self.head_fc = nn.Linear(prev_size, output_size)
        self.head_lif = snn.Leaky(
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
        for layer in self.encoder_fc:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.head_fc.weight)
        nn.init.zeros_(self.head_fc.bias)
    
    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize membrane potentials for all layers"""
        mems = []
        for lif in self.encoder_lif:
            mems.append(lif.init_leaky())
        mems.append(self.head_lif.init_leaky())
        return mems
    
    def forward(
        self,
        x: torch.Tensor,
        return_recordings: bool = True
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
        for lif in self.encoder_lif:
            mems.append(lif.init_leaky())
        mems.append(self.head_lif.init_leaky())
        
        # Recording containers
        if return_recordings:
            recordings = {
                'input': x,
                'encoder_spk': [[] for _ in range(len(self.encoder_fc))],
                'encoder_mem': [[] for _ in range(len(self.encoder_fc))],
                'output_spk': [],
                'output_mem': []
            }
        
        # Process timesteps
        output_spk_rec = []
        output_mem_rec = []
        
        for step in range(num_steps):
            x_step = x[step]
            
            # Encoder layers
            for i, (fc, lif) in enumerate(zip(self.encoder_fc, self.encoder_lif)):
                cur = fc(x_step)
                if self.dropout is not None:
                    cur = self.dropout(cur)
                spk, mems[i] = lif(cur, mems[i])
                x_step = spk
                
                if return_recordings:
                    recordings['encoder_spk'][i].append(spk)
                    recordings['encoder_mem'][i].append(mems[i])
            
            # Output head
            cur = self.head_fc(x_step)
            spk, mems[-1] = self.head_lif(cur, mems[-1])
            
            output_spk_rec.append(spk)
            output_mem_rec.append(mems[-1])
            
            if return_recordings:
                recordings['output_spk'].append(spk)
                recordings['output_mem'].append(mems[-1])
        
        # Stack recordings
        output_spk = torch.stack(output_spk_rec)
        output_mem = torch.stack(output_mem_rec)
        
        if return_recordings:
            for i in range(len(self.encoder_fc)):
                recordings['encoder_spk'][i] = torch.stack(recordings['encoder_spk'][i])
                recordings['encoder_mem'][i] = torch.stack(recordings['encoder_mem'][i])
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
    
    def clone(self) -> 'MetaSNN':
        """Create a deep copy of the model (for meta-learning)"""
        return copy.deepcopy(self)
    
    def get_weights(self) -> List[torch.Tensor]:
        """Get weight matrices for TSA explainability"""
        weights = []
        for fc in self.encoder_fc:
            weights.append(fc.weight.data.T)  # Transpose for forward direction
        weights.append(self.head_fc.weight.data.T)
        return weights
    
    @property
    def weights(self) -> List[torch.Tensor]:
        """Property to get weights (for TSA compatibility)"""
        return self.get_weights()
    
    def adapt_head(self, new_output_size: int, device: torch.device = None):
        """
        Adapt the output head for a new number of classes.
        
        Args:
            new_output_size: New number of output classes
            device: Device to place new layer on
        """
        if new_output_size == self.output_size:
            return
        
        old_head = self.head_fc
        self.output_size = new_output_size
        self.layer_sizes[-1] = new_output_size
        
        # Create new head
        self.head_fc = nn.Linear(self.hidden_sizes[-1], new_output_size)
        nn.init.xavier_uniform_(self.head_fc.weight)
        nn.init.zeros_(self.head_fc.bias)
        
        if device is not None:
            self.head_fc = self.head_fc.to(device)


class MetaSNNWithTaskHead(nn.Module):
    """
    Meta-SNN with separate task-specific heads.
    
    This variant maintains a shared encoder and can have multiple
    task-specific output heads for different classification tasks.
    """
    
    def __init__(
        self,
        input_size: int = 10,
        hidden_sizes: List[int] = [64, 32],
        beta: float = 0.9,
        spike_grad: str = 'fast_sigmoid',
        dropout: float = 0.0
    ):
        """
        Initialize the Meta-SNN with task heads.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            beta: Membrane potential decay rate
            spike_grad: Surrogate gradient function
            dropout: Dropout probability
        """
        super(MetaSNNWithTaskHead, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.beta = beta
        
        # Surrogate gradient
        spike_grad_fn = self._get_spike_grad(spike_grad)
        
        # Shared encoder
        self.encoder_fc = nn.ModuleList()
        self.encoder_lif = nn.ModuleList()
        
        prev_size = input_size
        for h_size in hidden_sizes:
            self.encoder_fc.append(nn.Linear(prev_size, h_size))
            self.encoder_lif.append(snn.Leaky(
                beta=beta,
                spike_grad=spike_grad_fn
            ))
            prev_size = h_size
        
        # Task heads dictionary (task_name -> head)
        self.task_heads = nn.ModuleDict()
        self.task_lifs = nn.ModuleDict()
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else None
        
        # For TSA compatibility
        self.layer_sizes = [input_size] + hidden_sizes
        self.nb_layers = len(hidden_sizes)
        
        self._init_weights()
    
    def _get_spike_grad(self, name: str):
        """Get surrogate gradient function"""
        if name == 'fast_sigmoid':
            return surrogate.fast_sigmoid(slope=25)
        elif name == 'atan':
            return surrogate.atan(alpha=2.0)
        else:
            return surrogate.fast_sigmoid(slope=25)
    
    def _init_weights(self):
        """Initialize encoder weights"""
        for layer in self.encoder_fc:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def add_task_head(self, task_name: str, num_classes: int, device: torch.device = None):
        """
        Add a new task-specific head.
        
        Args:
            task_name: Unique identifier for the task
            num_classes: Number of output classes for this task
            device: Device to place the new head on
        """
        head = nn.Linear(self.hidden_sizes[-1], num_classes)
        nn.init.xavier_uniform_(head.weight)
        nn.init.zeros_(head.bias)
        
        lif = snn.Leaky(beta=self.beta, spike_grad=surrogate.fast_sigmoid(slope=25))
        
        if device is not None:
            head = head.to(device)
        
        self.task_heads[task_name] = head
        self.task_lifs[task_name] = lif
    
    def forward(
        self,
        x: torch.Tensor,
        task_name: str,
        return_recordings: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass with task-specific head.
        
        Args:
            x: Input spike train [timesteps, batch_size, input_size]
            task_name: Name of the task (determines which head to use)
            return_recordings: Whether to return recordings
            
        Returns:
            mem_out, spk_out, recordings
        """
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}. Add it first with add_task_head()")
        
        num_steps = x.size(0)
        
        # Initialize membrane potentials
        mems = [lif.init_leaky() for lif in self.encoder_lif]
        head_mem = self.task_lifs[task_name].init_leaky()
        
        # Recordings
        if return_recordings:
            recordings = {
                'input': x,
                'encoder_spk': [[] for _ in range(len(self.encoder_fc))],
                'encoder_mem': [[] for _ in range(len(self.encoder_fc))],
                'output_spk': [],
                'output_mem': []
            }
        
        output_spk_rec = []
        output_mem_rec = []
        
        for step in range(num_steps):
            x_step = x[step]
            
            # Encoder
            for i, (fc, lif) in enumerate(zip(self.encoder_fc, self.encoder_lif)):
                cur = fc(x_step)
                if self.dropout is not None:
                    cur = self.dropout(cur)
                spk, mems[i] = lif(cur, mems[i])
                x_step = spk
                
                if return_recordings:
                    recordings['encoder_spk'][i].append(spk)
                    recordings['encoder_mem'][i].append(mems[i])
            
            # Task-specific head
            cur = self.task_heads[task_name](x_step)
            spk, head_mem = self.task_lifs[task_name](cur, head_mem)
            
            output_spk_rec.append(spk)
            output_mem_rec.append(head_mem)
            
            if return_recordings:
                recordings['output_spk'].append(spk)
                recordings['output_mem'].append(head_mem)
        
        output_spk = torch.stack(output_spk_rec)
        output_mem = torch.stack(output_mem_rec)
        
        if return_recordings:
            for i in range(len(self.encoder_fc)):
                recordings['encoder_spk'][i] = torch.stack(recordings['encoder_spk'][i])
                recordings['encoder_mem'][i] = torch.stack(recordings['encoder_mem'][i])
            recordings['output_spk'] = output_spk
            recordings['output_mem'] = output_mem
            return output_mem, output_spk, recordings
        
        return output_mem, output_spk, None
    
    def get_output_logits(self, mem_rec: torch.Tensor) -> torch.Tensor:
        """Convert membrane recordings to logits"""
        return mem_rec.sum(dim=0)
    
    def get_encoder_params(self):
        """Get encoder parameters only"""
        params = []
        for fc in self.encoder_fc:
            params.extend(fc.parameters())
        return params
    
    def get_head_params(self, task_name: str):
        """Get task head parameters"""
        if task_name not in self.task_heads:
            raise ValueError(f"Unknown task: {task_name}")
        return self.task_heads[task_name].parameters()


def create_meta_snn(
    input_size: int = 10,
    hidden_sizes: List[int] = [64, 32],
    output_size: int = 3,
    beta: float = 0.9,
    device: torch.device = None
) -> MetaSNN:
    """
    Factory function to create a MetaSNN model.
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes
        output_size: Number of output classes
        beta: Membrane decay rate
        device: Device to place model on
        
    Returns:
        Initialized MetaSNN model
    """
    model = MetaSNN(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        beta=beta
    )
    
    if device is not None:
        model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Create model
    model = create_meta_snn(
        input_size=10,
        hidden_sizes=[64, 32],
        output_size=3,
        beta=0.9,
        device=device
    )
    
    print(f"\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 8
    timesteps = 100
    input_size = 10
    
    # Create random input spikes
    x = torch.rand(timesteps, batch_size, input_size).to(device)
    x = (x > 0.5).float()  # Binary spikes
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    mem_out, spk_out, recordings = model(x, return_recordings=True)
    
    print(f"Output membrane shape: {mem_out.shape}")
    print(f"Output spikes shape: {spk_out.shape}")
    
    # Get logits
    logits = model.get_output_logits(mem_out)
    print(f"Logits shape: {logits.shape}")
    
    # Test cloning (for meta-learning)
    model_clone = model.clone()
    print(f"\nModel cloned successfully")
    
    # Test head adaptation
    model.adapt_head(6, device)
    print(f"Head adapted to 6 classes")
    
    mem_out_new, _, _ = model(x, return_recordings=False)
    logits_new = model.get_output_logits(mem_out_new)
    print(f"New logits shape: {logits_new.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
