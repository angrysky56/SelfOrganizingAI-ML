"""
Base Simulation Module - Optimized for consumer GPUs
Uses lightweight implementation suitable for GPUs with 6-12GB VRAM
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..utils.gpu_memory import estimate_memory_usage, optimize_batch_size

@dataclass
class SimulationConfig:
    batch_size: int = 32  # Adjustable based on GPU memory
    max_agents: int = 1000
    learning_rate: float = 0.001
    max_iterations: int = 1000
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    memory_limit: float = 5.0  # GB - leaves room for system and other processes

class BaseSimulation:
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.device = torch.device(self.config.device)
        self._validate_resources()
        
    def _validate_resources(self):
        """Check if simulation can run within resource constraints."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            if total_memory < self.config.memory_limit:
                self.config.batch_size = max(8, self.config.batch_size // 2)
                self.config.max_agents = min(500, self.config.max_agents)
                
    async def initialize(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize simulation state."""
        # Convert numpy arrays to torch tensors
        state = {
            k: torch.tensor(v, device=self.device) 
            if isinstance(v, np.ndarray) else v 
            for k, v in initial_state.items()
        }
        
        # Add simulation metadata
        state['metadata'] = {
            'iteration': 0,
            'device': self.config.device,
            'memory_usage': estimate_memory_usage(state)
        }
        
        return state
        
    async def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one simulation step."""
        with torch.no_grad():  # Optimize memory usage
            # Update agent states in batches
            for i in range(0, len(state['agents']), self.config.batch_size):
                batch = state['agents'][i:i + self.config.batch_size]
                # Process batch
                updated_batch = self._process_batch(batch)
                state['agents'][i:i + self.config.batch_size] = updated_batch
                
            # Update simulation metadata
            state['metadata']['iteration'] += 1
            state['metadata']['memory_usage'] = estimate_memory_usage(state)
            
        return state
        
    def _process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Process a batch of agents. Override in subclasses."""
        raise NotImplementedError

    @property
    def memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0

    def cleanup(self):
        """Free GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()