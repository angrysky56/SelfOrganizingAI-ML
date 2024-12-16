"""GPU memory management utilities."""

import torch
from typing import Dict, Any

def estimate_memory_usage(state: Dict[str, Any]) -> float:
    """Estimate memory usage of state in GB."""
    total_bytes = 0
    
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            total_bytes += value.element_size() * value.nelement()
        elif isinstance(value, dict):
            total_bytes += estimate_memory_usage(value)
            
    return total_bytes / 1e9  # Convert to GB

def optimize_batch_size(total_samples: int, target_memory_gb: float = 4.0) -> int:
    """Calculate optimal batch size based on available memory."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Leave some headroom for other operations
        available_memory = min(target_memory_gb * 1e9, total_memory * 0.8)
        
        # Estimate memory per sample (very rough estimate)
        bytes_per_sample = 1024 * 1024  # 1MB per sample as baseline
        
        optimal_batch = int(available_memory / bytes_per_sample)
        return min(optimal_batch, total_samples)
    
    return 32  # Default batch size for CPU