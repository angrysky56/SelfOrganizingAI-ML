"""
Base Analyzer Module for Ethical AI Systems
Implements core analyzer class and common utilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
import torch

@dataclass
class AnalyzerConfig:
    """Configuration for analyzers."""
    min_samples: int = 100
    confidence_threshold: float = 0.9
    variance_weight: float = 0.2
    bias_threshold: float = 0.1
    fairness_threshold: float = 0.8
    batch_size: int = 32
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_memory_usage: float = 0.9  # Maximum fraction of GPU memory to use

@dataclass
class AnalyzerContext:
    """Context information for analysis."""
    timestamp: datetime
    config: AnalyzerConfig
    metadata: Optional[Dict[str, Any]] = None
    
@dataclass
class AnalyzerResult:
    """Container for analysis results."""
    value: float
    confidence: float
    context: AnalyzerContext
    details: Optional[Dict[str, Any]] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
    
    def validate(self) -> bool:
        """Validate result constraints."""
        valid = (
            0.0 <= self.value <= 1.0 and 
            0.0 <= self.confidence <= 1.0
        )
        return valid
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers."""
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize analyzer with configuration.
        Args:
            config: Optional analyzer configuration
        """
        self.config = config or AnalyzerConfig()
        self.device = torch.device(self.config.device)
        self._validate_config()
        self._initialize_metrics()
    
    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> AnalyzerResult:
        """
        Execute core analysis logic.
        Args:
            data: Input data for analysis
        Returns:
            AnalyzerResult containing analysis outcome
        Raises:
            ValueError: If input data is invalid
        """
        pass
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.config.min_samples < 1:
            raise ValueError("min_samples must be positive")
        if not 0 < self.config.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in (0, 1]")
        if not 0 <= self.config.variance_weight <= 1.0:
            raise ValueError("variance_weight must be in [0, 1]")
    
    def _initialize_metrics(self):
        """Initialize tracking metrics."""
        self.metrics = {
            'processed_samples': 0,
            'warnings_count': 0,
            'last_confidence': 0.0,
            'accumulated_value': 0.0
        }
    
    def _compute_confidence(self, 
                          sample_size: int,
                          variance: float) -> float:
        """
        Compute statistical confidence score.
        Args:
            sample_size: Size of analyzed sample
            variance: Measured variance
        Returns:
            float: Confidence score [0.0 - 1.0]
        """
        if sample_size < self.config.min_samples:
            return 0.0
            
        # Base confidence from sample size
        base_confidence = np.clip(
            sample_size / (2 * self.config.min_samples),
            0.0,
            1.0
        )
        
        # Variance penalty
        variance_factor = np.exp(-variance * self.config.variance_weight)
        
        # Combined confidence
        confidence = base_confidence * variance_factor
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def _check_memory_usage(self) -> Tuple[bool, float]:
        """
        Check if memory usage is within limits.
        Returns:
            Tuple[bool, float]: (is_safe, current_usage_fraction)
        """
        if not torch.cuda.is_available():
            return True, 0.0
            
        current = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        is_safe = current <= self.config.max_memory_usage
        
        return is_safe, current
    
    def _create_context(self, metadata: Optional[Dict[str, Any]] = None) -> AnalyzerContext:
        """Create analysis context."""
        return AnalyzerContext(
            timestamp=datetime.now(),
            config=self.config,
            metadata=metadata
        )
    
    def _update_metrics(self, result: AnalyzerResult):
        """Update tracking metrics."""
        self.metrics['processed_samples'] += 1
        self.metrics['warnings_count'] += len(result.warnings)
        self.metrics['last_confidence'] = result.confidence
        self.metrics['accumulated_value'] += result.value
    
    async def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        avg_value = (
            self.metrics['accumulated_value'] / 
            max(1, self.metrics['processed_samples'])
        )
        
        return {
            'samples': self.metrics['processed_samples'],
            'warnings': self.metrics['warnings_count'],
            'last_confidence': self.metrics['last_confidence'],
            'average_value': avg_value
        }
    
    def cleanup(self):
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()