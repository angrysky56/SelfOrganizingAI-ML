"""
Core Evaluators Module: Base Classes for Ethical Evaluation

Provides foundational abstractions and interfaces for implementing
ethical evaluation components in self-organizing AI systems.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union
import numpy as np

class EvaluationDomain(Enum):
    """Classification of ethical evaluation domains."""
    FAIRNESS = "fairness"
    BIAS = "bias"
    SAFETY = "safety"
    TRANSPARENCY = "transparency"
    PRIVACY = "privacy"

@dataclass
class EvaluationMetric:
    """
    Base structure for evaluation metrics.
    
    Attributes:
        domain: Evaluation domain (fairness, bias, etc.)
        value: Computed metric value [0.0 - 1.0]
        confidence: Statistical confidence in measurement
        timestamp: Evaluation timestamp
        metadata: Additional metric-specific information
    """
    domain: EvaluationDomain
    value: float
    confidence: float
    timestamp: datetime
    metadata: Optional[Dict] = None

class BaseEvaluator(ABC):
    """
    Abstract base class for ethical evaluators.
    
    Provides common functionality and interfaces for implementing
    specific evaluation components like fairness or bias detection.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or self._default_config()
        self.evaluation_history: List[EvaluationMetric] = []
        
    @abstractmethod
    async def evaluate(self, 
                      system_state: Dict,
                      context: Optional[Dict] = None) -> EvaluationMetric:
        """
        Execute evaluation logic for specific ethical domain.
        
        Args:
            system_state: Current system state and metrics
            context: Optional contextual information
            
        Returns:
            EvaluationMetric containing results and confidence
        """
        pass
    
    @abstractmethod
    def _default_config(self) -> Dict:
        """Define default configuration parameters."""
        pass
    
    def _compute_confidence(self,
                          sample_size: int,
                          variance: float,
                          complexity: float) -> float:
        """
        Compute statistical confidence in evaluation results.
        
        Args:
            sample_size: Size of evaluated sample
            variance: Measured variance in metrics
            complexity: Complexity of evaluation context
            
        Returns:
            float: Confidence score [0.0 - 1.0]
        """
        # Base confidence from sample size
        size_confidence = min(sample_size / self.config.get('ideal_sample_size', 1000), 1.0)
        
        # Variance penalty
        variance_factor = np.exp(-variance * self.config.get('variance_sensitivity', 2))
        
        # Complexity penalty
        complexity_factor = 1 / (1 + complexity * self.config.get('complexity_penalty', 0.5))
        
        # Combine factors with weights
        confidence = (
            size_confidence * 0.4 +
            variance_factor * 0.4 +
            complexity_factor * 0.2
        )
        
        return np.clip(confidence, 0, 1)
    
    def _update_history(self, metric: EvaluationMetric):
        """
        Update evaluation history with new metric.
        
        Args:
            metric: New evaluation metric to store
        """
        self.evaluation_history.append(metric)
        
        # Maintain history size limit
        if len(self.evaluation_history) > self.config.get('max_history_size', 1000):
            self.evaluation_history.pop(0)
    
    def get_temporal_trend(self,
                          window_size: Optional[int] = None) -> Dict[str, float]:
        """
        Analyze temporal trends in evaluation metrics.
        
        Args:
            window_size: Optional size of analysis window
            
        Returns:
            Dict containing trend analysis metrics
        """
        if not self.evaluation_history:
            return {'trend': 0.0, 'stability': 1.0}
            
        # Use default or specified window size
        window = window_size or self.config.get('default_window_size', 10)
        recent_metrics = self.evaluation_history[-window:]
        
        # Extract metric values
        values = [metric.value for metric in recent_metrics]
        if len(values) < 2:
            return {'trend': 0.0, 'stability': 1.0}
            
        # Compute trend direction and magnitude
        trend = np.polyfit(range(len(values)), values, 1)[0]
        
        # Compute stability (inverse of variance)
        stability = 1 / (1 + np.var(values))
        
        return {
            'trend': float(trend),
            'stability': float(stability),
            'window_size': len(values)
        }

class EvaluationContext:
    """
    Manages contextual information for ethical evaluation.
    
    Provides structured access to historical data, system state,
    and environmental factors affecting evaluation accuracy.
    """
    
    def __init__(self):
        self.historical_data: List[Dict] = []
        self.environmental_factors: Dict = {}
        self.confidence_modifiers: Dict[str, float] = {}
    
    def add_historical_data(self, data: Dict):
        """Add historical data point to context."""
        self.historical_data.append({
            'timestamp': datetime.now(),
            'data': data
        })
    
    def update_environmental_factors(self, factors: Dict):
        """Update environmental factors affecting evaluation."""
        self.environmental_factors.update(factors)
    
    def set_confidence_modifier(self, factor: str, value: float):
        """Set confidence modification factor."""
        self.confidence_modifiers[factor] = value
    
    def get_confidence_multiplier(self) -> float:
        """Compute overall confidence multiplier from modifiers."""
        if not self.confidence_modifiers:
            return 1.0
        return np.prod(list(self.confidence_modifiers.values()))
