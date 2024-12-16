# Base Module: Core Analysis Framework for Ethical Reasoning
# Implements foundational abstractions for ethical analysis components.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Any, List
import numpy as np

@dataclass
class AnalysisContext:
    """
    Contextual information for ethical analysis.

    Attributes:
        timestamp: Time of analysis
        parameters: Analysis configuration parameters
        metadata: Additional contextual information
    """
    timestamp: datetime
    parameters: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AnalysisResult:
    """
    Structured container for analysis outcomes.

    Attributes:
        value: Primary metric value [0.0 - 1.0]
        confidence: Confidence score for the analysis
        context: Analysis context information
        details: Detailed analysis components
    """
    value: float
    confidence: float
    context: AnalysisContext
    details: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """Validate result constraints."""
        return (0.0 <= self.value <= 1.0 and
                0.0 <= self.confidence <= 1.0)

class BaseAnalyzer(ABC):
    """
    Abstract base class defining core analysis interface.

    Provides foundational capabilities for ethical analysis components
    while enforcing consistent interfaces and validation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize analyzer with configuration.

        Args:
            config: Optional configuration override
        """
        self.config = config or self._default_config()
        self._validate_config()

    @abstractmethod
    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Execute core analysis logic.

        Args:
            data: Input data for analysis

        Returns:
            AnalysisResult containing analysis outcome

        Raises:
            ValueError: If input data is invalid
        """
        pass

    @abstractmethod
    def _default_config(self) -> Dict[str, Any]:
        """
        Define default component configuration.

        Returns:
            Dict containing default parameters
        """
        pass

    def _validate_config(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = {'min_samples', 'confidence_threshold'}
        if not required_keys.issubset(self.config.keys()):
            raise ValueError(f"Missing required config keys: {required_keys}")

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
        if sample_size < self.config['min_samples']:
            return 0.0

        # Base confidence from sample size
        base_confidence = np.clip(
            sample_size / self.config['ideal_sample_size'],
            0.0,
            1.0
        )

        # Variance penalty
        variance_factor = np.exp(-variance * self.config['variance_weight'])

        # Combined confidence
        confidence = base_confidence * variance_factor

        return float(np.clip(confidence, 0.0, 1.0))
