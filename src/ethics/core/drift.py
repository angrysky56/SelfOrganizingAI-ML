"""
Drift Module for detecting changes in model behavior and data distributions
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from .utils.statistical_utils import (
    check_distribution_shift,
    compute_distribution_stats,
    estimate_confidence_interval
)

@dataclass
class DriftResult:
    """Results from drift detection."""
    drift_detected: bool
    drift_score: float
    confidence: float
    metric_changes: Dict[str, float]
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None

class DriftAnalyzer:
    """Analyzes data and model behavior for drift."""
    
    def __init__(self, 
                 window_size: int = 1000,
                 threshold: float = 0.05,
                 metrics: List[str] = None):
        """
        Initialize drift analyzer.
        
        Args:
            window_size: Size of reference window
            threshold: P-value threshold for drift detection
            metrics: List of metrics to monitor for drift
        """
        self.window_size = window_size
        self.threshold = threshold
        self.metrics = metrics or ['mean', 'std', 'median', 'skewness']
        
        self.reference_data = None
        self.reference_stats = None
        self.historical_drifts = []
        
    def update_reference(self, data: np.ndarray):
        """Update reference distribution."""
        self.reference_data = data[-self.window_size:]
        self.reference_stats = compute_distribution_stats(self.reference_data)
        
    async def check_drift(self, 
                         current_data: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> DriftResult:
        """
        Check for drift in current data compared to reference.
        
        Args:
            current_data: New data to check for drift
            metadata: Optional metadata about the data
            
        Returns:
            DriftResult containing drift analysis
        """
        if self.reference_data is None:
            self.update_reference(current_data)
            return DriftResult(
                drift_detected=False,
                drift_score=0.0,
                confidence=1.0,
                metric_changes={},
                timestamp=datetime.now()
            )
            
        # Check distribution shift
        statistic, p_value = check_distribution_shift(
            self.reference_data,
            current_data,
            test_method='ks'
        )
        
        # Compute current statistics
        current_stats = compute_distribution_stats(current_data)
        
        # Calculate changes in metrics
        metric_changes = {
            metric: abs(current_stats[metric] - self.reference_stats[metric])
            for metric in self.metrics
        }
        
        # Calculate drift score
        drift_score = 1.0 - p_value
        
        # Determine if drift is significant
        drift_detected = p_value < self.threshold
        
        # Calculate confidence
        _, confidence = estimate_confidence_interval(current_data)
        confidence = min(1.0, confidence)
        
        result = DriftResult(
            drift_detected=drift_detected,
            drift_score=drift_score,
            confidence=confidence,
            metric_changes=metric_changes,
            timestamp=datetime.now(),
            details={
                'statistic': statistic,
                'p_value': p_value,
                'reference_stats': self.reference_stats,
                'current_stats': current_stats,
                'metadata': metadata
            }
        )
        
        # Store result
        self.historical_drifts.append(result)
        
        # Update reference if no drift
        if not drift_detected:
            self.update_reference(current_data)
            
        return result
    
    def get_drift_history(self, 
                         window: Optional[int] = None) -> List[DriftResult]:
        """Get historical drift results."""
        if window is None:
            return self.historical_drifts
        return self.historical_drifts[-window:]
    
    def get_stability_score(self) -> float:
        """Calculate stability score based on drift history."""
        if not self.historical_drifts:
            return 1.0
            
        recent_drifts = self.get_drift_history(window=10)
        drift_scores = [d.drift_score for d in recent_drifts]
        
        stability = 1.0 - np.mean(drift_scores)
        return float(stability)
    
    async def analyze_feature_drift(self,
                                  reference_features: Dict[str, np.ndarray],
                                  current_features: Dict[str, np.ndarray]
                                  ) -> Dict[str, DriftResult]:
        """
        Analyze drift for each feature separately.
        
        Args:
            reference_features: Dictionary of reference feature arrays
            current_features: Dictionary of current feature arrays
            
        Returns:
            Dictionary of feature names to drift results
        """
        feature_drifts = {}
        
        for feature_name in reference_features:
            if feature_name not in current_features:
                continue
                
            result = await self.check_drift(
                reference_features[feature_name],
                current_features[feature_name]
            )
            
            feature_drifts[feature_name] = result
            
        return feature_drifts