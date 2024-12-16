"""
Drift Detector for monitoring and responding to data and model drift
"""

from typing import Dict, List, Any, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from .drift import DriftAnalyzer, DriftResult
from .utils.statistical_utils import compute_correlation_matrix

@dataclass
class DriftAlert:
    """Alert generated when significant drift is detected."""
    severity: str  # 'low', 'medium', 'high'
    message: str
    drift_result: DriftResult
    recommended_actions: List[str]
    timestamp: datetime

class DriftDetector:
    """Monitors and responds to drift in data and model behavior."""
    
    def __init__(self,
                 alert_threshold: float = 0.1,
                 correlation_threshold: float = 0.7,
                 callback: Optional[Callable[[DriftAlert], None]] = None):
        """
        Initialize drift detector.
        
        Args:
            alert_threshold: Threshold for generating alerts
            correlation_threshold: Threshold for correlation analysis
            callback: Optional callback function for alerts
        """
        self.alert_threshold = alert_threshold
        self.correlation_threshold = correlation_threshold
        self.callback = callback
        
        self.analyzers = {}  # Multiple analyzers for different aspects
        self.alerts = []
        self.active_mitigations = set()
        
    def add_analyzer(self, name: str, analyzer: DriftAnalyzer):
        """Add a drift analyzer for monitoring."""
        self.analyzers[name] = analyzer
        
    async def check_drift(self,
                         name: str,
                         current_data: np.ndarray,
                         metadata: Optional[Dict[str, Any]] = None) -> DriftResult:
        """
        Check for drift using named analyzer.
        
        Args:
            name: Name of analyzer to use
            current_data: Data to check for drift
            metadata: Optional metadata
            
        Returns:
            DriftResult from analysis
        """
        if name not in self.analyzers:
            raise ValueError(f"No analyzer named {name}")
            
        result = await self.analyzers[name].check_drift(current_data, metadata)
        
        # Generate alert if needed
        if result.drift_detected and result.drift_score > self.alert_threshold:
            alert = self._generate_alert(name, result)
            self.alerts.append(alert)
            
            if self.callback:
                self.callback(alert)
                
        return result
    
    def _generate_alert(self, analyzer_name: str, result: DriftResult) -> DriftAlert:
        """Generate drift alert with severity and recommendations."""
        # Determine severity
        if result.drift_score > 0.5:
            severity = 'high'
        elif result.drift_score > 0.25:
            severity = 'medium'
        else:
            severity = 'low'
            
        # Generate message
        message = (
            f"Significant drift detected by {analyzer_name} analyzer. "
            f"Drift score: {result.drift_score:.3f}, "
            f"Confidence: {result.confidence:.3f}"
        )
        
        # Generate recommendations
        recommendations = self._get_recommendations(
            analyzer_name,
            result,
            severity
        )
        
        return DriftAlert(
            severity=severity,
            message=message,
            drift_result=result,
            recommended_actions=recommendations,
            timestamp=datetime.now()
        )
    
    def _get_recommendations(self,
                           analyzer_name: str,
                           result: DriftResult,
                           severity: str) -> List[str]:
        """Get recommended actions based on drift analysis."""
        recommendations = []
        
        if severity == 'high':
            recommendations.extend([
                "Immediately investigate root cause of drift",
                "Consider retraining model with recent data",
                "Review and update monitoring thresholds"
            ])
            
        elif severity == 'medium':
            recommendations.extend([
                "Monitor closely for continued drift",
                "Analyze impacted features or components",
                "Prepare for potential model update"
            ])
            
        else:  # low
            recommendations.extend([
                "Continue monitoring drift patterns",
                "Review feature correlations",
                "Document drift occurrence"
            ])
            
        # Add specific recommendations based on metrics
        for metric, change in result.metric_changes.items():
            if change > self.alert_threshold:
                recommendations.append(
                    f"Investigate change in {metric}: {change:.3f}"
                )
                
        return recommendations
    
    async def analyze_correlations(self,
                                 features: Dict[str, np.ndarray]) -> Dict[str, List[str]]:
        """
        Analyze feature correlations for drift patterns.
        
        Args:
            features: Dictionary of feature arrays
            
        Returns:
            Dictionary mapping features to their correlated features
        """
        # Convert features to matrix
        feature_names = list(features.keys())
        feature_matrix = np.column_stack([
            features[name] for name in feature_names
        ])
        
        # Compute correlation matrix
        corr_matrix, p_values = compute_correlation_matrix(feature_matrix)
        
        # Find significant correlations
        correlations = {}
        for i, feature in enumerate(feature_names):
            correlated = []
            for j, other in enumerate(feature_names):
                if i != j and abs(corr_matrix[i, j]) > self.correlation_threshold:
                    correlated.append(other)
            if correlated:
                correlations[feature] = correlated
                
        return correlations
    
    def get_active_alerts(self, 
                         min_severity: str = 'low') -> List[DriftAlert]:
        """Get currently active alerts above minimum severity."""
        severity_levels = {
            'low': 0,
            'medium': 1,
            'high': 2
        }
        min_level = severity_levels[min_severity]
        
        return [
            alert for alert in self.alerts
            if severity_levels[alert.severity] >= min_level
        ]
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Generate stability report across all analyzers."""
        report = {
            'overall_stability': np.mean([
                analyzer.get_stability_score()
                for analyzer in self.analyzers.values()
            ]),
            'analyzer_scores': {
                name: analyzer.get_stability_score()
                for name, analyzer in self.analyzers.items()
            },
            'active_alerts': len(self.get_active_alerts()),
            'timestamp': datetime.now()
        }
        return report