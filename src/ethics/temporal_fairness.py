"""
Temporal Fairness Analysis Module

Implements advanced temporal analysis for fairness metrics in self-organizing AI systems.
Tracks evolution of fairness measures over time and detects concerning patterns.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import logging
from .evaluators import BaseEvaluator, EvaluationMetric, EvaluationDomain

@dataclass
class TemporalMetric:
    """
    Temporal fairness measurement structure.
    
    Attributes:
        metric_type: Type of fairness measure
        current_value: Most recent metric value
        trend: Detected trend direction and magnitude
        stability: Measure of metric stability over time
        forecast: Predicted future values
        anomalies: Detected anomalous patterns
    """
    metric_type: str
    current_value: float
    trend: float
    stability: float
    forecast: List[float]
    anomalies: List[Dict]

class TemporalFairnessAnalyzer(BaseEvaluator):
    """
    Advanced temporal analysis system for fairness metrics.
    
    Key Capabilities:
        - Time series analysis of fairness metrics
        - Trend detection and forecasting
        - Anomaly detection in fairness patterns
        - Stability assessment over time windows
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
    
    def _default_config(self) -> Dict:
        return {
            'window_sizes': {
                'short_term': 10,
                'medium_term': 50,
                'long_term': 200
            },
            'trend_thresholds': {
                'significant_change': 0.1,
                'critical_change': 0.2
            },
            'stability_thresholds': {
                'unstable': 0.3,
                'highly_unstable': 0.5
            },
            'forecast_params': {
                'horizon': 10,
                'confidence_interval': 0.95
            },
            'anomaly_detection': {
                'z_score_threshold': 3.0,
                'min_sequence_length': 5
            }
        }
    
    async def evaluate(self,
                      system_state: Dict,
                      context: Optional[Dict] = None) -> EvaluationMetric:
        """
        Execute temporal fairness analysis.
        
        Args:
            system_state: Current system state including fairness metrics
            context: Optional additional context
            
        Returns:
            EvaluationMetric for temporal fairness assessment
        """
        try:
            fairness_metrics = self._extract_fairness_metrics(system_state)
            temporal_analysis = self._analyze_temporal_patterns(
                fairness_metrics,
                context
            )
            
            evaluation = EvaluationMetric(
                domain=EvaluationDomain.FAIRNESS,
                value=self._compute_temporal_score(temporal_analysis),
                confidence=self._compute_temporal_confidence(temporal_analysis),
                timestamp=datetime.now(),
                metadata=temporal_analysis
            )
            
            self._update_history(evaluation)
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Temporal fairness analysis failed: {str(e)}")
            raise
    
    def _extract_fairness_metrics(self, system_state: Dict) -> Dict[str, List[float]]:
        """
        Extract fairness metrics from system state.
        
        Args:
            system_state: Current system state
            
        Returns:
            Dict mapping metric names to historical values
        """
        metrics = {}
        for key, value in system_state.items():
            if key.startswith('fairness_'):
                metric_name = key.replace('fairness_', '')
                if isinstance(value, (list, np.ndarray)):
                    metrics[metric_name] = list(value)
                else:
                    metrics[metric_name] = [float(value)]
        return metrics

    def _forecast_metric(self, 
                        values: List[float], 
                        horizon: int) -> List[float]:
        """
        Generate forecasts for fairness metrics.
        
        Args:
            values: Historical metric values
            horizon: Number of steps to forecast
            
        Returns:
            List[float]: Forecasted values
        """
        if len(values) < 3:
            return [values[-1]] * horizon if values else [0.0] * horizon
            
        # Implement exponential smoothing
        alpha = 0.3  # Smoothing factor
        smoothed = []
        s = values[0]
        
        for value in values[1:]:
            s = alpha * value + (1 - alpha) * s
            smoothed.append(s)
        
        # Project trend
        trend = (smoothed[-1] - smoothed[0]) / len(smoothed)
        forecast = []
        last_value = smoothed[-1]
        
        for i in range(horizon):
            next_value = last_value + trend
            # Ensure forecast stays in valid range
            next_value = np.clip(next_value, 0, 1)
            forecast.append(float(next_value))
            last_value = next_value
            
        return forecast

    def _detect_anomalies(self, 
                         values: List[float],
                         context: Optional[Dict]) -> List[Dict]:
        """
        Detect anomalous patterns in fairness metrics.
        
        Args:
            values: Historical metric values
            context: Optional detection context
            
        Returns:
            List[Dict]: Detected anomalies with metadata
        """
        if len(values) < self.config['anomaly_detection']['min_sequence_length']:
            return []
            
        anomalies = []
        values_array = np.array(values)
        
        # Z-score based detection
        z_scores = np.abs(stats.zscore(values_array))
        threshold = self.config['anomaly_detection']['z_score_threshold']
        
        for idx, z_score in enumerate(z_scores):
            if z_score > threshold:
                anomaly = {
                    'index': idx,
                    'value': float(values[idx]),
                    'z_score': float(z_score),
                    'type': 'point_anomaly',
                    'severity': float(z_score / threshold),
                    'timestamp': datetime.now() - timedelta(periods=len(values)-idx-1)
                }
                anomalies.append(anomaly)
        
        # Detect trend anomalies
        trend_changes = np.diff(values_array)
        rapid_changes = np.where(np.abs(trend_changes) > 
                               self.config['trend_thresholds']['critical_change'])[0]
        
        for idx in rapid_changes:
            anomaly = {
                'index': idx,
                'value': float(values[idx]),
                'change_magnitude': float(trend_changes[idx]),
                'type': 'trend_anomaly',
                'severity': float(abs(trend_changes[idx]) / 
                                self.config['trend_thresholds']['critical_change']),
                'timestamp': datetime.now() - timedelta(periods=len(values)-idx-1)
            }
            anomalies.append(anomaly)
        
        return anomalies

    def _compute_temporal_score(self, temporal_analysis: Dict[str, TemporalMetric]) -> float:
        """
        Compute overall temporal fairness score.
        
        Args:
            temporal_analysis: Results of temporal analysis
            
        Returns:
            float: Aggregated temporal fairness score [0.0 - 1.0]
        """
        scores = []
        weights = []
        
        for metric in temporal_analysis.values():
            # Base score from current value
            base_score = metric.current_value
            
            # Stability penalty
            stability_penalty = (1 - metric.stability) * 0.3
            
            # Trend penalty
            trend_penalty = abs(metric.trend) * 0.2 if metric.trend < 0 else 0
            
            # Anomaly penalty
            anomaly_penalty = min(len(metric.anomalies) * 0.1, 0.3)
            
            # Compute weighted score
            score = base_score * (1 - stability_penalty - trend_penalty - anomaly_penalty)
            
            scores.append(score)
            weights.append(1.0)  # Equal weights for now, could be customized
            
        if not scores:
            return 0.0
            
        # Weighted average
        return float(np.average(scores, weights=weights))

    def _compute_temporal_confidence(self, 
                                   temporal_analysis: Dict[str, TemporalMetric]) -> float:
        """
        Compute confidence in temporal fairness assessment.
        
        Args:
            temporal_analysis: Results of temporal analysis
            
        Returns:
            float: Confidence score [0.0 - 1.0]
        """
        if not temporal_analysis:
            return 0.0
            
        confidence_factors = []
        
        for metric in temporal_analysis.values():
            # Stability confidence
            stability_confidence = metric.stability
            
            # Trend confidence based on consistency
            trend_confidence = 1.0 - abs(metric.trend)
            
            # Anomaly impact on confidence
            anomaly_confidence = 1.0 - min(len(metric.anomalies) * 0.1, 0.5)
            
            # Forecast confidence based on stability and trend
            forecast_confidence = (stability_confidence + trend_confidence) / 2
            
            confidence_factors.append(
                stability_confidence * 0.4 +
                trend_confidence * 0.3 +
                anomaly_confidence * 0.2 +
                forecast_confidence * 0.1
            )
            
        return float(np.mean(confidence_factors))
