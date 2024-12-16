"""
Bias Detection Module for Ethical AI Systems
Implements comprehensive bias detection across different dimensions.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from ..base import BaseAnalyzer, AnalysisResult, AnalysisContext
from datetime import datetime

class BiasDetector(BaseAnalyzer):
    """Detects various types of bias in ML systems and data."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.bias_types = {
            'demographic': self._check_demographic_bias,
            'representation': self._check_representation_bias,
            'temporal': self._check_temporal_bias,
            'measurement': self._check_measurement_bias,
            'algorithm': self._check_algorithm_bias
        }

    def _default_config(self) -> Dict[str, Any]:
        return {
            'min_samples': 1000,
            'confidence_threshold': 0.85,
            'variance_weight': 0.3,
            'ideal_sample_size': 10000,
            'demographic_threshold': 0.05,
            'representation_threshold': 0.1,
            'temporal_window': 30,  # days
            'measurement_precision': 0.01
        }

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Perform comprehensive bias analysis on input data.
        
        Args:
            data: Dictionary containing:
                - features: Dataset features
                - labels: Target variables
                - metadata: Additional context
                
        Returns:
            AnalysisResult with bias metrics and confidence scores
        """
        results = {}
        confidences = []
        
        # Run all bias checks
        for bias_type, check_func in self.bias_types.items():
            bias_score, confidence = await check_func(data)
            results[bias_type] = bias_score
            confidences.append(confidence)
        
        # Aggregate results
        overall_bias = np.mean(list(results.values()))
        overall_confidence = np.mean(confidences)
        
        context = AnalysisContext(
            timestamp=datetime.now(),
            parameters=self.config,
            metadata={'bias_breakdown': results}
        )
        
        return AnalysisResult(
            value=float(overall_bias),
            confidence=float(overall_confidence),
            context=context,
            details=results
        )

    async def _check_demographic_bias(self, data: Dict[str, Any]) -> tuple[float, float]:
        """Check for demographic disparities in outcomes."""
        if 'protected_attributes' not in data:
            return 0.0, 0.0
            
        disparities = []
        for attr in data['protected_attributes']:
            group_outcomes = self._compute_group_outcomes(data, attr)
            disparity = self._calculate_disparity(group_outcomes)
            disparities.append(disparity)
            
        bias_score = np.mean(disparities)
        confidence = self._compute_confidence(
            len(data.get('features', [])),
            np.var(disparities)
        )
        
        return bias_score, confidence

    async def _check_representation_bias(self, data: Dict[str, Any]) -> tuple[float, float]:
        """Analyze dataset representation across different groups."""
        if 'features' not in data:
            return 0.0, 0.0
            
        distributions = self._compute_distributions(data['features'])
        bias_score = self._calculate_distribution_bias(distributions)
        confidence = self._compute_confidence(
            len(data['features']),
            np.var(list(distributions.values()))
        )
        
        return bias_score, confidence

    async def _check_temporal_bias(self, data: Dict[str, Any]) -> tuple[float, float]:
        """Detect bias patterns over time."""
        if 'timestamps' not in data:
            return 0.0, 0.0
            
        temporal_patterns = self._analyze_temporal_patterns(
            data['timestamps'],
            data.get('outcomes', [])
        )
        bias_score = self._calculate_temporal_bias(temporal_patterns)
        confidence = self._compute_confidence(
            len(data.get('timestamps', [])),
            np.var(temporal_patterns)
        )
        
        return bias_score, confidence

    async def _check_measurement_bias(self, data: Dict[str, Any]) -> tuple[float, float]:
        """Identify bias in measurement or data collection."""
        if 'features' not in data:
            return 0.0, 0.0
            
        measurement_errors = self._analyze_measurement_errors(data['features'])
        bias_score = np.mean(measurement_errors)
        confidence = self._compute_confidence(
            len(data['features']),
            np.var(measurement_errors)
        )
        
        return bias_score, confidence

    async def _check_algorithm_bias(self, data: Dict[str, Any]) -> tuple[float, float]:
        """Evaluate algorithmic decision-making bias."""
        if 'predictions' not in data or 'ground_truth' not in data:
            return 0.0, 0.0
            
        bias_patterns = self._analyze_algorithm_patterns(
            data['predictions'],
            data['ground_truth']
        )
        bias_score = np.mean(bias_patterns)
        confidence = self._compute_confidence(
            len(data.get('predictions', [])),
            np.var(bias_patterns)
        )
        
        return bias_score, confidence

    def _compute_group_outcomes(self, data: Dict[str, Any], attribute: str) -> Dict[str, float]:
        """Calculate outcomes for different demographic groups."""
        groups = {}
        for idx, attr_val in enumerate(data[attribute]):
            if attr_val not in groups:
                groups[attr_val] = []
            groups[attr_val].append(data['outcomes'][idx])
        return {k: np.mean(v) for k, v in groups.items()}

    def _calculate_disparity(self, group_outcomes: Dict[str, float]) -> float:
        """Calculate disparity between group outcomes."""
        if not group_outcomes:
            return 0.0
        values = list(group_outcomes.values())
        return max(values) - min(values)

    def _compute_distributions(self, features: List[Any]) -> Dict[str, float]:
        """Compute distribution of features across dataset."""
        if not features:
            return {}
        unique, counts = np.unique(features, return_counts=True)
        total = len(features)
        return {str(u): c/total for u, c in zip(unique, counts)}

    def _calculate_distribution_bias(self, distributions: Dict[str, float]) -> float:
        """Calculate bias in feature distributions."""
        if not distributions:
            return 0.0
        expected = 1.0 / len(distributions)
        return np.mean([abs(v - expected) for v in distributions.values()])

    def _analyze_temporal_patterns(self, timestamps: List[datetime], 
                                 outcomes: List[float]) -> List[float]:
        """Analyze patterns in outcomes over time."""
        if not timestamps or not outcomes:
            return [0.0]
        # Sort by timestamp
        sorted_pairs = sorted(zip(timestamps, outcomes))
        return [o for _, o in sorted_pairs]

    def _calculate_temporal_bias(self, patterns: List[float]) -> float:
        """Calculate bias in temporal patterns."""
        if len(patterns) < 2:
            return 0.0
        differences = np.diff(patterns)
        return np.mean(np.abs(differences))

    def _analyze_measurement_errors(self, features: List[Any]) -> List[float]:
        """Analyze potential measurement errors in features."""
        if not features:
            return [0.0]
        # Look for anomalies and inconsistencies
        z_scores = np.abs(stats.zscore(features))
        return z_scores.tolist()

    def _analyze_algorithm_patterns(self, predictions: List[Any], 
                                  ground_truth: List[Any]) -> List[float]:
        """Analyze patterns in algorithmic decisions."""
        if not predictions or not ground_truth:
            return [0.0]
        # Compare predictions to ground truth
        errors = [1.0 if p != g else 0.0 for p, g in zip(predictions, ground_truth)]
        return errors
