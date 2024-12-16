"""
Fairness Metrics Module for ML Models
Implements various fairness metrics and evaluation criteria.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from scipy import stats
from ..base import BaseAnalyzer, AnalysisResult, AnalysisContext
from datetime import datetime

class FairnessMetrics(BaseAnalyzer):
    """Evaluates various aspects of ML model fairness."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.metrics = {
            'demographic_parity': self._compute_demographic_parity,
            'equal_opportunity': self._compute_equal_opportunity,
            'equalized_odds': self._compute_equalized_odds,
            'disparate_impact': self._compute_disparate_impact,
            'individual_fairness': self._compute_individual_fairness
        }

    def _default_config(self) -> Dict[str, Any]:
        return {
            'min_samples': 1000,
            'confidence_threshold': 0.9,
            'variance_weight': 0.2,
            'ideal_sample_size': 10000,
            'fairness_threshold': 0.8,
            'similarity_threshold': 0.95,
            'protected_attributes': ['gender', 'race', 'age'],
            'metric_weights': {
                'demographic_parity': 0.25,
                'equal_opportunity': 0.25,
                'equalized_odds': 0.2,
                'disparate_impact': 0.15,
                'individual_fairness': 0.15
            }
        }

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """
        Compute comprehensive fairness metrics for ML model.
        
        Args:
            data: Dictionary containing:
                - predictions: Model predictions
                - ground_truth: Actual outcomes
                - protected_attributes: Protected characteristic values
                - features: Input features
                
        Returns:
            AnalysisResult with fairness metrics and confidence scores
        """
        results = {}
        confidences = []
        
        for metric_name, metric_func in self.metrics.items():
            metric_value, confidence = await metric_func(data)
            results[metric_name] = metric_value
            confidences.append(confidence)
        
        weights = self.config['metric_weights']
        overall_fairness = sum(weights[m] * results[m] for m in results.keys())
        overall_confidence = np.mean(confidences)
        
        context = AnalysisContext(
            timestamp=datetime.now(),
            parameters=self.config,
            metadata={'metric_breakdown': results}
        )
        
        return AnalysisResult(
            value=float(overall_fairness),
            confidence=float(overall_confidence),
            context=context,
            details=results
        )

    async def _compute_disparate_impact(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute disparate impact ratio.
        Measures the ratio of favorable outcomes between protected groups.
        """
        if not self._validate_inputs(data, ['predictions', 'protected_attributes']):
            return 0.0, 0.0
            
        impact_ratios = []
        
        for attr in self.config['protected_attributes']:
            if attr not in data['protected_attributes']:
                continue
                
            groups = self._group_predictions(
                data['predictions'],
                data['protected_attributes'][attr]
            )
            
            # Calculate acceptance rates for each group
            acceptance_rates = {}
            for group, indices in groups.items():
                acceptance_rates[group] = np.mean(data['predictions'][indices])
            
            # Calculate ratio of min to max rate
            min_rate = min(acceptance_rates.values())
            max_rate = max(acceptance_rates.values())
            
            if max_rate > 0:
                impact_ratio = min_rate / max_rate
                impact_ratios.append(impact_ratio)
        
        if not impact_ratios:
            return 0.0, 0.0
            
        fairness_score = np.mean(impact_ratios)
        confidence = self._compute_confidence(
            len(data['predictions']),
            np.var(impact_ratios)
        )
        
        return fairness_score, confidence

    async def _compute_individual_fairness(self, data: Dict[str, Any]) -> Tuple[float, float]:
        """
        Compute individual fairness metric.
        Measures whether similar individuals receive similar predictions.
        """
        if not self._validate_inputs(data, ['predictions', 'features']):
            return 0.0, 0.0
            
        # Sample pairs for efficiency if dataset is large
        n_samples = min(1000, len(data['predictions']))
        indices = np.random.choice(len(data['predictions']), n_samples, replace=False)
        
        fairness_scores = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                # Compute similarity between instances
                similarity = self._compute_similarity(
                    data['features'][indices[i]],
                    data['features'][indices[j]]
                )
                
                # Compute prediction difference
                pred_diff = abs(
                    data['predictions'][indices[i]] - 
                    data['predictions'][indices[j]]
                )
                
                # Similar individuals should have similar predictions
                if similarity > self.config['similarity_threshold']:
                    fairness_scores.append(1.0 - pred_diff)
        
        if not fairness_scores:
            return 0.0, 0.0
            
        fairness_score = np.mean(fairness_scores)
        confidence = self._compute_confidence(
            len(fairness_scores),
            np.var(fairness_scores)
        )
        
        return fairness_score, confidence

    def _compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute similarity between two feature vectors."""
        if len(features1) != len(features2):
            return 0.0
        
        # Normalize features
        f1_norm = features1 / np.linalg.norm(features1)
        f2_norm = features2 / np.linalg.norm(features2)
        
        # Compute cosine similarity
        similarity = np.dot(f1_norm, f2_norm)
        return float(similarity)

    def _validate_inputs(self, data: Dict[str, Any], required_keys: List[str]) -> bool:
        """Validate that required data is present."""
        return all(key in data for key in required_keys)

    def _group_predictions(self, predictions: np.ndarray, 
                         protected_attr: np.ndarray,
                         ground_truth: Optional[np.ndarray] = None) -> Dict[Any, np.ndarray]:
        """Group predictions by protected attribute values."""
        unique_values = np.unique(protected_attr)
        groups = {}
        
        for value in unique_values:
            groups[value] = np.where(protected_attr == value)[0]
            
        return groups

    def _compute_rates(self, predictions: np.ndarray, 
                      ground_truth: np.ndarray) -> Tuple[float, float]:
        """Compute true positive and false positive rates."""
        positives = ground_truth == 1
        negatives = ground_truth == 0
        
        tpr = np.sum(predictions[positives]) / np.sum(positives) if np.sum(positives) > 0 else 0
        fpr = np.sum(predictions[negatives]) / np.sum(negatives) if np.sum(negatives) > 0 else 0
        
        return tpr, fpr
