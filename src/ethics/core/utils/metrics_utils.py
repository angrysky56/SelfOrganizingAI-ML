"""
Utility functions for ethical metrics calculations
"""

import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MetricResult:
    value: float
    confidence: float
    details: Optional[Dict[str, Any]] = None

def calculate_distribution_difference(
    distribution1: np.ndarray,
    distribution2: np.ndarray
) -> float:
    """Calculate statistical difference between two distributions."""
    if len(distribution1) == 0 or len(distribution2) == 0:
        return 1.0
        
    # Normalize distributions
    dist1 = distribution1 / np.sum(distribution1)
    dist2 = distribution2 / np.sum(distribution2)
    
    # Calculate Jensen-Shannon divergence
    m = (dist1 + dist2) / 2
    divergence = (
        entropy(dist1, m) / 2 + 
        entropy(dist2, m) / 2
    )
    
    return float(divergence)

def entropy(distribution: np.ndarray, base: Optional[np.ndarray] = None) -> float:
    """Calculate entropy of a distribution."""
    if base is None:
        return -np.sum(distribution * np.log2(distribution + 1e-10))
    return -np.sum(distribution * np.log2(distribution / (base + 1e-10) + 1e-10))

def calculate_fairness_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    protected_attributes: Dict[str, np.ndarray]
) -> Dict[str, MetricResult]:
    """Calculate comprehensive fairness metrics."""
    metrics = {}
    
    # Demographic parity
    metrics['demographic_parity'] = calculate_demographic_parity(
        predictions, protected_attributes
    )
    
    # Equal opportunity
    metrics['equal_opportunity'] = calculate_equal_opportunity(
        predictions, ground_truth, protected_attributes
    )
    
    # Equalized odds
    metrics['equalized_odds'] = calculate_equalized_odds(
        predictions, ground_truth, protected_attributes
    )
    
    return metrics

def calculate_demographic_parity(
    predictions: np.ndarray,
    protected_attributes: Dict[str, np.ndarray]
) -> MetricResult:
    """Calculate demographic parity difference."""
    parities = []
    
    for attr_name, attr_values in protected_attributes.items():
        unique_values = np.unique(attr_values)
        acceptance_rates = []
        
        for value in unique_values:
            mask = attr_values == value
            rate = np.mean(predictions[mask])
            acceptance_rates.append(rate)
        
        # Calculate max difference in acceptance rates
        parity_diff = max(acceptance_rates) - min(acceptance_rates)
        parities.append(parity_diff)
    
    avg_parity = np.mean(parities)
    confidence = 1.0 - np.std(parities)
    
    return MetricResult(
        value=1.0 - avg_parity,  # Convert to fairness score
        confidence=confidence,
        details={'per_attribute_parity': dict(zip(protected_attributes.keys(), parities))}
    )

def calculate_equal_opportunity(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    protected_attributes: Dict[str, np.ndarray]
) -> MetricResult:
    """Calculate equal opportunity difference."""
    opportunities = []
    
    for attr_name, attr_values in protected_attributes.items():
        unique_values = np.unique(attr_values)
        true_positive_rates = []
        
        for value in unique_values:
            mask = attr_values == value
            positives = ground_truth[mask] == 1
            if np.sum(positives) > 0:
                true_pos = np.logical_and(predictions[mask] == 1, ground_truth[mask] == 1)
                rate = np.sum(true_pos) / np.sum(positives)
                true_positive_rates.append(rate)
        
        if true_positive_rates:
            opp_diff = max(true_positive_rates) - min(true_positive_rates)
            opportunities.append(opp_diff)
    
    avg_opp = np.mean(opportunities) if opportunities else 1.0
    confidence = 1.0 - np.std(opportunities) if opportunities else 0.0
    
    return MetricResult(
        value=1.0 - avg_opp,  # Convert to fairness score
        confidence=confidence,
        details={'per_attribute_opportunity': dict(zip(protected_attributes.keys(), opportunities))}
    )

def calculate_equalized_odds(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    protected_attributes: Dict[str, np.ndarray]
) -> MetricResult:
    """Calculate equalized odds difference."""
    odds_differences = []
    
    for attr_name, attr_values in protected_attributes.items():
        unique_values = np.unique(attr_values)
        tpr_differences = []
        fpr_differences = []
        
        rates = []
        for value in unique_values:
            mask = attr_values == value
            pos_mask = ground_truth[mask] == 1
            neg_mask = ground_truth[mask] == 0
            
            if np.sum(pos_mask) > 0:
                tpr = np.mean(predictions[mask][pos_mask])
                rates.append(('tpr', value, tpr))
            
            if np.sum(neg_mask) > 0:
                fpr = np.mean(predictions[mask][neg_mask])
                rates.append(('fpr', value, fpr))
        
        # Calculate differences
        for rate_type in ['tpr', 'fpr']:
            type_rates = [r[2] for r in rates if r[0] == rate_type]
            if type_rates:
                diff = max(type_rates) - min(type_rates)
                if rate_type == 'tpr':
                    tpr_differences.append(diff)
                else:
                    fpr_differences.append(diff)
        
        odds_diff = np.mean(tpr_differences + fpr_differences)
        odds_differences.append(odds_diff)
    
    avg_odds = np.mean(odds_differences) if odds_differences else 1.0
    confidence = 1.0 - np.std(odds_differences) if odds_differences else 0.0
    
    return MetricResult(
        value=1.0 - avg_odds,  # Convert to fairness score
        confidence=confidence,
        details={'per_attribute_odds': dict(zip(protected_attributes.keys(), odds_differences))}
    )

def check_bias_in_embeddings(
    embeddings: torch.Tensor,
    attributes: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Check for bias in learned embeddings."""
    bias_scores = {}
    
    for attr_name, attr_values in attributes.items():
        # Calculate centroid for each attribute value
        unique_values = torch.unique(attr_values)
        centroids = {}
        
        for value in unique_values:
            mask = attr_values == value
            centroids[value.item()] = embeddings[mask].mean(dim=0)
        
        # Calculate bias as maximum distance between centroids
        distances = []
        values = list(centroids.keys())
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                dist = torch.norm(centroids[values[i]] - centroids[values[j]])
                distances.append(dist.item())
        
        bias_scores[attr_name] = max(distances) if distances else 0.0
    
    return bias_scores