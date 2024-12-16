"""
Statistical utilities for ethical analysis
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Tuple, Optional

def detect_outliers(
    data: np.ndarray,
    method: str = 'zscore',
    threshold: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in data using various methods.
    
    Args:
        data: Input data array
        method: Detection method ('zscore', 'iqr', or 'isolation_forest')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of (outlier_mask, scores)
    """
    if method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        return outliers, z_scores
        
    elif method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        outliers = (data < lower) | (data > upper)
        scores = np.abs(data - np.median(data)) / IQR
        return outliers, scores
        
    elif method == 'isolation_forest':
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        scores = iso_forest.fit_predict(data.reshape(-1, 1))
        outliers = scores == -1
        return outliers, scores
        
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def compute_distribution_stats(
    data: np.ndarray
) -> Dict[str, float]:
    """Compute comprehensive distribution statistics."""
    stats_dict = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'median': float(np.median(data)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'iqr': float(np.percentile(data, 75) - np.percentile(data, 25))
    }
    return stats_dict

def check_distribution_shift(
    reference: np.ndarray,
    current: np.ndarray,
    test_method: str = 'ks'
) -> Tuple[float, float]:
    """
    Check for distribution shift between reference and current data.
    
    Args:
        reference: Reference distribution
        current: Current distribution
        test_method: Statistical test method ('ks' or 'anderson')
        
    Returns:
        Tuple of (test_statistic, p_value)
    """
    if test_method == 'ks':
        statistic, pvalue = stats.ks_2samp(reference, current)
        return float(statistic), float(pvalue)
        
    elif test_method == 'anderson':
        statistic, _, pvalue = stats.anderson_ksamp([reference, current])
        return float(statistic), float(pvalue)
        
    else:
        raise ValueError(f"Unknown test method: {test_method}")

def compute_correlation_matrix(
    data: np.ndarray,
    method: str = 'pearson'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation matrix with significance levels.
    
    Args:
        data: Input data array (features as columns)
        method: Correlation method ('pearson' or 'spearman')
        
    Returns:
        Tuple of (correlation_matrix, p_values)
    """
    n_features = data.shape[1]
    corr_matrix = np.zeros((n_features, n_features))
    p_matrix = np.zeros((n_features, n_features))
    
    for i in range(n_features):
        for j in range(i, n_features):
            if method == 'pearson':
                corr, p_value = stats.pearsonr(data[:, i], data[:, j])
            else:
                corr, p_value = stats.spearmanr(data[:, i], data[:, j])
                
            corr_matrix[i, j] = corr_matrix[j, i] = corr
            p_matrix[i, j] = p_matrix[j, i] = p_value
            
    return corr_matrix, p_matrix

def compute_entropy(data: np.ndarray) -> float:
    """Compute Shannon entropy of the data."""
    hist, _ = np.histogram(data, bins='auto', density=True)
    hist = hist[hist > 0]  # Remove zero probabilities
    return float(-np.sum(hist * np.log2(hist)))

def estimate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Estimate confidence interval for the mean.
    
    Args:
        data: Input data array
        confidence: Confidence level (0 to 1)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    mean = np.mean(data)
    sem = stats.sem(data)
    interval = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)
    return float(interval[0]), float(interval[1])