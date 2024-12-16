"""
Stability Analyzer: Advanced Statistical Process Control Framework

Implements sophisticated stability analysis for ethical metric evolution through:
- Process capability analysis (Cp, Cpk, Pp, Ppk)
- Multi-dimensional variance decomposition
- Non-linear drift detection
- Probabilistic stability assessment

Core Architecture:
1. Statistical Process Control (SPC)
2. Variance Component Analysis (VCA)
3. Time Series Stability Analysis
4. Drift Pattern Recognition
"""

from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
import logging
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

@dataclass
class StabilityMetrics:
    """
    Comprehensive stability measurement structure.
    
    Attributes:
        process_capability: Short and long-term capability indices
        control_limits: Multi-level control boundaries
        variance_components: Decomposed variance sources
        stability_indices: Multi-dimensional stability metrics
        drift_characteristics: Detected drift patterns and properties
    """
    process_capability: Dict[str, float]
    control_limits: Dict[str, float]
    variance_components: Dict[str, float]
    stability_indices: Dict[str, float]
    drift_characteristics: Optional[Dict[str, float]] = None

class StabilityAnalyzer:
    """
    Advanced stability analysis system for ethical metric evolution.
    
    Key Features:
    1. Process Capability Analysis
        - Short-term capability (Cp/Cpk)
        - Long-term capability (Pp/Ppk)
        - Within/between group variation
        
    2. Control Limit Computation
        - Multi-level control boundaries
        - Adaptive threshold refinement
        - Non-parametric limit estimation
        
    3. Variance Decomposition
        - Component-wise analysis
        - Interaction effects
        - Temporal evolution
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize stability analyzer with configuration parameters."""
        self.config = config or self._default_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize statistical parameters
        self._init_statistical_parameters()
        
    def _compute_process_capability(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive process capability indices.
        
        Methodology:
        1. Short-term capability (Cp/Cpk)
            - Within-group variation analysis
            - Natural process limits
            
        2. Long-term capability (Pp/Ppk)
            - Overall variation assessment
            - Performance boundaries
        
        Args:
            time_series: Input measurement time series
            
        Returns:
            Dict containing capability metrics and confidence intervals
        """
        # Extract process parameters
        mean = np.mean(time_series)
        std = np.std(time_series)
        
        # Compute specification limits
        usl = mean + self.config['process_limits']['sigma_multiplier'] * std
        lsl = mean - self.config['process_limits']['sigma_multiplier'] * std
        
        # Short-term capability
        cp = (usl - lsl) / (6 * std)
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)
        
        # Long-term capability
        overall_std = np.std(time_series)
        pp = (usl - lsl) / (6 * overall_std)
        ppu = (usl - mean) / (3 * overall_std)
        ppl = (mean - lsl) / (3 * overall_std)
        ppk = min(ppu, ppl)
        
        return {
            'cp': cp,
            'cpk': cpk,
            'cpu': cpu,
            'cpl': cpl,
            'pp': pp,
            'ppk': ppk,
            'ppu': ppu,
            'ppl': ppl,
            'confidence_intervals': self._compute_capability_confidence(
                time_series, cp, cpk, pp, ppk
            )
        }
        
    def _decompose_variance(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Perform comprehensive variance component analysis.
        
        Components:
        1. Within-group variance
        2. Between-group variance
        3. Temporal variance
        4. Interaction effects
        
        Args:
            time_series: Input time series data
            
        Returns:
            Dict containing variance components and their contributions
        """
        components = {}
        
        # Compute within-group variance
        components['within'] = self._compute_within_variance(time_series)
        
        # Compute between-group variance
        components['between'] = self._compute_between_variance(time_series)
        
        # Compute temporal variance
        components['temporal'] = self._compute_temporal_variance(time_series)
        
        # Analyze interaction effects
        components['interactions'] = self._analyze_variance_interactions(
            components['within'],
            components['between'],
            components['temporal']
        )
        
        # Calculate relative contributions
        total_variance = sum(components.values())
        components['relative_contributions'] = {
            key: value / total_variance
            for key, value in components.items()
            if key != 'interactions'
        }
        
        return components
        
    def _compute_stability_indices(self,
                                 time_series: np.ndarray,
                                 capability_metrics: Dict[str, float],
                                 variance_components: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate multi-dimensional stability indices.
        
        Indices:
        1. Process Stability Index (PSI)
        2. Variance Stability Index (VSI)
        3. Temporal Stability Index (TSI)
        4. Combined Stability Score (CSS)
        
        Args:
            time_series: Input time series
            capability_metrics: Process capability metrics
            variance_components: Decomposed variance components
            
        Returns:
            Dict containing stability indices and confidence scores
        """
        indices = {}
        
        # Process Stability Index
        indices['psi'] = self._calculate_process_stability(
            capability_metrics,
            self.config['stability_thresholds']['process']
        )
        
        # Variance Stability Index
        indices['vsi'] = self._calculate_variance_stability(
            variance_components,
            self.config['stability_thresholds']['variance']
        )
        
        # Temporal Stability Index
        indices['tsi'] = self._calculate_temporal_stability(
            time_series,
            self.config['stability_thresholds']['temporal']
        )
        
        # Combined Stability Score
        indices['css'] = self._compute_combined_stability(indices)
        
        # Compute confidence intervals
        indices['confidence_intervals'] = self._compute_stability_confidence(indices)
        
        return indices
        
    def _analyze_drift_patterns(self,
                              time_series: np.ndarray,
                              control_limits: Dict[str, float]) -> Dict[str, float]:
        """
        Detect and characterize drift patterns in the time series.
        
        Analysis Components:
        1. Linear drift detection
        2. Non-linear pattern recognition
        3. Regime change detection
        4. Cyclic pattern analysis
        
        Args:
            time_series: Input time series
            control_limits: Statistical control boundaries
            
        Returns:
            Dict containing drift characteristics and pattern metrics
        """
        drift_analysis = {}
        
        # Linear drift detection
        drift_analysis['linear'] = self._detect_linear_drift(
            time_series,
            self.config['drift_thresholds']['linear']
        )
        
        # Non-linear pattern detection
        drift_analysis['nonlinear'] = self._detect_nonlinear_patterns(
            time_series,
            self.config['drift_thresholds']['nonlinear']
        )
        
        # Regime change analysis
        drift_analysis['regime_changes'] = self._detect_regime_changes(
            time_series,
            control_limits
        )
        
        # Cyclic pattern analysis
        drift_analysis['cyclic'] = self._analyze_cyclic_patterns(
            time_series,
            self.config['drift_thresholds']['cyclic']
        )
        
        return drift_analysis
        
    def _default_config(self) -> Dict:
        """
        Generate default configuration for stability analysis.
        
        Parameters:
        1. Process control parameters
        2. Stability thresholds
        3. Drift detection settings
        4. Confidence levels
        """
        return {
            'process_limits': {
                'sigma_multiplier': 3.0,
                'minimum_samples': 30,
                'confidence_level': 0.95
            },
            'stability_thresholds': {
                'process': {'cp_min': 1.33, 'cpk_min': 1.0},
                'variance': {'cv_max': 0.1, 'ratio_max': 1.5},
                'temporal': {'drift_max': 0.02, 'cycle_min': 0.1}
            },
            'drift_thresholds': {
                'linear': 0.01,
                'nonlinear': 0.05,
                'cyclic': 0.1,
                'regime': 0.2
            }
        }
