"""
Simulation Guard Module for Ethical AI Systems
Implements safety bounds and monitoring for self-organizing simulations.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from ..base import BaseAnalyzer, AnalysisResult, AnalysisContext
from datetime import datetime
import logging

class SimulationGuard(BaseAnalyzer):
    """Ensures ethical bounds and safety in self-organizing simulations."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.checks = {
            'boundary_conditions': self._check_boundary_conditions,
            'convergence': self._check_convergence,
            'stability': self._check_stability,
            'diversity': self._check_diversity,
            'resource_usage': self._check_resource_usage
        }
        self.history = []
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict[str, Any]:
        return {
            'min_samples': 100,
            'confidence_threshold': 0.9,
            'stability_window': 50,
            'diversity_threshold': 0.3,
            'max_resource_usage': 0.9,
            'boundary_limits': {
                'min_value': -1000.0,
                'max_value': 1000.0,
                'max_gradient': 100.0
            }
        }

    async def analyze(self, data: Dict[str, Any]) -> AnalysisResult:
        """Analyze simulation state for ethical compliance and safety."""
        self._update_history(data)
        results, confidences, violations = await self._run_checks(data)
        
        overall_safety = np.mean(list(results.values()))
        overall_confidence = np.mean(confidences)
        
        context = AnalysisContext(
            timestamp=datetime.now(),
            parameters=self.config,
            metadata={
                'check_results': results,
                'violations': violations,
                'recommendations': self._generate_recommendations(violations)
            }
        )
        
        if violations:
            self.logger.warning(f"Simulation violations: {violations}")
        
        return AnalysisResult(
            value=float(overall_safety),
            confidence=float(overall_confidence),
            context=context,
            details={'violations': violations}
        )

    async def _check_diversity(self, data: Dict[str, Any]) -> Tuple[float, float, List[str]]:
        """Ensure simulation maintains adequate diversity."""
        state = data.get('state', {})
        violations = []
        
        if not state or 'values' not in state:
            return 0.0, 0.0, ['No diversity data']
            
        unique_ratio = len(np.unique(state['values'])) / len(state['values'])
        
        if unique_ratio < self.config['diversity_threshold']:
            violations.append('Insufficient diversity')
            
        return float(unique_ratio), 1.0, violations

    async def _check_resource_usage(self, data: Dict[str, Any]) -> Tuple[float, float, List[str]]:
        """Monitor computational resource usage."""
        resources = data.get('resources', {})
        violations = []
        
        if not resources:
            return 0.0, 0.0, ['No resource data']
            
        usage_metrics = {
            'memory': resources.get('memory_usage', 0),
            'cpu': resources.get('cpu_usage', 0),
        }
        
        max_usage = max(usage_metrics.values())
        if max_usage > self.config['max_resource_usage']:
            violations.append(f'Excessive resource usage: {max_usage:.2f}')
            
        safety_score = 1.0 - (max_usage / self.config['max_resource_usage'])
        return float(safety_score), 1.0, violations

    def _update_history(self, data: Dict[str, Any]) -> None:
        """Update simulation history with new data."""
        self.history.append(data.get('state', {}))
        if len(self.history) > self.config['stability_window']:
            self.history.pop(0)

    async def _run_checks(self, data: Dict[str, Any]) -> Tuple[Dict[str, float], List[float], List[str]]:
        """Run all safety checks."""
        results = {}
        confidences = []
        violations = []
        
        for check_name, check_func in self.checks.items():
            score, confidence, check_violations = await check_func(data)
            results[check_name] = score
            confidences.append(confidence)
            violations.extend(check_violations)
            
        return results, confidences, violations

    def _generate_recommendations(self, violations: List[str]) -> List[str]:
        """Generate actionable recommendations based on violations."""
        recommendations = []
        
        violation_handlers = {
            'Excessive resource usage': 'Consider reducing batch size or simulation complexity',
            'Insufficient diversity': 'Increase exploration parameters or add perturbation',
            'Values below minimum boundary': 'Add value clamping or gradient clipping',
            'Values above maximum boundary': 'Add value clamping or gradient clipping',
            'Gradient too steep': 'Reduce learning rate or add gradient smoothing',
            'Unstable variance': 'Add stability constraints or increase regularization'
        }
        
        for violation in violations:
            for pattern, recommendation in violation_handlers.items():
                if pattern in violation:
                    recommendations.append(recommendation)
                    
        return recommendations

    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status summary."""
        return {
            'total_violations': len(self.history),
            'current_checks': len(self.checks),
            'safety_score': np.mean([h.get('safety_score', 0) for h in self.history[-5:]]),
            'active_warnings': [h.get('violations', []) for h in self.history[-1:]][0] if self.history else []
        }
