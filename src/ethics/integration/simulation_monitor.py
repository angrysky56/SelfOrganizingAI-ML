"""
Integration between ethics system and simulations
"""

from typing import Dict, Any, Optional
import asyncio
from ..core.analyzers.bias_detector import BiasDetector
from ..core.analyzers.fairness_metrics import FairnessMetrics
from ..core.analyzers.simulation_guard import SimulationGuard

class SimulationMonitor:
    def __init__(self):
        self.bias_detector = BiasDetector()
        self.fairness_metrics = FairnessMetrics()
        self.simulation_guard = SimulationGuard()
        
    async def monitor_simulation(self, simulation_state: Dict[str, Any]) -> Dict[str, Any]:
        """Run ethical checks during simulation."""
        tasks = [
            self.bias_detector.analyze(simulation_state),
            self.fairness_metrics.analyze(simulation_state),
            self.simulation_guard.analyze(simulation_state)
        ]
        
        results = await asyncio.gather(*tasks)
        
        return {
            'bias_score': results[0].value,
            'fairness_score': results[1].value,
            'safety_score': results[2].value,
            'violations': results[2].details.get('violations', []),
            'recommendations': results[2].context.metadata.get('recommendations', [])
        }
    
    async def check_simulation(self, simulation_state: Dict[str, Any]) -> bool:
        """Quick check if simulation can continue."""
        guard_result = await self.simulation_guard.analyze(simulation_state)
        return len(guard_result.details['violations']) == 0