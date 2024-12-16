"""
Metrics exporter for Prometheus monitoring
"""

from prometheus_client import start_http_server, Gauge, Histogram
import time
from typing import Dict, Any
import torch

class MetricsExporter:
    def __init__(self, port: int = 8000):
        self.port = port
        
        # GPU Metrics
        self.gpu_utilization = Gauge(
            'simulation_gpu_utilization',
            'GPU utilization percentage'
        )
        self.gpu_memory = Gauge(
            'simulation_gpu_memory_usage',
            'GPU memory usage in bytes'
        )
        
        # Simulation Metrics
        self.agent_count = Gauge(
            'simulation_agent_count',
            'Number of active agents'
        )
        self.mean_velocity = Gauge(
            'simulation_agent_velocity_mean',
            'Mean agent velocity'
        )
        self.pattern_density = Gauge(
            'simulation_pattern_density',
            'Agent pattern density'
        )
        
        # Performance Metrics
        self.step_duration = Histogram(
            'simulation_step_duration_seconds',
            'Time taken for each simulation step',
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
        )
        
        # Memory Metrics
        self.batch_size = Gauge(
            'simulation_batch_size',
            'Current processing batch size'
        )
        self.memory_efficiency = Gauge(
            'simulation_memory_efficiency',
            'Memory usage efficiency score'
        )
        
        # Emergence Metrics
        self.emergence_score = Gauge(
            'simulation_emergence_score',
            'Score indicating emergent behavior strength'
        )
        self.pattern_complexity = Gauge(
            'simulation_pattern_complexity',
            'Complexity measure of emerged patterns'
        )
        
        # Group Dynamics Metrics
        self.group_coherence = Gauge(
            'simulation_group_coherence',
            'Measure of agent group coherence'
        )
        self.interaction_strength = Gauge(
            'simulation_interaction_strength',
            'Strength of agent interactions'
        )
        
    def start(self):
        """Start the metrics server."""
        start_http_server(self.port)
        print(f"Metrics server started on port {self.port}")
        
    def update_simulation_metrics(self, state: Dict[str, Any], stats: Dict[str, float]):
        """Update simulation-related metrics."""
        try:
            # Update agent metrics
            agents = state['agents']
            self.agent_count.set(len(agents))
            
            # Update velocity metrics
            velocities = agents[:, 2:]
            mean_vel = float(velocities.mean())
            self.mean_velocity.set(mean_vel)
            
            # Update pattern metrics
            if 'pattern_density' in stats:
                self.pattern_density.set(stats['pattern_density'])
                
            # Calculate and update emergence metrics
            emergence = self._calculate_emergence_score(state)
            self.emergence_score.set(emergence)
            
            # Update group dynamics
            coherence = self._calculate_group_coherence(state)
            self.group_coherence.set(coherence)
            
            interactions = self._calculate_interaction_strength(state)
            self.interaction_strength.set(interactions)
            
        except Exception as e:
            print(f"Error updating simulation metrics: {e}")
            
    def update_performance_metrics(self, step_time: float, 
                                 memory_usage: float,
                                 batch_size: int):
        """Update performance-related metrics."""
        try:
            self.step_duration.observe(step_time)
            self.batch_size.set(batch_size)
            
            # Calculate memory efficiency
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory
                efficiency = 1.0 - (memory_usage / total_memory)
                self.memory_efficiency.set(efficiency)
            
        except Exception as e:
            print(f"Error updating performance metrics: {e}")
            
    def update_gpu_metrics(self):
        """Update GPU-related metrics."""
        try:
            if torch.cuda.is_available():
                # Get GPU utilization
                utilization = torch.cuda.utilization()
                self.gpu_utilization.set(utilization)
                
                # Get memory usage
                memory_used = torch.cuda.memory_allocated()
                self.gpu_memory.set(memory_used)
                
        except Exception as e:
            print(f"Error updating GPU metrics: {e}")
            
    def _calculate_emergence_score(self, state: Dict[str, Any]) -> float:
        """Calculate a score indicating the strength of emergent behavior."""
        try:
            agents = state['agents']
            positions = agents[:, :2]
            velocities = agents[:, 2:]
            
            # Calculate local density variations
            distances = torch.cdist(positions, positions)
            local_density = (distances < 0.3).float().mean(dim=1)
            density_variation = float(torch.std(local_density))
            
            # Calculate velocity alignment
            velocity_alignment = float(torch.mean(torch.abs(
                velocities / (torch.norm(velocities, dim=1, keepdim=True) + 1e-8)
            )))
            
            # Combine metrics
            emergence_score = 0.5 * density_variation + 0.5 * velocity_alignment
            return float(emergence_score)
            
        except Exception as e:
            print(f"Error calculating emergence score: {e}")
            return 0.0
            
    def _calculate_group_coherence(self, state: Dict[str, Any]) -> float:
        """Calculate how coherently agents move together."""
        try:
            velocities = state['agents'][:, 2:]
            
            # Calculate average velocity alignment
            norm_velocities = velocities / (torch.norm(velocities, dim=1, keepdim=True) + 1e-8)
            coherence = float(torch.mean(torch.abs(
                torch.mm(norm_velocities, norm_velocities.t())
            )))
            
            return coherence
            
        except Exception as e:
            print(f"Error calculating group coherence: {e}")
            return 0.0
            
    def _calculate_interaction_strength(self, state: Dict[str, Any]) -> float:
        """Calculate strength of agent interactions."""
        try:
            positions = state['agents'][:, :2]
            
            # Calculate pairwise distances
            distances = torch.cdist(positions, positions)
            
            # Calculate interaction strength (inverse of average distance)
            interaction_strength = float(1.0 / (torch.mean(distances) + 1e-8))
            
            return interaction_strength
            
        except Exception as e:
            print(f"Error calculating interaction strength: {e}")
            return 0.0

# Example usage:
if __name__ == "__main__":
    exporter = MetricsExporter()
    exporter.start()
    
    # Keep the server running
    while True:
        time.sleep(1)