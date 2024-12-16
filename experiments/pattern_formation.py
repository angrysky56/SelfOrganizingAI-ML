"""
Pattern formation experiment
Studies emergence of spatial patterns and structures
"""

import asyncio
import torch
import numpy as np
from typing import Dict, Any, List
from src.simulations.prototypes.self_organizing_sim import SelfOrganizingSimulation, SimulationConfig
from src.simulations.utils.advanced_visualizer import AdvancedVisualizer
from src.integrations.milvus_connector import SimulationStore

class PatternFormationExperiment:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig(
            batch_size=32,
            max_agents=200,
            learning_rate=0.005,
            max_iterations=2000
        )
        self.sim = SelfOrganizingSimulation(self.config)
        self.viz = AdvancedVisualizer()
        
    async def run_experiment(self, pattern_type: str, 
                           num_steps: int = 1000,
                           save_results: bool = True) -> Dict[str, Any]:
        """Run pattern formation experiment."""
        print(f"Starting pattern formation experiment with {pattern_type} pattern...")
        
        # Initialize storage if needed
        if save_results:
            store = SimulationStore()
            store.connect()
            store.initialize_collection()
        
        try:
            # Initialize with selected pattern
            state = await self.initialize_pattern(pattern_type)
            states_history = []
            stats_history = []
            pattern_metrics = []
            
            # Run simulation
            for step in range(num_steps):
                # Update simulation
                state = await self.sim.step(state)
                stats = self.sim.get_state_statistics(state)
                pattern_score = self._calculate_pattern_metrics(state)
                
                # Store data
                states_history.append(state)
                stats_history.append(stats)
                pattern_metrics.append(pattern_score)
                
                if save_results:
                    await store.store_simulation_state(state, stats)
                
                # Print progress
                if step % 100 == 0:
                    print(f"Step {step}/{num_steps}, Pattern Score: {pattern_score:.3f}")
            
            # Generate visualizations
            print("Generating visualizations...")
            self._generate_pattern_analysis(states_history, pattern_metrics)
            
            return {
                'states': states_history,
                'stats': stats_history,
                'pattern_metrics': pattern_metrics,
                'final_state': states_history[-1]
            }
            
        finally:
            if save_results:
                store.cleanup()
                
    def _calculate_pattern_metrics(self, state: Dict[str, Any]) -> float:
        """Calculate metrics for pattern quality."""
        agents = state['agents'].cpu().numpy()
        positions = agents[:, :2]
        
        # Calculate local density variations
        distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        local_density = (distances < 0.3).sum(axis=1) / len(positions)
        density_variation = np.std(local_density)
        
        # Calculate spatial order parameter
        velocities = agents[:, 2:]
        velocity_alignment = np.abs(np.mean(velocities / np.linalg.norm(velocities, axis=1)[:, np.newaxis], axis=0))
        spatial_order = np.mean(velocity_alignment)
        
        # Combine metrics
        pattern_score = 0.6 * density_variation + 0.4 * spatial_order
        return float(pattern_score)
        
    def _generate_pattern_analysis(self, 
                                 states_history: List[Dict[str, Any]],
                                 pattern_metrics: List[float]) -> None:
        """Generate comprehensive pattern analysis visualizations."""
        plt.figure(figsize=(15, 10))
        
        # Pattern evolution plot
        plt.subplot(221)
        plt.plot(pattern_metrics)
        plt.title('Pattern Formation Evolution')
        plt.xlabel('Time Steps')
        plt.ylabel('Pattern Score')
        
        # Final pattern visualization
        plt.subplot(222)
        final_state = states_history[-1]
        agents = final_state['agents'].cpu().numpy()
        plt.scatter(agents[:, 0], agents[:, 1], 
                   c=np.linalg.norm(agents[:, 2:], axis=1),
                   cmap='viridis')
        plt.title('Final Pattern Formation')
        plt.colorbar(label='Velocity Magnitude')
        
        # Local density plot
        plt.subplot(223)
        self.viz.plot_density_evolution([states_history[0], states_history[-1]], num_frames=2)
        
        # Spatial correlation plot
        plt.subplot(224)
        self.viz.plot_correlation_matrix([self.sim.get_state_statistics(state) 
                                        for state in states_history[-10:]])
        
        plt.tight_layout()
        plt.savefig(f'pattern_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        
    async def initialize_pattern(self, pattern_type: str) -> Dict[str, Any]:
        """Initialize agents in specific patterns."""
        if pattern_type == "circle":
            return await self._init_circle_pattern()
        elif pattern_type == "grid":
            return await self._init_grid_pattern()
        elif pattern_type == "random_clusters":
            return await self._init_random_clusters()
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
            
    async def _init_circle_pattern(self) -> Dict[str, Any]:
        """Initialize agents in a circular pattern."""
        n_agents = self.config.max_agents
        angles = torch.linspace(0, 2*np.pi, n_agents)
        radius = 0.8
        
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        
        # Add small random perturbations
        positions = torch.stack([x, y], dim=1) + torch.randn(n_agents, 2) * 0.05
        velocities = torch.zeros(n_agents, 2)
        
        agents = torch.cat([positions, velocities], dim=1)
        return await self.sim.initialize({'agents': agents})
        
    async def _init_grid_pattern(self) -> Dict[str, Any]:
        """Initialize agents in a grid pattern."""
        n_per_side = int(np.sqrt(self.config.max_agents))
        x = torch.linspace(-0.8, 0.8, n_per_side)
        y = torch.linspace(-0.8, 0.8, n_per_side)
        
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        velocities = torch.zeros_like(positions)
        
        agents = torch.cat([positions, velocities], dim=1)
        return await self.sim.initialize({'agents': agents})
        
    async def _init_random_clusters(self) -> Dict[str, Any]:
        """Initialize agents in random clusters."""
        n_clusters = 4
        agents_per_cluster = self.config.max_agents // n_clusters
        
        cluster_centers = torch.rand(n_clusters, 2) * 1.6 - 0.8
        
        all_positions = []
        all_velocities = []
        
        for center in cluster_centers:
            positions = center + torch.randn(agents_per_cluster, 2) * 0.2
            velocities = torch.zeros(agents_per_cluster, 2)
            
            all_positions.append(positions)
            all_velocities.append(velocities)
            
        positions = torch.cat(all_positions, dim=0)
        velocities = torch.cat(all_velocities, dim=0)
        
        agents = torch.cat([positions, velocities], dim=1)
        return await self.sim.initialize({'agents': agents})

if __name__ == "__main__":
    # Run experiments with different patterns
    async def main():
        experiment = PatternFormationExperiment()
        patterns = ["circle", "grid", "random_clusters"]
        
        for pattern in patterns:
            print(f"\nRunning experiment with {pattern} pattern")
            results = await experiment.run_experiment(pattern)
            print(f"Completed {pattern} pattern experiment")
            
    asyncio.run(main())