"""
Pattern formation experiment with real-time streaming updates
Studies emergence of spatial patterns and structures
"""

import asyncio
import torch
import numpy as np
from typing import Dict, Any, AsyncGenerator
from datetime import datetime
from src.simulations.prototypes.self_organizing_sim import SelfOrganizingSimulation, SimulationConfig

class PatternFormationExperiment:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig(
            batch_size=32,
            max_agents=200,
            learning_rate=0.005,
            max_iterations=2000
        )
        self.sim = SelfOrganizingSimulation(self.config)
        
    async def run_with_updates(self, pattern_type: str = "circle",
                             num_steps: int = 1000) -> AsyncGenerator[Dict[str, Any], None]:
        """Run experiment with streaming updates."""
        print(f"Starting pattern formation experiment with {pattern_type} pattern...")
        
        # Initialize with selected pattern
        state = await self.initialize_pattern(pattern_type)
        
        for step in range(num_steps):
            # Update simulation
            state = await self.sim.step(state)
            metrics = self._calculate_pattern_metrics(state)
            
            # Prepare visualization data
            viz_data = {
                'step': step,
                'positions': state['agents'][:, :2],  # x,y coordinates
                'velocities': state['agents'][:, 2:], # velocity vectors
                'pattern_score': metrics['pattern_score'],
                'density_variation': metrics['density_variation'],
                'spatial_order': metrics['spatial_order'],
                'timestamp': datetime.now().isoformat()
            }
            
            yield viz_data
            
            # Control update rate
            await asyncio.sleep(0.05)  # 20 FPS
            
    def _calculate_pattern_metrics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics for pattern quality."""
        agents = state['agents'].cpu().numpy()
        positions = agents[:, :2]
        
        # Calculate local density variations
        distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
        local_density = (distances < 0.3).sum(axis=1) / len(positions)
        density_variation = np.std(local_density)
        
        # Calculate spatial order parameter
        velocities = agents[:, 2:]
        velocity_norms = np.linalg.norm(velocities, axis=1)
        velocity_norms = np.where(velocity_norms > 0, velocity_norms, 1)  # Avoid division by zero
        normalized_velocities = velocities / velocity_norms[:, np.newaxis]
        spatial_order = np.abs(np.mean(normalized_velocities, axis=0))
        
        # Combined pattern score
        pattern_score = 0.6 * density_variation + 0.4 * np.mean(spatial_order)
        
        return {
            'pattern_score': float(pattern_score),
            'density_variation': float(density_variation),
            'spatial_order': float(np.mean(spatial_order))
        }
        
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