"""
Self-Organizing Agents Simulation
Implements emergent behavior while being memory-efficient
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base_simulation import BaseSimulation, SimulationConfig

class SelfOrganizingSimulation(BaseSimulation):
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__(config)
        self.attraction_strength = 0.5
        self.repulsion_strength = 0.3
        self.coherence_strength = 0.2
        
    async def initialize(self, num_agents: int = None) -> Dict[str, Any]:
        """Initialize agents with random positions and velocities."""
        num_agents = num_agents or self.config.max_agents
        
        # Initialize on CPU then transfer to GPU to manage memory
        initial_state = {
            'agents': torch.rand((num_agents, 4), device='cpu') * 2 - 1,  # x, y, vx, vy
            'properties': torch.rand((num_agents, 4), device='cpu'),  # Additional properties, matching dimension
        }
        
        # Transfer to GPU if available
        state = await super().initialize(
            {k: v.to(self.device) for k, v in initial_state.items()}
        )
        
        return state
        
    def _process_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Update agent positions and velocities."""
        positions = batch[:, :2]
        velocities = batch[:, 2:]
        
        # Calculate forces between agents
        distances = torch.cdist(positions, positions)
        
        # Calculate forces
        attraction = self._calculate_attraction(positions, distances)
        repulsion = self._calculate_repulsion(positions, distances)
        coherence = self._calculate_coherence(velocities)
        
        # Combine forces
        total_force = (
            self.attraction_strength * attraction +
            self.repulsion_strength * repulsion +
            self.coherence_strength * coherence
        )
        
        # Update velocities and positions
        new_velocities = velocities + total_force * self.config.learning_rate
        new_positions = positions + new_velocities
        
        # Limit speeds to prevent instability
        velocities_norm = torch.norm(new_velocities, dim=1, keepdim=True)
        new_velocities = torch.where(
            velocities_norm > 1.0,
            new_velocities / velocities_norm,
            new_velocities
        )
        
        # Keep agents within bounds [-1, 1]
        new_positions = torch.clamp(new_positions, -1.0, 1.0)
        
        # Combine updates
        return torch.cat([new_positions, new_velocities], dim=1)
        
    def _calculate_attraction(self, positions: torch.Tensor, 
                            distances: torch.Tensor) -> torch.Tensor:
        """Calculate attraction forces between agents."""
        # Prevent self-attraction
        distances = distances + torch.eye(distances.shape[0], device=self.device) * 1e6
        
        # Calculate direction vectors
        directions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Normalize directions (ensure matching dimensions)
        norm_directions = directions / (distances.unsqueeze(-1).expand(-1, -1, 2) + 1e-8)
        
        # Calculate attraction force (stronger at medium distances)
        force = torch.exp(-distances.unsqueeze(-1).expand(-1, -1, 2) / 0.5) * norm_directions
        
        return force.mean(dim=1)
        
    def _calculate_repulsion(self, positions: torch.Tensor, 
                           distances: torch.Tensor) -> torch.Tensor:
        """Calculate repulsion forces between agents."""
        # Prevent self-repulsion
        distances = distances + torch.eye(distances.shape[0], device=self.device) * 1e6
        
        # Calculate direction vectors
        directions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Normalize directions (ensure matching dimensions)
        norm_directions = directions / (distances.unsqueeze(-1).expand(-1, -1, 2) + 1e-8)
        
        # Calculate repulsion force (stronger at close distances)
        force = torch.exp(-distances.unsqueeze(-1).expand(-1, -1, 2) * 2) * norm_directions
        
        return -force.mean(dim=1)  # Negative for repulsion
        
    def _calculate_coherence(self, velocities: torch.Tensor) -> torch.Tensor:
        """Calculate coherence forces to align velocities."""
        # Calculate average velocity of neighbors
        avg_velocity = velocities.mean(dim=0, keepdim=True)
        
        # Force towards average velocity (ensure matching dimensions)
        force = avg_velocity.expand(velocities.shape[0], -1) - velocities
        
        return force
        
    def get_state_statistics(self, state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate statistics about the current simulation state."""
        agents = state['agents']
        positions = agents[:, :2]
        velocities = agents[:, 2:]
        
        stats = {
            'mean_position': positions.mean().item(),
            'position_std': positions.std().item(),
            'mean_velocity': velocities.mean().item(),
            'velocity_std': velocities.std().item(),
            'mean_distance': torch.cdist(positions, positions).mean().item(),
            'energy': torch.norm(velocities, dim=1).mean().item()
        }
        
        return stats
        
    async def run_simulation(self, num_steps: int) -> Dict[str, Any]:
        """Run simulation for specified number of steps."""
        state = await self.initialize()
        stats_history = []
        
        try:
            for _ in range(num_steps):
                state = await self.step(state)
                stats = self.get_state_statistics(state)
                stats_history.append(stats)
                
                # Check memory usage
                if state['metadata']['memory_usage'] > self.config.memory_limit:
                    print(f"Warning: Memory usage exceeded limit: {state['metadata']['memory_usage']:.2f} GB")
                    break
                    
        except Exception as e:
            print(f"Simulation error: {e}")
            self.cleanup()
            raise
            
        return {
            'final_state': state,
            'stats_history': stats_history
        }

# Main entry point
import asyncio

async def main():
    print("Initializing Self-Organizing Simulation...")
    
    # Configure simulation
    config = SimulationConfig(
        batch_size=64,
        max_agents=500,
        learning_rate=0.001,
        max_iterations=1000
    )
    
    # Create and run simulation
    sim = SelfOrganizingSimulation(config)
    print(f"Running simulation on device: {sim.device}")
    
    try:
        results = await sim.run_simulation(num_steps=100)
        print("\nSimulation completed successfully!")
        print("\nFinal Statistics:")
        for key, value in results['stats_history'][-1].items():
            print(f"{key}: {value:.4f}")
            
    except Exception as e:
        print(f"Simulation failed: {e}")
    finally:
        sim.cleanup()

if __name__ == "__main__":
    asyncio.run(main())