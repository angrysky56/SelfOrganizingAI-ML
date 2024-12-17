"""
Self-Organizing Agents Simulation
Implements emergent behavior while being memory-efficient
"""

import asyncio
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base_simulation import BaseSimulation, SimulationConfig
import prometheus_client
from prometheus_client import start_http_server, Gauge
import time
import signal
import sys

# Metrics
AGENT_COUNT = Gauge('agent_count', 'Number of active agents')
MEAN_POSITION = Gauge('mean_position', 'Mean position of agents')
MEAN_VELOCITY = Gauge('mean_velocity', 'Mean velocity of agents')
SYSTEM_ENERGY = Gauge('system_energy', 'Total system energy')

class SelfOrganizingSimulation(BaseSimulation):
    def __init__(self, config: Optional[SimulationConfig] = None):
        super().__init__(config)
        self.attraction_strength = 0.5
        self.repulsion_strength = 0.3
        self.coherence_strength = 0.2
        self.running = True
        
    def signal_handler(self, signum, frame):
        print("\nStopping simulation gracefully...")
        self.running = False
        
    async def initialize(self, num_agents: int = None) -> Dict[str, Any]:
        """Initialize agents with random positions and velocities."""
        num_agents = num_agents or self.config.max_agents
        
        # Initialize on CPU then transfer to GPU to manage memory
        initial_state = {
            'agents': torch.rand((num_agents, 4), device='cpu') * 2 - 1,  # x, y, vx, vy
            'properties': torch.rand((num_agents, 4), device='cpu'),  # Additional properties
        }
        
        # Transfer to GPU if available
        state = await super().initialize(
            {k: v.to(self.device) for k, v in initial_state.items()}
        )
        
        return state

    # ... [previous simulation code remains unchanged until run_simulation] ...

    async def run_continuous(self):
        """Run simulation continuously and update metrics."""
        try:
            # Start Prometheus metrics server
            start_http_server(8000)
            print("Metrics server started on port 8000")
            
            # Set up signal handlers
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            print("Starting continuous simulation. Press Ctrl+C to stop.")
            
            while self.running:
                state = await self.initialize()
                AGENT_COUNT.set(len(state['agents']))
                
                try:
                    for _ in range(100):  # Run in episodes
                        if not self.running:
                            break
                            
                        state = await self.step(state)
                        stats = self.get_state_statistics(state)
                        
                        # Update metrics
                        MEAN_POSITION.set(stats['mean_position'])
                        MEAN_VELOCITY.set(stats['mean_velocity'])
                        SYSTEM_ENERGY.set(stats['energy'])
                        
                        # Print progress
                        if _ % 10 == 0:
                            print(f"Episode {_}/100 - Energy: {stats['energy']:.4f}")
                        
                        await asyncio.sleep(0.1)  # Control simulation speed
                        
                except Exception as e:
                    print(f"Error during simulation: {e}")
                finally:
                    self.cleanup()
                    
                if self.running:
                    print("\nStarting new episode...")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            print(f"Fatal error: {e}")
            self.running = False
        finally:
            print("\nSimulation stopped.")
            self.cleanup()

# Main entry point
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
        await sim.run_continuous()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"Simulation failed: {e}")
    finally:
        sim.cleanup()

if __name__ == "__main__":
    asyncio.run(main())