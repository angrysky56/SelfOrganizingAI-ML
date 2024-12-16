"""
Test self-organizing simulation with memory constraints
"""

import pytest
import torch
import asyncio
from src.simulations.prototypes.self_organizing_sim import (
    SelfOrganizingSimulation,
    SimulationConfig
)

@pytest.fixture
def sim_config():
    return SimulationConfig(
        batch_size=16,
        max_agents=500,
        learning_rate=0.01,
        max_iterations=100,
        memory_limit=2.0  # 2GB limit for testing
    )

@pytest.mark.asyncio
async def test_simulation_memory():
    """Test simulation stays within memory limits."""
    config = SimulationConfig(max_agents=100, memory_limit=1.0)
    sim = SelfOrganizingSimulation(config)
    
    state = await sim.initialize()
    assert sim.memory_usage < config.memory_limit
    
    # Run a few steps
    for _ in range(10):
        state = await sim.step(state)
        assert sim.memory_usage < config.memory_limit
        
    sim.cleanup()

@pytest.mark.asyncio
async def test_simulation_behavior():
    """Test basic self-organizing behavior."""
    config = SimulationConfig(max_agents=50)
    sim = SelfOrganizingSimulation(config)
    
    result = await sim.run_simulation(num_steps=20)
    stats = result['stats_history']
    
    # Check for emergent behavior
    initial_stats = stats[0]
    final_stats = stats[-1]
    
    # Should show some organization (lower standard deviation)
    assert final_stats['position_std'] <= initial_stats['position_std']
    
    sim.cleanup()

@pytest.mark.asyncio
async def test_simulation_stability():
    """Test simulation remains stable."""
    config = SimulationConfig(max_agents=50)
    sim = SelfOrganizingSimulation(config)
    
    result = await sim.run_simulation(num_steps=50)
    stats = result['stats_history']
    
    # Check energy doesn't explode
    energies = [s['energy'] for s in stats]
    assert max(energies) < 10.0
    
    # Positions should remain bounded
    final_state = result['final_state']
    positions = final_state['agents'][:, :2]
    assert torch.all(torch.abs(positions) <= 1.0)
    
    sim.cleanup()

if __name__ == '__main__':
    asyncio.run(test_simulation_memory())
    asyncio.run(test_simulation_behavior())
    asyncio.run(test_simulation_stability())