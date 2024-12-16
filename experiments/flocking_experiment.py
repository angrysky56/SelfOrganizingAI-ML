"""
Flocking behavior experiment
Demonstrates emergence of coordinated movement patterns
"""

import asyncio
import torch
from typing import Dict, Any
from src.simulations.prototypes.self_organizing_sim import SelfOrganizingSimulation, SimulationConfig
from src.simulations.utils.advanced_visualizer import AdvancedVisualizer
from src.integrations.milvus_connector import SimulationStore

async def run_flocking_experiment(num_agents: int = 100, 
                                num_steps: int = 1000,
                                save_results: bool = True) -> Dict[str, Any]:
    """Run flocking behavior experiment."""
    
    # Configure simulation
    config = SimulationConfig(
        batch_size=32,
        max_agents=num_agents,
        learning_rate=0.01,
        max_iterations=num_steps
    )
    
    # Initialize components
    sim = SelfOrganizingSimulation(config)
    viz = AdvancedVisualizer()
    
    if save_results:
        store = SimulationStore()
        store.connect()
        store.initialize_collection()
    
    # Run simulation
    print("Starting flocking experiment...")
    state = await sim.initialize(num_agents)
    states_history = []
    stats_history = []
    
    try:
        for step in range(num_steps):
            # Update simulation
            state = await sim.step(state)
            stats = sim.get_state_statistics(state)
            
            # Store state
            states_history.append(state)
            stats_history.append(stats)
            
            # Save to Milvus if requested
            if save_results:
                await store.store_simulation_state(state, stats)
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}/{num_steps}, Energy: {stats['energy']:.3f}")
                
    except Exception as e:
        print(f"Error during simulation: {e}")
        raise
    finally:
        if save_results:
            store.cleanup()
    
    # Generate visualizations
    print("Generating visualizations...")
    
    # Plot network structure
    viz.plot_agent_network(states_history[-1])
    
    # Plot phase space
    viz.plot_phase_space(states_history)
    
    # Plot correlation matrix
    viz.plot_correlation_matrix(stats_history)
    
    # Plot density evolution
    viz.plot_density_evolution(states_history)
    
    # Plot emergent patterns
    viz.plot_emergent_patterns(states_history[-1])
    
    print("Experiment completed successfully!")
    
    return {
        'states': states_history,
        'stats': stats_history,
        'final_state': states_history[-1]
    }

if __name__ == "__main__":
    # Run the experiment
    asyncio.run(run_flocking_experiment())