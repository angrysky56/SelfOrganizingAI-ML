"""
Competitive Dynamics Experiment
Studies emergence of competitive and cooperative behaviors between agent groups
"""

import asyncio
import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
from src.simulations.prototypes.self_organizing_sim import SelfOrganizingSimulation, SimulationConfig
from src.simulations.utils.advanced_visualizer import AdvancedVisualizer
from src.integrations.milvus_connector import SimulationStore

class CompetitiveDynamicsExperiment:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig(
            batch_size=32,
            max_agents=300,  # More agents for interesting group dynamics
            learning_rate=0.008,
            max_iterations=1500
        )
        self.sim = SelfOrganizingSimulation(self.config)
        self.viz = AdvancedVisualizer()
        
    async def initialize_competing_groups(self, 
                                       num_groups: int = 3,
                                       group_properties: List[Dict] = None) -> Dict[str, Any]:
        """Initialize multiple competing groups with different properties."""
        if group_properties is None:
            # Default properties for different groups
            group_properties = [
                {'speed': 1.0, 'size': 1.0, 'attraction': 0.6},  # Fast, normal group
                {'speed': 0.7, 'size': 1.5, 'attraction': 0.8},  # Slower, larger group
                {'speed': 1.3, 'size': 0.7, 'attraction': 0.4}   # Fast, small group
            ]
        
        agents_per_group = self.config.max_agents // num_groups
        
        all_positions = []
        all_velocities = []
        all_properties = []
        
        # Initialize each group with distinct properties
        for i, props in enumerate(group_properties[:num_groups]):
            # Position groups in different sectors
            angle = 2 * np.pi * i / num_groups
            center = torch.tensor([
                np.cos(angle) * 0.5,
                np.sin(angle) * 0.5
            ])
            
            # Create group positions with some randomness
            positions = center + torch.randn(agents_per_group, 2) * 0.2
            
            # Initialize velocities based on group speed
            velocities = torch.randn(agents_per_group, 2) * props['speed']
            
            # Store group properties
            properties = torch.ones(agents_per_group, 1) * i  # Group ID
            
            all_positions.append(positions)
            all_velocities.append(velocities)
            all_properties.append(properties)
        
        # Combine all groups
        positions = torch.cat(all_positions, dim=0)
        velocities = torch.cat(all_velocities, dim=0)
        properties = torch.cat(all_properties, dim=0)
        
        agents = torch.cat([positions, velocities], dim=1)
        
        initial_state = {
            'agents': agents,
            'group_properties': properties,
            'group_params': group_properties
        }
        
        return await self.sim.initialize(initial_state)
    
    async def run_experiment(self, num_groups: int = 3, 
                           num_steps: int = 1000,
                           save_results: bool = True) -> Dict[str, Any]:
        """Run competitive dynamics experiment."""
        print(f"Starting competitive dynamics experiment with {num_groups} groups...")
        
        if save_results:
            store = SimulationStore()
            store.connect()
            store.initialize_collection()
        
        try:
            # Initialize competing groups
            state = await self.initialize_competing_groups(num_groups)
            states_history = []
            stats_history = []
            competition_metrics = []
            
            # Run simulation
            for step in range(num_steps):
                state = await self.sim.step(state)
                stats = self.sim.get_state_statistics(state)
                
                # Calculate competition metrics
                metrics = self._calculate_competition_metrics(state)
                competition_metrics.append(metrics)
                
                states_history.append(state)
                stats_history.append(stats)
                
                if save_results:
                    await store.store_simulation_state(state, stats)
                
                if step % 100 == 0:
                    dom_group = np.argmax(metrics['dominance_scores'])
                    print(f"Step {step}/{num_steps}, Dominant Group: {dom_group}, "
                          f"Dominance Score: {metrics['dominance_scores'][dom_group]:.3f}")
            
            # Generate visualizations
            print("Generating competition analysis...")
            self._generate_competition_analysis(states_history, competition_metrics)
            
            return {
                'states': states_history,
                'stats': stats_history,
                'competition_metrics': competition_metrics,
                'final_state': states_history[-1]
            }
            
        finally:
            if save_results:
                store.cleanup()
    
    def _calculate_competition_metrics(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for group competition and dominance."""
        agents = state['agents'].cpu().numpy()
        positions = agents[:, :2]
        velocities = agents[:, 2:]
        group_ids = state['group_properties'].cpu().numpy()
        unique_groups = np.unique(group_ids)
        
        # Calculate metrics for each group
        dominance_scores = []
        territory_sizes = []
        interaction_strengths = []
        
        for group_id in unique_groups:
            group_mask = group_ids == group_id
            group_pos = positions[group_mask]
            group_vel = velocities[group_mask]
            
            # Calculate territory size (convex hull area)
            from scipy.spatial import ConvexHull
            if len(group_pos) >= 3:
                hull = ConvexHull(group_pos)
                territory = hull.area
            else:
                territory = 0
                
            # Calculate group cohesion and velocity alignment
            group_center = np.mean(group_pos, axis=0)
            dispersion = np.mean(np.linalg.norm(group_pos - group_center, axis=1))
            alignment = np.mean(np.abs(np.dot(group_vel / np.linalg.norm(group_vel, axis=1, keepdims=True),
                                            np.array([1, 0]))))
            
            # Combined dominance score
            dominance = (0.4 * territory + 0.3 * (1 - dispersion) + 0.3 * alignment)
            
            dominance_scores.append(dominance)
            territory_sizes.append(territory)
            
            # Calculate interaction strengths with other groups
            interactions = []
            for other_id in unique_groups:
                if other_id != group_id:
                    other_mask = group_ids == other_id
                    other_pos = positions[other_mask]
                    
                    # Calculate minimum distances between groups
                    distances = np.min(np.linalg.norm(group_pos[:, np.newaxis] - other_pos, axis=2))
                    interactions.append(np.mean(1 / (1 + distances)))
                    
            interaction_strengths.append(np.mean(interactions))
        
        return {
            'dominance_scores': np.array(dominance_scores),
            'territory_sizes': np.array(territory_sizes),
            'interaction_strengths': np.array(interaction_strengths)
        }
    
    def _generate_competition_analysis(self,
                                    states_history: List[Dict[str, Any]],
                                    competition_metrics: List[Dict[str, Any]]) -> None:
        """Generate visualizations for competition analysis."""
        plt.figure(figsize=(20, 15))
        
        # Plot dominance scores over time
        plt.subplot(231)
        dominance_scores = np.array([m['dominance_scores'] for m in competition_metrics])
        for i in range(dominance_scores.shape[1]):
            plt.plot(dominance_scores[:, i], label=f'Group {i}')
        plt.title('Group Dominance Over Time')
        plt.xlabel('Time Steps')
        plt.ylabel('Dominance Score')
        plt.legend()
        
        # Plot territory sizes
        plt.subplot(232)
        territory_sizes = np.array([m['territory_sizes'] for m in competition_metrics])
        for i in range(territory_sizes.shape[1]):
            plt.plot(territory_sizes[:, i], label=f'Group {i}')
        plt.title('Territory Sizes')
        plt.xlabel('Time Steps')
        plt.ylabel('Territory Size')
        plt.legend()
        
        # Plot interaction strengths
        plt.subplot(233)
        interaction_strengths = np.array([m['interaction_strengths'] for m in competition_metrics])
        for i in range(interaction_strengths.shape[1]):
            plt.plot(interaction_strengths[:, i], label=f'Group {i}')
        plt.title('Inter-group Interaction Strength')
        plt.xlabel('Time Steps')
        plt.ylabel('Interaction Strength')
        plt.legend()
        
        # Final state visualization
        plt.subplot(234)
        final_state = states_history[-1]
        agents = final_state['agents'].cpu().numpy()
        group_ids = final_state['group_properties'].cpu().numpy()
        
        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            plt.scatter(agents[mask, 0], agents[mask, 1], 
                       label=f'Group {group_id}',
                       alpha=0.6)
        plt.title('Final Group Positions')
        plt.legend()
        
        # Group velocity distributions
        plt.subplot(235)
        velocities = agents[:, 2:]
        for group_id in np.unique(group_ids):
            mask = group_ids == group_id
            speed = np.linalg.norm(velocities[mask], axis=1)
            plt.hist(speed, bins=20, alpha=0.5, label=f'Group {group_id}')
        plt.title('Velocity Distributions')
        plt.xlabel('Speed')
        plt.ylabel('Count')
        plt.legend()
        
        # Phase space trajectory
        plt.subplot(236)
        self.viz.plot_phase_space([states_history[0], states_history[-1]])
        plt.title('Phase Space Evolution')
        
        plt.tight_layout()
        plt.savefig(f'competition_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')

if __name__ == "__main__":
    async def main():
        experiment = CompetitiveDynamicsExperiment()
        results = await experiment.run_experiment(num_groups=3, num_steps=1000)
        print("Experiment completed successfully!")
        
    asyncio.run(main())