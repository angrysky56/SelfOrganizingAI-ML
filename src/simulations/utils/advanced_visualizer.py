"""
Advanced visualization tools for simulation analysis
Includes network graphs, correlation analysis, and phase space exploration
"""

import torch
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import gaussian_kde
from ..utils.gpu_memory import estimate_memory_usage

class AdvancedVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('dark_background')
        self.color_map = plt.cm.viridis

    def plot_agent_network(self, state: Dict[str, Any], distance_threshold: float = 0.3) -> None:
        """Create network graph of agent interactions."""
        agents = state['agents'].cpu().numpy()
        positions = agents[:, :2]
        velocities = agents[:, 2:]

        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(len(positions)):
            G.add_node(i, pos=positions[i], velocity=velocities[i])
        
        # Add edges based on proximity
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < distance_threshold:
                    G.add_edge(i, j, weight=1.0 - dist/distance_threshold)

        # Plot
        plt.figure(figsize=self.figsize)
        pos = nx.get_node_attributes(G, 'pos')
        
        # Edge colors based on weight
        edge_colors = [G[u][v]['weight'] for u, v in G.edges()]
        
        # Node colors based on velocity magnitude
        node_colors = [np.linalg.norm(G.nodes[n]['velocity']) for n in G.nodes()]
        
        nx.draw_networkx(G, pos=pos, 
                        node_color=node_colors, 
                        edge_color=edge_colors,
                        node_size=100,
                        with_labels=False,
                        cmap=self.color_map,
                        edge_cmap=self.color_map)
        
        plt.title('Agent Interaction Network')
        plt.colorbar(plt.cm.ScalarMappable(cmap=self.color_map), 
                    label='Velocity Magnitude')

    def plot_phase_space(self, states_history: List[Dict[str, Any]], 
                        sample_rate: int = 10) -> None:
        """Create phase space visualization of system evolution."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Sample states to prevent overcrowding
        sampled_states = states_history[::sample_rate]
        
        # Extract positions and velocities
        positions = []
        velocities = []
        for state in sampled_states:
            agents = state['agents'].cpu().numpy()
            pos = agents[:, :2]
            vel = agents[:, 2:]
            positions.append(pos.mean(axis=0))
            velocities.append(vel.mean(axis=0))
            
        positions = np.array(positions)
        velocities = np.array(velocities)
        
        # Create phase space plot
        scatter = ax.scatter(positions[:, 0], 
                           positions[:, 1], 
                           velocities[:, 0],
                           c=velocities[:, 1],
                           cmap=self.color_map)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('X Velocity')
        plt.colorbar(scatter, label='Y Velocity')
        plt.title('System Phase Space Evolution')

    def plot_correlation_matrix(self, stats_history: List[Dict[str, float]]) -> None:
        """Create correlation matrix of system metrics."""
        # Convert stats history to DataFrame
        data = np.array([[
            stats['mean_position'],
            stats['position_std'],
            stats['mean_velocity'],
            stats['velocity_std'],
            stats['mean_distance'],
            stats['energy']
        ] for stats in stats_history])
        
        # Calculate correlation matrix
        corr = np.corrcoef(data.T)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, 
                   xticklabels=['Position μ', 'Position σ', 'Velocity μ', 
                               'Velocity σ', 'Distance μ', 'Energy'],
                   yticklabels=['Position μ', 'Position σ', 'Velocity μ', 
                               'Velocity σ', 'Distance μ', 'Energy'],
                   annot=True, 
                   cmap='coolwarm')
        plt.title('Metric Correlation Matrix')

    def plot_density_evolution(self, states_history: List[Dict[str, Any]], 
                             num_frames: int = 4) -> None:
        """Plot evolution of agent density distribution."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        indices = np.linspace(0, len(states_history)-1, num_frames, dtype=int)
        
        for idx, ax in zip(indices, axes.flat):
            state = states_history[idx]
            agents = state['agents'].cpu().numpy()
            positions = agents[:, :2]
            
            # Calculate density
            x = positions[:, 0]
            y = positions[:, 1]
            k = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[-1:1:100j, -1:1:100j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))
            
            # Plot density
            ax.contourf(xi, yi, zi.reshape(xi.shape), levels=50, cmap='viridis')
            ax.set_title(f'Step {state["metadata"]["iteration"]}')
            
        plt.tight_layout()
        plt.suptitle('Agent Density Evolution', y=1.02)

    def plot_emergent_patterns(self, state: Dict[str, Any], 
                             window_size: int = 50) -> None:
        """Analyze and visualize emergent patterns in agent behavior."""
        agents = state['agents'].cpu().numpy()
        positions = agents[:, :2]
        velocities = agents[:, 2:]
        
        plt.figure(figsize=self.figsize)
        
        # Subplot 1: Position clusters
        plt.subplot(221)
        plt.scatter(positions[:, 0], positions[:, 1], 
                   c=np.linalg.norm(velocities, axis=1),
                   cmap=self.color_map)
        plt.title('Position Clusters')
        plt.colorbar(label='Velocity Magnitude')
        
        # Subplot 2: Velocity alignment
        plt.subplot(222)
        plt.quiver(positions[:, 0], positions[:, 1],
                  velocities[:, 0], velocities[:, 1],
                  np.linalg.norm(velocities, axis=1))
        plt.title('Velocity Alignment')
        
        # Subplot 3: Local density
        plt.subplot(223)
        density = gaussian_kde(positions.T)(positions.T)
        plt.scatter(positions[:, 0], positions[:, 1], 
                   c=density, cmap='plasma')
        plt.colorbar(label='Local Density')
        plt.title('Agent Density')
        
        # Subplot 4: Velocity distribution
        plt.subplot(224)
        vel_mag = np.linalg.norm(velocities, axis=1)
        sns.histplot(vel_mag, kde=True)
        plt.title('Velocity Distribution')
        
        plt.tight_layout()