"""
Visualization tools for monitoring self-organizing simulations
"""

import torch
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

class SimulationVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('dark_background')  # Better for GPU monitoring

    def plot_realtime(self, state: Dict[str, Any]) -> None:
        """Create real-time plot of agent positions and velocities."""
        fig = plt.figure(figsize=self.figsize)
        
        # Convert to numpy for plotting
        agents = state['agents'].cpu().numpy()
        positions = agents[:, :2]
        velocities = agents[:, 2:]
        
        # Agent positions scatter plot
        ax1 = fig.add_subplot(221)
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], 
                            c=np.linalg.norm(velocities, axis=1),
                            cmap='viridis')
        ax1.set_title('Agent Positions and Velocities')
        plt.colorbar(scatter, ax=ax1, label='Velocity Magnitude')
        
        # Velocity distribution
        ax2 = fig.add_subplot(222)
        vel_mag = np.linalg.norm(velocities, axis=1)
        sns.kdeplot(data=vel_mag, ax=ax2)
        ax2.set_title('Velocity Distribution')
        
        # Memory usage
        ax3 = fig.add_subplot(223)
        memory = state['metadata']['memory_usage']
        ax3.bar(['GPU Memory'], [memory], color='cyan')
        ax3.set_title(f'GPU Memory Usage: {memory:.2f} GB')
        
        plt.tight_layout()
        return fig

    def create_animation(self, states_history: List[Dict[str, Any]], 
                        interval: int = 50) -> FuncAnimation:
        """Create animation of simulation evolution."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        
        # First frame
        agents = states_history[0]['agents'].cpu().numpy()
        scatter = ax.scatter(agents[:, 0], agents[:, 1], c='cyan', alpha=0.6)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        def update(frame):
            agents = states_history[frame]['agents'].cpu().numpy()
            scatter.set_offsets(agents[:, :2])
            velocities = agents[:, 2:]
            scatter.set_array(np.linalg.norm(velocities, axis=1))
            ax.set_title(f'Step {frame}')
            return scatter,
            
        anim = FuncAnimation(fig, update, frames=len(states_history),
                           interval=interval, blit=True)
        return anim

    def plot_statistics(self, stats_history: List[Dict[str, float]]) -> None:
        """Plot evolution of simulation statistics."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Extract statistics
        steps = range(len(stats_history))
        mean_pos = [s['mean_position'] for s in stats_history]
        pos_std = [s['position_std'] for s in stats_history]
        mean_vel = [s['mean_velocity'] for s in stats_history]
        energy = [s['energy'] for s in stats_history]
        
        # Position statistics
        axes[0,0].plot(steps, mean_pos, label='Mean Position')
        axes[0,0].fill_between(steps, 
                             np.array(mean_pos) - np.array(pos_std),
                             np.array(mean_pos) + np.array(pos_std),
                             alpha=0.3)
        axes[0,0].set_title('Position Statistics')
        axes[0,0].legend()
        
        # Velocity
        axes[0,1].plot(steps, mean_vel, label='Mean Velocity', color='orange')
        axes[0,1].set_title('Velocity Evolution')
        axes[0,1].legend()
        
        # Energy
        axes[1,0].plot(steps, energy, label='System Energy', color='red')
        axes[1,0].set_title('Energy Evolution')
        axes[1,0].legend()
        
        # Phase space (optional)
        if len(mean_pos) > 1:
            axes[1,1].plot(mean_pos[:-1], mean_pos[1:], 'g.', alpha=0.5)
            axes[1,1].set_title('Position Phase Space')
        
        plt.tight_layout()
        return fig

    def plot_gpu_metrics(self, memory_history: List[float]) -> None:
        """Plot GPU memory usage over time."""
        plt.figure(figsize=(10, 5))
        plt.plot(memory_history, label='GPU Memory', color='cyan')
        plt.axhline(y=torch.cuda.get_device_properties(0).total_memory / 1e9,
                   color='r', linestyle='--', label='Total VRAM')
        plt.title('GPU Memory Usage Over Time')
        plt.xlabel('Step')
        plt.ylabel('Memory (GB)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()