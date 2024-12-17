"""
Visualization utilities for self-organizing simulations.
Provides real-time and static visualization of agent behaviors.
"""

import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
import plotly.express as px

def create_2d_scatter(state: Dict[str, Any], frame_idx: int = 0, title: str = "Agent Positions") -> go.Figure:
    """
    Create a 2D scatter plot of agent positions with velocity vectors.
    
    Args:
        state: Current simulation state
        frame_idx: Frame number for animation
        title: Plot title
    """
    agents = state['agents'].cpu().numpy()
    positions = agents[:, :2]
    velocities = agents[:, 2:]
    
    # Create quiver plot
    fig = go.Figure()
    
    # Add agent positions
    fig.add_trace(go.Scatter(
        x=positions[:, 0],
        y=positions[:, 1],
        mode='markers',
        marker=dict(
            size=8,
            color=np.linalg.norm(velocities, axis=1),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Speed')
        ),
        name='Agents'
    ))
    
    # Add velocity vectors
    fig.add_trace(go.Scatter(
        x=np.concatenate([positions[:, 0], positions[:, 0] + velocities[:, 0], [None]*len(positions)]),
        y=np.concatenate([positions[:, 1], positions[:, 1] + velocities[:, 1], [None]*len(positions)]),
        mode='lines',
        line=dict(color='rgba(50,50,50,0.2)'),
        name='Velocities'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{title} (Frame {frame_idx})",
        xaxis=dict(range=[-1.2, 1.2], title='X Position'),
        yaxis=dict(range=[-1.2, 1.2], title='Y Position', scaleanchor="x", scaleratio=1),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

def create_3d_visualization(states_history: List[Dict[str, Any]], skip_frames: int = 1) -> go.Figure:
    """
    Create a 3D visualization of agent trajectories over time.
    
    Args:
        states_history: List of simulation states
        skip_frames: Number of frames to skip for performance
    """
    fig = go.Figure()
    
    # Get agent positions over time
    positions = []
    for state in states_history[::skip_frames]:
        positions.append(state['agents'].cpu().numpy()[:, :2])
    positions = np.array(positions)
    
    # Plot trajectory for each agent
    num_agents = positions.shape[1]
    time_points = np.arange(len(positions))
    
    for agent_idx in range(num_agents):
        fig.add_trace(go.Scatter3d(
            x=positions[:, agent_idx, 0],
            y=positions[:, agent_idx, 1],
            z=time_points,
            mode='lines+markers',
            marker=dict(
                size=2,
                colorscale='Viridis',
                color=time_points,
                showscale=False
            ),
            line=dict(width=1),
            opacity=0.6,
            name=f'Agent {agent_idx}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Agent Trajectories Over Time',
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Time Step',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2)
        ),
        showlegend=False
    )
    
    return fig

def plot_statistics(stats_history: List[Dict[str, float]]) -> go.Figure:
    """
    Create a multi-line plot of simulation statistics over time.
    
    Args:
        stats_history: List of statistics dictionaries
    """
    # Convert stats history to DataFrame-like structure
    stats_data = {key: [] for key in stats_history[0].keys()}
    time_steps = list(range(len(stats_history)))
    
    for stats in stats_history:
        for key, value in stats.items():
            stats_data[key].append(value)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Plot each statistic
    for key, values in stats_data.items():
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=values,
            name=key,
            mode='lines',
            hovertemplate=f"{key}: %{{y:.3f}}<br>Step: %{{x}}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title='Simulation Statistics Over Time',
        xaxis_title='Time Step',
        yaxis_title='Value',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    return fig

def create_phase_space_plot(state: Dict[str, Any]) -> go.Figure:
    """
    Create a phase space plot showing position vs velocity distributions.
    
    Args:
        state: Current simulation state
    """
    agents = state['agents'].cpu().numpy()
    positions = agents[:, :2]
    velocities = agents[:, 2:]
    
    # Calculate polar coordinates for coloring
    angles = np.arctan2(velocities[:, 1], velocities[:, 0])
    speeds = np.linalg.norm(velocities, axis=1)
    
    fig = go.Figure()
    
    # Add traces for X and Y components
    for i, (pos, vel, label) in enumerate(zip(
        [positions[:, 0], positions[:, 1]],
        [velocities[:, 0], velocities[:, 1]],
        ['X', 'Y']
    )):
        fig.add_trace(go.Scatter(
            x=pos,
            y=vel,
            mode='markers',
            marker=dict(
                size=8,
                color=angles if i == 0 else speeds,
                colorscale='Twilight' if i == 0 else 'Viridis',
                showscale=True,
                colorbar=dict(
                    title='Angle (rad)' if i == 0 else 'Speed',
                    x=1.1 + i * 0.15
                )
            ),
            name=f'{label} Component'
        ))
    
    # Update layout
    fig.update_layout(
        title='Phase Space Plot',
        xaxis_title='Position',
        yaxis_title='Velocity',
        showlegend=True,
        hovermode='closest'
    )
    
    return fig