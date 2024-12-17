# SelfOrganizingAI-ML

A comprehensive framework for self-organizing AI systems with integrated machine learning capabilities, featuring real-time visualizations and emergent behavior analysis.

## Features
- Self-organizing agent simulations using PyTorch
- Real-time 2D and 3D visualizations of agent behaviors
- Vector database integration with Milvus
- Monitoring stack with Prometheus and Grafana
- GPU acceleration support
- Interactive analysis tools

## Prerequisites
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/angrysk56/SelfOrganizingAI-ML.git
cd SelfOrganizingAI-ML
```

2. Start the services:
```bash
# CPU-only mode
docker-compose up -d

# With GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

3. Check service status:
```bash
docker-compose ps
```

4. Access the interfaces:
- Simulation Metrics: http://localhost:8000
- Grafana Dashboard: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Milvus: localhost:19530

## Architecture

### Core Components
- `src/core/`: Central task controller and orchestration
- `src/knowledge/`: Knowledge representation and mapping
- `src/ml_pipeline/`: Machine learning infrastructure
- `src/simulations/`: Self-organizing simulation framework
- `src/ethics/`: Ethical oversight and bias mitigation

### Visualization Components
- 2D real-time agent plotting
- 3D trajectory visualization
- Phase space analysis
- Statistical monitoring
- Interactive Jupyter notebooks

### Infrastructure
- Milvus for vector operations
- Prometheus for metrics collection
- Grafana for monitoring
- Docker for containerization

## Development Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up pre-commit hooks:
```bash
pre-commit install
```

4. Run tests:
```bash
pytest tests/
```

## Configuration

### Environment Variables
Create a `.env` file based on `.env.example`:
```env
MILVUS_HOST=localhost
MILVUS_PORT=19530
CUDA_VISIBLE_DEVICES=0  # For GPU support
```

### Simulation Parameters
Adjust simulation parameters in `config/default.yaml`:
```yaml
simulation:
  max_agents: 500
  batch_size: 64
  learning_rate: 0.001
```

## Usage Examples

### Running a Simulation
```python
from src.simulations.prototypes.self_organizing_sim import SelfOrganizingSimulation
from src.simulations.utils.visualization import create_2d_scatter

# Initialize simulation
sim = SelfOrganizingSimulation()

# Run simulation
results = await sim.run_simulation(num_steps=100)

# Visualize results
fig = create_2d_scatter(results['final_state'])
fig.show()
```

### Custom Pattern Generation
```python
# Example coming soon
```

## Contributing
1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make changes and commit: `git commit -m 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap
- [ ] Advanced pattern recognition
- [ ] Multi-agent communication protocols
- [ ] Enhanced visualization tools
- [ ] Extended documentation

#

 command: docker logs -f selforganizingai-ml-simulation-1
  By pulling and using the container, you accept the terms and conditions of this license: https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license NOTE: The SHMEM allocation limit is set to the default of 64MB. This may be insufficient for PyTorch. NVIDIA recommends the use of the following flags: docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

Initializing Self-Organizing Simulation...Running simulation on device: cuda.

## Simulation completed successfully! Final Statistics: mean_position: -0.0504 position_std: 0.9959 mean_velocity: -0.0319 velocity_std: 0.5233 mean_distance: 1.6934 energy: 0.6971
Success! The simulation is now working correctly. We can see from the final statistics that:

The agents are well distributed (position_std ≈ 1.0)
They have stabilized velocities (velocity_std ≈ 0.52)
The energy level is stable (energy ≈ 0.70)
Mean positions and velocities are near zero, indicating balanced movement
