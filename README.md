## More broken now- just something to think about currently.

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
python scripts/setup.py
```

## Usage
See `docs/` for detailed documentation on each module.
