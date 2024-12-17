"""
Main entry point for running experiments with metrics collection
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import start_http_server, Gauge
import uvicorn
import logging
from typing import Dict, Any
import importlib
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Self-Organizing AI Experiments")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Prometheus metrics
AGENT_COUNT = Gauge('agent_count', 'Number of agents in simulation')
SYSTEM_ENERGY = Gauge('system_energy', 'Total system energy')
PATTERN_SCORE = Gauge('pattern_score', 'Pattern formation score')
INTERACTION_STRENGTH = Gauge('interaction_strength', 'Agent interaction strength')

# Track active experiments
active_experiments = {}

@app.on_event("startup")
async def startup_event():
    """Start Prometheus metrics server"""
    start_http_server(8000)
    logger.info("Metrics server started on port 8000")

@app.get("/experiments")
async def list_experiments():
    """List available experiments"""
    experiments_dir = os.path.join(os.path.dirname(__file__), '../experiments')
    experiments = []

    for file in os.listdir(experiments_dir):
        if file.endswith('.py') and not file.startswith('__'):
            name = file[:-3]
            # Load module to get description
            module = importlib.import_module(f'experiments.{name}')
            doc = module.__doc__ or "No description available"
            experiments.append({"name": name, "description": doc.strip()})

    return experiments

@app.post("/experiments/{experiment_name}/run")
async def run_experiment(experiment_name: str, params: Dict[str, Any] = None):
    """Run a specific experiment with parameters"""
    try:
        # Import experiment module
        module = importlib.import_module(f'experiments.{experiment_name}')

        # Get experiment class or function
        if hasattr(module, f'{experiment_name.title()}Experiment'):
            experiment = getattr(module, f'{experiment_name.title()}Experiment')()
            result = await experiment.run_experiment(**(params or {}))
        else:
            # Assume it's a function like run_flocking_experiment
            experiment_func = getattr(module, f'run_{experiment_name}_experiment')
            result = await experiment_func(**(params or {}))

        # Update metrics from result
        if 'stats' in result:
            last_stats = result['stats'][-1]
            SYSTEM_ENERGY.set(last_stats.get('energy', 0))

        if 'final_state' in result:
            final_state = result['final_state']
            if 'agents' in final_state:
                AGENT_COUNT.set(len(final_state['agents']))

        return {"status": "completed", "message": f"{experiment_name} experiment completed successfully"}

    except Exception as e:
        logger.error(f"Error running experiment {experiment_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/experiments/status")
async def get_experiment_status():
    """Get status of active experiments"""
    return active_experiments

@app.get("/metrics")
async def get_metrics():
    """Get current experiment metrics"""
    return {
        "agent_count": AGENT_COUNT._value.get(),
        "system_energy": SYSTEM_ENERGY._value.get(),
        "pattern_score": PATTERN_SCORE._value.get(),
        "interaction_strength": INTERACTION_STRENGTH._value.get()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
