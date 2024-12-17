from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import importlib
import os
import sys
from typing import Dict, List, Any
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Self-Organizing AI Experiments API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track running experiments
active_experiments = {}

class ExperimentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

def load_experiment_module(experiment_name: str):
    """Dynamically load an experiment module."""
    try:
        # Ensure experiments directory is in path
        experiments_dir = os.path.join(os.path.dirname(__file__), '../../experiments')
        sys.path.append(experiments_dir)
        
        # Import the module
        module = importlib.import_module(experiment_name)
        return module
    except Exception as e:
        logger.error(f"Failed to load experiment {experiment_name}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_name} not found")

@app.get("/experiments")
async def list_experiments() -> List[str]:
    """List all available experiments."""
    experiments_dir = os.path.join(os.path.dirname(__file__), '../../experiments')
    experiments = []
    
    for file in os.listdir(experiments_dir):
        if file.endswith('.py') and not file.startswith('__'):
            experiments.append(file[:-3])  # Remove .py extension
            
    return experiments

@app.get("/experiments/status")
async def get_experiment_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all running experiments."""
    return active_experiments

@app.post("/experiments/{experiment_name}/run")
async def run_experiment(experiment_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run a specific experiment with optional parameters."""
    try:
        # Load the experiment module
        module = load_experiment_module(experiment_name)
        
        # Get the main experiment class
        exp_class = None
        for item in dir(module):
            if "Experiment" in item and item != "Experiment":
                exp_class = getattr(module, item)
                break
                
        if not exp_class:
            raise HTTPException(status_code=404, detail="No experiment class found")
            
        # Initialize and run the experiment
        experiment = exp_class()
        
        # Track the experiment
        active_experiments[experiment_name] = {
            "status": ExperimentStatus.RUNNING,
            "params": params,
            "start_time": asyncio.get_event_loop().time()
        }
        
        # Run experiment in background task
        async def run_and_update():
            try:
                if params:
                    result = await experiment.run_experiment(**params)
                else:
                    result = await experiment.run_experiment()
                    
                active_experiments[experiment_name].update({
                    "status": ExperimentStatus.COMPLETED,
                    "result": "Success",
                    "end_time": asyncio.get_event_loop().time()
                })
            except Exception as e:
                logger.error(f"Experiment {experiment_name} failed: {str(e)}")
                active_experiments[experiment_name].update({
                    "status": ExperimentStatus.FAILED,
                    "error": str(e),
                    "end_time": asyncio.get_event_loop().time()
                })
        
        # Start the experiment without waiting
        asyncio.create_task(run_and_update())
        
        return {
            "message": f"Experiment {experiment_name} started",
            "status": ExperimentStatus.RUNNING
        }
        
    except Exception as e:
        logger.error(f"Failed to run experiment {experiment_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/experiments/{experiment_name}/stop")
async def stop_experiment(experiment_name: str) -> Dict[str, str]:
    """Stop a running experiment."""
    if experiment_name not in active_experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
        
    # TODO: Implement proper experiment stopping mechanism
    active_experiments[experiment_name]["status"] = ExperimentStatus.COMPLETED
    return {"message": f"Experiment {experiment_name} stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)