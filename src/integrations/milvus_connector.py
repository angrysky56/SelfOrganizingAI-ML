"""
Milvus integration for storing and retrieving simulation results
"""

import numpy as np
from typing import Dict, List, Any, Optional
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)

class SimulationStore:
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.collection_name = "simulation_results"
        
    def connect(self) -> None:
        """Establish connection to Milvus."""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            print("Connected to Milvus successfully")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

    def initialize_collection(self) -> None:
        """Create collection for simulation results if it doesn't exist."""
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="agent_states", dtype=DataType.FLOAT_VECTOR, dim=4),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="iteration", dtype=DataType.INT64),
                FieldSchema(name="metrics", dtype=DataType.FLOAT_VECTOR, dim=6)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Storage for self-organizing simulation results"
            )
            
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            
            collection.create_index(
                field_name="agent_states",
                index_params=index_params
            )
            collection.create_index(
                field_name="metrics",
                index_params=index_params
            )
            
            print(f"Collection '{self.collection_name}' initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize collection: {e}")
            raise

    async def store_simulation_state(self, state: Dict[str, Any], 
                                   stats: Dict[str, float]) -> None:
        """Store simulation state and statistics in Milvus."""
        try:
            collection = Collection(self.collection_name)
            
            agents = state['agents'].cpu().numpy()
            metrics = np.array([
                stats['mean_position'],
                stats['position_std'],
                stats['mean_velocity'],
                stats['velocity_std'],
                stats['mean_distance'],
                stats['energy']
            ])
            
            data = [
                agents.flatten(),
                int(state['metadata']['timestamp']),
                state['metadata']['iteration'],
                metrics
            ]
            
            collection.insert(data)
            
        except Exception as e:
            print(f"Failed to store simulation state: {e}")
            raise

    async def query_similar_states(self, 
                                 reference_state: np.ndarray,
                                 limit: int = 10) -> List[Dict[str, Any]]:
        """Query for similar simulation states."""
        try:
            collection = Collection(self.collection_name)
            collection.load()
            
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10}
            }
            
            results = collection.search(
                data=[reference_state],
                anns_field="agent_states",
                param=search_params,
                limit=limit,
                output_fields=["timestamp", "iteration", "metrics"]
            )
            
            similar_states = []
            for hits in results:
                for hit in hits:
                    similar_states.append({
                        "id": hit.id,
                        "distance": hit.distance,
                        "timestamp": hit.entity.get("timestamp"),
                        "iteration": hit.entity.get("iteration"),
                        "metrics": hit.entity.get("metrics")
                    })
            
            return similar_states
            
        except Exception as e:
            print(f"Failed to query similar states: {e}")
            raise

    def cleanup(self) -> None:
        """Close Milvus connection."""
        try:
            connections.disconnect("default")
            print("Disconnected from Milvus")
        except Exception as e:
            print(f"Error during cleanup: {e}")
