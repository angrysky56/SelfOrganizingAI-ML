from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import json
import numpy as np

def setup_vector_collections():
    try:
        # Connect to Milvus
        connections.connect(
            alias="default", 
            host='localhost',
            port='19530'
        )
        
        # Create correlation patterns collection
        correlation_fields = [
            FieldSchema(name="pattern_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="pattern_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="coherence_metrics", dtype=DataType.JSON),
            FieldSchema(name="temporal_context", dtype=DataType.JSON)
        ]
        
        correlation_schema = CollectionSchema(
            fields=correlation_fields,
            description="Correlation patterns for predictive analysis"
        )
        
        correlation_collection = Collection(
            name="correlation_patterns",
            schema=correlation_schema
        )
        
        # Create index for vector field
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 500}
        }
        correlation_collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        # Create predictive indicators collection
        indicator_fields = [
            FieldSchema(name="indicator_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="indicator_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="forecast_window", dtype=DataType.JSON),
            FieldSchema(name="pattern_memory", dtype=DataType.JSON),
            FieldSchema(name="coherence_metrics", dtype=DataType.JSON)
        ]
        
        indicator_schema = CollectionSchema(
            fields=indicator_fields,
            description="Predictive indicators for system monitoring"
        )
        
        indicator_collection = Collection(
            name="predictive_indicators",
            schema=indicator_schema
        )
        
        # Create index for vector field
        indicator_collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        
        # Load collections
        correlation_collection.load()
        indicator_collection.load()
        
        print("Collections created and loaded successfully")
        
        # Insert initial test vector
        test_pattern = {
            "pattern_id": "test_pattern_001",
            "vector": np.random.rand(512).tolist(),
            "pattern_type": "test",
            "coherence_metrics": json.dumps({"coherence": 0.95}),
            "temporal_context": json.dumps({"timestamp": "2024-12-07"})
        }
        
        correlation_collection.insert([test_pattern])
        
        print("Test pattern inserted successfully")
        return True
        
    except Exception as e:
        print(f"Error setting up collections: {e}")
        return False

if __name__ == "__main__":
    setup_vector_collections()