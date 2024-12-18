from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import json
import numpy as np

def setup_collections():
    try:
        # Connect to Milvus
        connections.connect(
            alias="default",
            host="milvus-standalone",
            port="19530"
        )
        print("Successfully connected to Milvus")

        # Define fields for the correlation patterns collection
        fields = [
            FieldSchema(name="pattern_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name="pattern_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="coherence_metrics", dtype=DataType.JSON),
            FieldSchema(name="temporal_context", dtype=DataType.JSON)
        ]

        # Create correlation patterns collection schema
        correlation_schema = CollectionSchema(
            fields=fields,
            description="Correlation patterns with integrated quantum and fractal states"
        )

        # Drop existing collection if it exists
        if "correlation_patterns" in utility.list_collections():
            utility.drop_collection("correlation_patterns")

        # Create the new collection
        correlation_collection = Collection(
            name="correlation_patterns",
            schema=correlation_schema
        )

        # Create HNSW index optimized for GPU
        index_params = {
            "metric_type": "L2",
            "index_type": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 500
            }
        }

        correlation_collection.create_index(
            field_name="vector",
            index_params=index_params
        )
        print("Created correlation_patterns collection with index")

        # Insert a test pattern
        test_vector = np.random.rand(512)  # Example vector
        test_pattern = [
            ["test_pattern_001"],  # pattern_id
            [test_vector.tolist()],  # vector
            ["quantum_neural"],  # pattern_type
            [json.dumps({
                "coherence": 0.95,
                "quantum_stability": 0.94,
                "pattern_coherence": 0.93
            })],  # coherence_metrics
            [json.dumps({
                "timestamp": "2024-12-07",
                "state": "initialized"
            })]  # temporal_context
        ]

        correlation_collection.insert(test_pattern)
        print("Inserted test pattern successfully")

        # Load the collection for searching
        correlation_collection.load()
        print("Collection loaded and ready for search")

        return True

    except Exception as e:
        print(f"Error setting up collections: {e}")
        return False

if __name__ == "__main__":
    setup_collections()