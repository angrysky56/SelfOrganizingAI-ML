from pymilvus import MilvusClient

def get_milvus_client():
    try:
        # Connect to local Milvus standalone instance
        client = MilvusClient("http://localhost:19530")
        
        # Verify connection
        status = client.get_version()
        print(f"Successfully connected to Milvus. Version: {status}")
        
        return client
    except Exception as e:
        print(f"Error connecting to Milvus: {e}")
        return None

if __name__ == "__main__":
    client = get_milvus_client()