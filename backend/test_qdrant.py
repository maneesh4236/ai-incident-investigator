from qdrant_client import QdrantClient

client = QdrantClient(path="./qdrant_data")

print(client.get_collections())

client.close()