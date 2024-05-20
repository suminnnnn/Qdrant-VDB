from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import uuid

def initialize_qdrant_client(host, port):
    return QdrantClient(host=host, port=port)

def initialize_qdrant_client_in_memory():
    return QdrantClient(":memory:")

def create_qdrant_collection(client, collection_name, encoder):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=encoder.get_sentence_embedding_dimension(), distance=Distance.COSINE)
    )
    
def create_points(docs, encoder, payload):
    points = []
    for idx, doc in enumerate(docs):
        payload_with_doc = dict(payload)
        payload_with_doc["doc"] = doc
        
        embedding = encoder.encode(doc)
        point = PointStruct(
            id=str(uuid.uuid1()),
            vector=embedding,
            payload=payload_with_doc,
        )
        points.append(point)
    return points

def upsert_points(client, collection_name, points):
    client.upsert(collection_name=collection_name, points=points)

def delete_points(client, collection_name, filter):
    points = client.scroll(collection_name=collection_name, scroll_filter=filter)[0]
    point_ids = [point.id for point in points]
    client.delete_vectors(collection_name=collection_name, wait=True, points=point_ids, vectors=[''])
    client.delete(collection_name=collection_name, wait=True, points_selector=point_ids)

def update_points(client, collection_name, filter, new_docs, encoder, payload):
    delete_points(client, collection_name, filter)
    new_points = create_points(new_docs, encoder, payload)
    upsert_points(client, collection_name, new_points)

def search_qdrant(client, collection_name, user_query, encoder, query_filter, limit=2):
    query_vector = encoder.encode(user_query).tolist()
    hits = client.search(collection_name=collection_name, query_vector=query_vector, query_filter=query_filter, limit=limit)
    if hits:
        return hits[1].payload['doc']
    else:
        return None
