"""

Soruyu vektöre çevir, Qdrant'Ta ara, ilgili chunkları döndür (select).

"""

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)
COLLECTION = "docs"


def search(question, k3):
    """ Soruya en yakın chunk'ı getir """
    vector = model.encode(question).tolist()
    hits = client.query_points(COLLECTION, query=vector, limit=k).points
    return [h.payload["text"] for h in hits]