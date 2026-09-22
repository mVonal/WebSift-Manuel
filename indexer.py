"""

- Bir URL'yi çek
- Temizle
- Parçala
- Embed'le 
- Qdrant'a yaz.

Sitelerde trafilatura ile ayıklar; olmazsa BeautifulSoup'a düşer.

Stratejiler:
Isolate + Compress(temizle), Write (Qdrant'a yazma)

"""

import requests
import trafilatura
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

model = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(host="localhost", port=6333)
COLLECTION = "docs"

def fetch_clean(url):
    """ Sayfayı çeker, asıl metni ayıklar. Ham HTML dışarı çıkmaz (Isolate + Compress). """
    html = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).text
    text = trafilatura.extract(html)
    if text:
        return text
    
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["nav", "footer", "header", "script", "style"]):
        tag.decompose()
    return soup.get_text(seperator="\n", strip=True)


def chunk(text, size=300, overlap=10):
    """Metni örtüşmeli parçalara böl (kelime bazlı örtüşme, kelime kesmez.) """
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current = [], ""
    for p  in paras:
        if len(current) + len(p) <=size:
            current = (current + "\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            tail = " ".join(current.split()[-overlap:])
            current = (tail + "\n" + p).strip()
    if current:
        chunks.append(current)
    return chunks


def index_urls(urls):
    """ Verilen URL'leri çek, parçala, Qdrant'a yaz. Her çağrıda temiz başlar. """
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(COLLECTION, vectors_config=VectorParams(size=384, distance=Distance.COSINE))

    points, pid = [], 0
    for url in urls:
        try:
            text = fetch_clean(url)
        except Exception as e:
            print(f"  UYARI: {url} çekilemedi ({e})")
            continue
        if not text:
            print(f"  UYARI: {url} boş içerik, atlandı")
            continue
        chunks = chunk(text)
        for c in chunks:
            points.append(PointStruct(id=pid, vector=model.encode(c).tolist(),
                                      payload={"text": c, "url": url}))
            pid += 1
        print(f"  {url} -> {chunks.__len__()} chunk")

    if points:
        client.upsert(COLLECTION, points=points)
    print(f"Toplam {len(points)} chunk indekslendi.")
    return len(points)