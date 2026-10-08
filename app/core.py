"""RAG retrieval with interchangeable lexical / embeddings+Qdrant backends."""
from __future__ import annotations
import os, re, time, uuid
from dataclasses import dataclass
from collections import Counter
from typing import Protocol

@dataclass
class Chunk:
    id: str
    tenant: str
    document_id: str
    content: str

@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def chunk_text(s: str, size: int = 650, overlap: int = 100) -> list[str]:
    s = s.strip()
    if not s: return []
    out = []
    i = 0
    while i < len(s):
        part = s[i:i+size]
        out.append(part)
        if i+size >= len(s): break
        i += size-overlap
    return out

class Store(Protocol):
    def upsert(self, tenant: str, doc_id: str, content: str) -> int: ...
    def search(self, tenant: str, query: str, limit: int) -> list[Hit]: ...

class LexicalStore:
    """Dependency-free deterministic baseline; NOT vector search."""
    def __init__(self): self.items: list[Chunk] = []
    def upsert(self, tenant, doc_id, content):
        self.items = [c for c in self.items if not (c.tenant == tenant and c.document_id == doc_id)]
        chunks = chunk_text(content)
        self.items.extend(Chunk(uuid.uuid4().hex, tenant, doc_id, c) for c in chunks)
        return len(chunks)
    def search(self, tenant, query, limit):
        terms = Counter(tokenize(query))
        hits = []
        for chunk in self.items:
            if chunk.tenant != tenant: continue
            words = Counter(tokenize(chunk.content))
            score = sum(min(q, words[w]) for w,q in terms.items()) / max(sum(terms.values()),1)
            if score > 0: hits.append(Hit(chunk, float(score)))
        return sorted(hits, key=lambda h: (-h.score, h.chunk.id))[:limit]

class SemanticStore:
    """Real embeddings, vector DB and optional cross-encoder reranking."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from qdrant_client import QdrantClient, models
        self.models = models
        self.encoder = SentenceTransformer(os.getenv('EMBED_MODEL','sentence-transformers/all-MiniLM-L6-v2'))
        self.reranker = CrossEncoder(os.getenv('RERANK_MODEL','cross-encoder/ms-marco-MiniLM-L-6-v2')) if os.getenv('RERANK','1') == '1' else None
        self.client = QdrantClient(path=os.getenv('QDRANT_PATH','./.qdrant'))
        self.collection = 'tenant_docs'
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(size=self.encoder.get_sentence_embedding_dimension(), distance=models.Distance.COSINE))
        self.client.create_payload_index(self.collection,field_name='tenant',field_schema=models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection,field_name='document_id',field_schema=models.PayloadSchemaType.KEYWORD)
    def upsert(self, tenant, doc_id, content):
        from qdrant_client import models
        old = models.Filter(must=[models.FieldCondition(key='tenant', match=models.MatchValue(value=tenant)),models.FieldCondition(key='document_id', match=models.MatchValue(value=doc_id))])
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=old))
        chunks = chunk_text(content)
        if not chunks: return 0
        vecs = self.encoder.encode(chunks, normalize_embeddings=True).tolist()
        points = [models.PointStruct(id=str(uuid.uuid4()), vector=v, payload={'tenant':tenant,'document_id':doc_id,'content':c}) for v,c in zip(vecs,chunks)]
        self.client.upsert(self.collection,points=points)
        return len(chunks)
    def search(self, tenant, query, limit):
        from qdrant_client import models
        query_vec = self.encoder.encode(query,normalize_embeddings=True).tolist()
        hits = self.client.query_points(collection_name=self.collection,query=query_vec,query_filter=models.Filter(must=[models.FieldCondition(key='tenant',match=models.MatchValue(value=tenant))]),limit=max(limit*3,limit),with_payload=True).points
        candidates = [Hit(Chunk(str(h.id),h.payload['tenant'],h.payload['document_id'],h.payload['content']),float(h.score)) for h in hits]
        if self.reranker and candidates:
            scores = self.reranker.predict([(query,h.chunk.content) for h in candidates])
            candidates = [Hit(h.chunk,float(s)) for h,s in zip(candidates,scores)]
            candidates.sort(key=lambda h:h.score, reverse=True)
        return candidates[:limit]

def make_store() -> Store:
    if os.getenv('RETRIEVER','lexical') == 'semantic': return SemanticStore()
    return LexicalStore()
