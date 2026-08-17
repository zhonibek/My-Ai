import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from app.providers.base import VectorDBProvider, VectorChunk
from app.rag.embeddings import embedding_engine

class PersistentHybridVectorStore(VectorDBProvider):
    """
    Persistent Hybrid Semantic Vector Store for Enterprise RAG.
    Combines Dense Vector Embeddings (Cosine Similarity) with Keyword Frequency Scoring (BM25-style).
    Automatically persists chunk index and embeddings to local storage.
    """

    def __init__(self, storage_path: str = "data/vector_store.json"):
        self.storage_path = storage_path
        # User chunks: { user_id: [ {chunk_dict, "embedding": [float...]} ] }
        self.store: Dict[str, List[Dict[str, Any]]] = {}
        self._load_from_disk()

    def _load_from_disk(self):
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    self.store = json.load(f)
        except Exception:
            self.store = {}

    def _save_to_disk(self):
        try:
            os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.store, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    async def insert_chunks(self, user_id: str, chunks: List[VectorChunk]) -> bool:
        if user_id not in self.store:
            self.store[user_id] = []

        for chunk in chunks:
            # Generate semantic dense embedding
            emb = embedding_engine.embed_text(chunk.content)
            record = {
                "id": chunk.id,
                "file_id": chunk.file_id,
                "file_name": chunk.file_name,
                "content": chunk.content,
                "metadata": chunk.metadata or {},
                "embedding": emb
            }
            self.store[user_id].append(record)

        self._save_to_disk()
        return True

    async def get_chunks_by_file_ids(self, user_id: str, file_ids: List[str]) -> List[VectorChunk]:
        user_records = self.store.get(user_id, [])
        if not user_records or not file_ids:
            return []
        target_ids = set(file_ids)
        result = []
        for r in user_records:
            if r.get("file_id") in target_ids:
                result.append(VectorChunk(
                    id=r["id"],
                    file_id=r["file_id"],
                    file_name=r["file_name"],
                    content=r["content"],
                    metadata=r.get("metadata", {}),
                    score=1.0
                ))
        return result

    async def search_similar(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        file_ids: Optional[List[str]] = None,
        project_id: Optional[str] = None
    ) -> List[VectorChunk]:
        user_records = self.store.get(user_id, [])
        if not user_records:
            return []

        # Filter by file_ids if specific files were attached
        if file_ids:
            target_ids = set(file_ids)
            filtered = [r for r in user_records if r.get("file_id") in target_ids]
            if filtered:
                user_records = filtered

        query_emb = embedding_engine.embed_text(query)
        query_terms = set(query.lower().split())
        scored_chunks = []

        for record in user_records:
            content_lower = record["content"].lower()
            
            # 1. Semantic Dense Cosine Similarity (Weight 0.7)
            semantic_score = embedding_engine.cosine_similarity(query_emb, record.get("embedding", []))
            
            # 2. Keyword lexical overlap (Weight 0.3)
            matches = sum(1 for term in query_terms if term in content_lower)
            keyword_score = matches / max(1, len(query_terms))
            
            # Hybrid combined score
            hybrid_score = round(0.7 * semantic_score + 0.3 * keyword_score, 4)

            scored_chunks.append(VectorChunk(
                id=record["id"],
                file_id=record["file_id"],
                file_name=record["file_name"],
                content=record["content"],
                metadata=record.get("metadata", {}),
                score=hybrid_score
            ))

        # Sort descending by hybrid score
        scored_chunks.sort(key=lambda x: x.score or 0, reverse=True)
        return scored_chunks[:top_k]

vector_store = PersistentHybridVectorStore()
