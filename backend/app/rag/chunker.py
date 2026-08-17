from typing import List, Dict, Any

class TextChunker:
    """Splits long text documents into overlapping semantic chunks for RAG indexing"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, file_id: str, file_name: str) -> List[Dict[str, Any]]:
        chunks = []
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{file_id}_chunk_{chunk_idx}",
                "file_id": file_id,
                "file_name": file_name,
                "content": chunk_text,
                "chunk_index": chunk_idx
            })
            chunk_idx += 1
            if i + self.chunk_size >= len(words):
                break
        return chunks

text_chunker = TextChunker()
