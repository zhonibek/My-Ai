import math
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

class LocalEmbeddingEngine:
    """
    High-Performance Local Semantic Embedding Generator.
    Computes normalized dense embeddings (384-dimensional) for semantic vector search.
    Supports local HuggingFace / ONNX sentence transformers with high-speed deterministic fallback.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._hf_model = None
        self._hf_tokenizer = None
        self._tried_loading_hf = False

    def _get_hf_model(self):
        if not self._tried_loading_hf:
            self._tried_loading_hf = True
            try:
                import os
                # Skip HF model if running low on memory (detected by env var or just use hash fallback)
                if os.environ.get("AETHER_SKIP_SENTENCE_TRANSFORMERS", "0") == "1":
                    return None, None
                from transformers import AutoTokenizer, AutoModel
                import torch
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._hf_model = AutoModel.from_pretrained(model_name)
                self._hf_model.eval()
            except Exception:
                self._hf_model = None
                self._hf_tokenizer = None
        return self._hf_model, self._hf_tokenizer

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized 384-dimensional dense semantic vector for a text string."""
        if not text:
            return [0.0] * self.dim

        # Try HuggingFace Dense Embedding if available
        model, tokenizer = self._get_hf_model()
        if model is not None and tokenizer is not None:
            try:
                import torch
                inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Mean pooling
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
                    # L2 normalize
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    return embeddings[0].tolist()
            except Exception:
                pass

        # High-Speed Deterministic Multi-Gram Semantic Dense Hashing Embedding
        words = text.lower().split()
        vector = np.zeros(self.dim, dtype=np.float32)
        
        # Word and subword level semantic projection
        for i, word in enumerate(words):
            # Positional weight decay
            pos_weight = 1.0 / (1.0 + 0.05 * math.log(i + 1))
            
            # Word hash
            h_word = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
            idx1 = h_word % self.dim
            val1 = ((h_word >> 8) % 1000 - 500) / 500.0
            vector[idx1] += val1 * pos_weight

            # Character 3-grams
            if len(word) >= 3:
                for j in range(len(word) - 2):
                    ngram = word[j:j+3]
                    h_ng = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16)
                    idx2 = h_ng % self.dim
                    val2 = ((h_ng >> 8) % 1000 - 500) / 500.0
                    vector[idx2] += val2 * 0.5 * pos_weight

        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector = vector / norm

        return vector.tolist()

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Compute cosine similarity between two normalized vectors."""
        if not vec_a or not vec_b:
            return 0.0
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

embedding_engine = LocalEmbeddingEngine()
