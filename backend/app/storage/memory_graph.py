import re
import uuid
from typing import List, Dict, Any, Optional
from app.storage.database import db
from app.rag.embeddings import embedding_engine

class UserMemoryGraph:
    """
    Episodic Long-Term Memory & Knowledge Graph for AETHER AI.
    
    1. Fact Extraction: Automatically captures user profile entities (name, stack, projects, rules).
    2. Vector Indexing: Computes dense semantic embeddings for each remembered fact.
    3. Semantic Memory Injection: Automatically recalls context across conversations.
    """

    PATTERNS = [
        (r"(?:my name is|меня зовут|менің атым)\s+([A-Za-zА-Яа-яЁёӘәІіҢңҒғҮүҰұҚқӨөҺһ]+)", "identity", "User's name is {match}"),
        (r"(?:i am a|я работаю|я являюсь|мен)\s+([A-Za-zА-Яа-яЁё0-9\s]{3,30}?)(?:\.|\,|$)", "profession", "User profession/role: {match}"),
        (r"(?:i (?:use|code in|program in)|пишу на|мой стек|стек)\s+([A-Za-z0-9\+\#\s\,]{3,40})", "tech_stack", "User tech stack: {match}"),
        (r"(?:my project is|мой проект|разрабатываю)\s+([A-Za-zА-Яа-яЁё0-9\s\-_]{3,40})", "project", "Active project: {match}"),
        (r"(?:i prefer|я предпочитаю|мне нравится)\s+([A-Za-zА-Яа-яЁё0-9\s\-_]{3,50})", "preference", "User preference: {match}")
    ]

    def extract_and_store_facts(self, user_message: str, user_id: str = "default_user") -> List[str]:
        """Scans user message for long-term facts and saves them to memory DB."""
        extracted_facts = []
        for pattern, category, template in self.PATTERNS:
            matches = re.findall(pattern, user_message, re.IGNORECASE)
            for match in matches:
                clean_match = match.strip()
                if len(clean_match) >= 2:
                    fact_str = template.format(match=clean_match)
                    fact_id = f"mem_{uuid.uuid4().hex[:8]}"
                    emb = embedding_engine.embed_text(fact_str)
                    db.save_memory_fact(
                        fact_id=fact_id,
                        category=category,
                        fact=fact_str,
                        embedding_list=emb,
                        user_id=user_id
                    )
                    extracted_facts.append(fact_str)
        return extracted_facts

    def recall_relevant_memories(self, query: str, user_id: str = "default_user", top_k: int = 4) -> List[str]:
        """Retrieves top-k semantically relevant facts from user's long-term memory."""
        all_memories = db.get_all_memories(user_id=user_id)
        if not all_memories:
            return []

        query_emb = embedding_engine.embed_text(query)
        scored = []
        for m in all_memories:
            emb = m.get("embedding")
            if emb:
                sim = embedding_engine.cosine_similarity(query_emb, emb)
                scored.append((sim, m["fact"]))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [fact for sim, fact in scored[:top_k] if sim > 0.15]

    def format_memory_context(self, query: str, user_id: str = "default_user") -> str:
        """Returns formatted string of user memories to inject into LLM system prompt."""
        memories = self.recall_relevant_memories(query, user_id=user_id)
        if not memories:
            return ""
        
        ctx = "--- RECALLED USER PROFILE & LONG-TERM MEMORY ---\n"
        for idx, m in enumerate(memories, 1):
            ctx += f"- {m}\n"
        ctx += "------------------------------------------------\n"
        return ctx

memory_graph = UserMemoryGraph()
