import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "aether_chat.db")

class ChatDatabase:
    """
    Persistent SQLite Database for AETHER AI Conversations, Messages, and RAG History.
    """
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT DEFAULT 'aether-neural-local',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model_used TEXT,
                    reasoning_steps TEXT,
                    sources TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def list_conversations(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, model, created_at, updated_at FROM conversations ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "title": r["title"],
                    "model": r["model"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"]
                }
                for r in rows
            ]

    def create_or_update_conversation(self, conversation_id: str, title: str, model: str = "aether-neural-local"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            cursor.execute("""
                INSERT INTO conversations (id, title, model, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = COALESCE(?, title),
                    model = COALESCE(?, model),
                    updated_at = ?
            """, (conversation_id, title, model, now, title, model, now))
            conn.commit()

    def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY timestamp ASC", (conversation_id,))
            rows = cursor.fetchall()
            messages = []
            for r in rows:
                reasoning = []
                sources = []
                try:
                    if r["reasoning_steps"]:
                        reasoning = json.loads(r["reasoning_steps"])
                except Exception:
                    pass
                try:
                    if r["sources"]:
                        sources = json.loads(r["sources"])
                except Exception:
                    pass

                messages.append({
                    "id": r["id"],
                    "conversation_id": r["conversation_id"],
                    "sender": r["sender"],
                    "content": r["content"],
                    "modelUsed": r["model_used"],
                    "reasoningSteps": reasoning,
                    "sources": sources,
                    "timestamp": r["timestamp"]
                })
            return messages

    def save_message(
        self,
        msg_id: str,
        conversation_id: str,
        sender: str,
        content: str,
        model_used: Optional[str] = None,
        reasoning_steps: Optional[List[Any]] = None,
        sources: Optional[List[Any]] = None
    ):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            reasoning_json = json.dumps(reasoning_steps or [], ensure_ascii=False)
            sources_json = json.dumps(sources or [], ensure_ascii=False)
            
            cursor.execute("""
                INSERT OR REPLACE INTO messages (id, conversation_id, sender, content, model_used, reasoning_steps, sources)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (msg_id, conversation_id, sender, content, model_used, reasoning_json, sources_json))
            
            # Touch conversation updated_at
            now = datetime.utcnow().isoformat()
            cursor.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            conn.commit()

    def delete_conversation(self, conversation_id: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()

db = ChatDatabase()
