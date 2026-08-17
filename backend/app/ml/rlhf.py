import json
import os
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class DPOPreferencePair(BaseModel):
    pair_id: str = None
    prompt: str
    chosen_response: str
    rejected_response: str
    user_id: str = "default_user"
    feedback_score: float = 1.0

class RLHFFeedbackLogger:
    """Logs user preference feedback (chosen vs rejected responses) for DPO / RLHF alignment training"""

    def __init__(self, filepath: str = "data/dpo_preferences.jsonl"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def log_preference(self, pair: DPOPreferencePair) -> bool:
        if not pair.pair_id:
            pair.pair_id = f"dpo_{uuid.uuid4().hex[:8]}"

        record = {
            "pair_id": pair.pair_id,
            "prompt": pair.prompt,
            "chosen": pair.chosen_response,
            "rejected": pair.rejected_response,
            "user_id": pair.user_id,
            "score": pair.feedback_score
        }

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True

rlhf_logger = RLHFFeedbackLogger()
