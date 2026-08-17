import json
import os
import uuid
from typing import List, Dict, Any, Optional

class FineTuningDatasetExporter:
    """Exports chat logs, RAG documents, and custom instruction pairs into JSONL fine-tuning datasets"""

    @staticmethod
    def format_sharegpt(conversations_list: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Formats conversation turns into ShareGPT format for LLaMA-Factory / Unsloth / Axolotl fine-tuning"""
        formatted = []
        for conv in conversations_list:
            items = []
            for msg in conv:
                role = "human" if msg["role"] == "user" else "gpt"
                items.append({"from": role, "value": msg["content"]})
            if items:
                formatted.append({"conversations": items})
        return formatted

    @staticmethod
    def format_alpaca(instruction_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Formats query-answer pairs into Alpaca format"""
        formatted = []
        for pair in instruction_pairs:
            formatted.append({
                "instruction": pair.get("instruction", ""),
                "input": pair.get("input", ""),
                "output": pair.get("output", "")
            })
        return formatted

    @staticmethod
    def export_to_jsonl(dataset: List[Dict[str, Any]], output_filepath: str) -> int:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return len(dataset)

dataset_exporter = FineTuningDatasetExporter()
