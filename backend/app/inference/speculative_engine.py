import os
import sys
import time
import math
import logging
import threading
from typing import AsyncGenerator, List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from app.providers.base import ChatMessage, CompletionResponse, StreamChunk, ModelInfo
from app.config import settings

logger = logging.getLogger(__name__)

class SpeculativeNeuralEngine:
    """
    High-Performance Speculative Decoding Neural Engine.
    Combines a fast Draft Model (e.g. 0.5B) with a heavy Target Model (e.g. 1.5B - 14B)
    to accelerate CPU/GPU token generation without losing any output quality.
    
    Mathematical Guarantee: Sampling distribution matches the Target model exactly.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SpeculativeNeuralEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        target_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        draft_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        gamma: int = 4  # Number of speculative draft tokens to generate per step
    ):
        if self._initialized:
            return

        self.target_model_name = target_model_name
        self.draft_model_name = draft_model_name
        self.device = device
        self.gamma = gamma

        self.tokenizer = None
        self.target_model = None
        self.draft_model = None
        self.is_loaded = False
        self._load_lock = threading.Lock()
        self._initialized = True

    def ensure_models_loaded(self):
        """Lazy load both Draft and Target models."""
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return
            try:
                print(f"[*] [SPECULATIVE] Loading Tokenizer from {self.draft_model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.draft_model_name,
                    trust_remote_code=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                # Load Draft Model (Fast 0.5B)
                print(f"[*] [SPECULATIVE] Loading Draft Model: {self.draft_model_name} on {self.device}...")
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.draft_model_name,
                    dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.draft_model.eval()

                # Load Target Model (Accurate 1.5B - 14B)
                print(f"[*] [SPECULATIVE] Loading Target Model: {self.target_model_name} on {self.device}...")
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.target_model_name,
                    dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to(self.device)
                self.target_model.eval()

                self.is_loaded = True
                print(f"[SUCCESS] [SPECULATIVE] Engine loaded successfully with Draft={self.draft_model_name} and Target={self.target_model_name}!")
            except Exception as e:
                self.is_loaded = False
                print(f"[ERROR] Failed to load speculative models: {e}")
                raise RuntimeError(f"Could not load speculative models: {e}")

    def format_messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Format messages using ChatML."""
        hf_messages = []
        for msg in messages:
            role = msg.role if msg.role in ["system", "user", "assistant"] else "user"
            hf_messages.append({"role": role, "content": msg.content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                hf_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = ""
            for m in hf_messages:
                prompt += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"
            return prompt

    def generate_speculative(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        gamma: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute Speculative Decoding:
        1. Draft model proposes `gamma` tokens.
        2. Target model verifies in a single forward pass.
        3. Accepts matching tokens using rejection sampling.
        """
        self.ensure_models_loaded()
        gamma = gamma or self.gamma

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        initial_len = input_ids.shape[1]

        t0 = time.time()
        
        # We can leverage transformers optimized assistant_model generation or custom loop
        with torch.inference_mode():
            output_ids = self.target_model.generate(
                input_ids,
                assistant_model=self.draft_model,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=max(temperature, 0.01),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        elapsed = time.time() - t0
        generated_ids = output_ids[0][initial_len:]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        num_tokens = len(generated_ids)
        speed = num_tokens / max(elapsed, 0.001)

        return {
            "text": output_text,
            "tokens_generated": num_tokens,
            "elapsed_seconds": round(elapsed, 3),
            "tokens_per_second": round(speed, 2),
            "target_model": self.target_model_name,
            "draft_model": self.draft_model_name
        }

    def generate_stream_tokens(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Yield tokens generated by speculative decoding in a worker thread."""
        self.ensure_models_loaded()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            **inputs,
            assistant_model=self.draft_model,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=max(temperature, 0.01),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )

        def worker():
            with torch.inference_mode():
                self.target_model.generate(**generate_kwargs)

        thread = threading.Thread(target=worker)
        thread.start()

        for new_text in streamer:
            if new_text:
                yield new_text

        thread.join()

speculative_engine = SpeculativeNeuralEngine()
