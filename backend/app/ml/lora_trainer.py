import os
import math
import time
import json
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    target_modules: List[str] = None
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 4
    max_seq_len: int = 512

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


class LoRALinear(nn.Module):
    """
    LoRA Linear Layer: adds trainable low-rank decomposition B*A to frozen weights W.
    
    Formula: output = (W + (alpha/r) * B @ A) * x
    W is frozen. Only B (d x r) and A (r x k) are trained.
    Param savings: (d*k) frozen vs. (d*r + r*k) trainable = up to 99% fewer trainable params.
    """
    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Frozen original weight
        self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Trainable LoRA matrices: A initialized with kaiming_uniform, B initialized with zeros
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * lora_out


class LoRATrainer:
    """
    Full LoRA fine-tuning pipeline for AETHER local models.
    Injects trainable LoRA adapters, trains on custom dataset, saves adapter weights.
    """
    def __init__(self, config: Optional[LoRAConfig] = None):
        self.config = config or LoRAConfig()
        self.adapters_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "lora_adapters"
        )
        os.makedirs(self.adapters_dir, exist_ok=True)

    def inject_lora_adapters(self, model: nn.Module) -> nn.Module:
        """Replace target projection layers with LoRA-augmented versions."""
        replaced = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if any(t in name for t in self.config.target_modules):
                    parts = name.split('.')
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    lora_layer = LoRALinear(module, self.config.rank, self.config.alpha)
                    setattr(parent, parts[-1], lora_layer)
                    replaced += 1

        print(f"[LoRA] Injected adapters into {replaced} layers (rank={self.config.rank}, alpha={self.config.alpha})")
        return model

    def train(self, model: nn.Module, tokenizer, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Training loop: freeze base model, train only LoRA matrices via AdamW.
        dataset: list of {"prompt": "...", "response": "..."} entries.
        """
        model = self.inject_lora_adapters(model)

        # Freeze all non-LoRA parameters
        trainable_params = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += param.numel()
            if "lora_A" in name or "lora_B" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        pct = trainable_params / max(1, total_params) * 100
        print(f"[LoRA] Trainable params: {trainable_params:,} / {total_params:,} ({pct:.3f}%)")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.config.learning_rate,
            weight_decay=0.01
        )

        model.train()
        logs = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            steps = 0

            for sample in dataset:
                prompt = sample.get("prompt", "")
                response = sample.get("response", "")
                full_text = f"{prompt}{response}"

                inputs = tokenizer(
                    full_text,
                    return_tensors="pt",
                    max_length=self.config.max_seq_len,
                    truncation=True
                )

                if inputs["input_ids"].shape[1] < 4:
                    continue

                input_ids = inputs["input_ids"]
                labels = input_ids.clone()

                try:
                    with torch.enable_grad():
                        outputs = model(input_ids=input_ids, labels=labels)
                        loss = outputs.loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    steps += 1
                except Exception as e:
                    continue

            avg_loss = epoch_loss / max(1, steps)
            logs.append({"epoch": epoch + 1, "loss": round(avg_loss, 4)})
            print(f"[LoRA] Epoch {epoch + 1}/{self.config.epochs} — Loss: {avg_loss:.4f}")

        self.save_adapters(model)
        return {"status": "success", "trainable_params": trainable_params, "logs": logs}

    def save_adapters(self, model: nn.Module):
        """Save only LoRA adapter weights (tiny files, ~5-50MB vs full model)."""
        adapter_state = {}
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                adapter_state[name] = param.data.cpu()

        path = os.path.join(self.adapters_dir, "aether_lora_v1.pt")
        torch.save(adapter_state, path)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"[LoRA] Adapters saved: {path} ({size_mb:.2f} MB)")

lora_trainer = LoRATrainer()
