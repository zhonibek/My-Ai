import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure ai-research directory is in sys.path
research_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if research_dir not in sys.path:
    sys.path.insert(0, research_dir)

from models.model_v01 import ModelV01


# =====================================================================
# 1. SYNTHETIC REASONING DATASET FOR GRPO REINFORCEMENT LEARNING
# =====================================================================

class ReasoningPromptDataset(Dataset):
    """Dataset providing prompt contexts and ground-truth validation targets."""
    def __init__(self, vocab_size: int = 32000, num_samples: int = 100, prompt_len: int = 16, answer_len: int = 32):
        self.vocab_size = vocab_size
        self.prompts = torch.randint(100, vocab_size - 100, (num_samples, prompt_len))
        self.targets = torch.randint(100, vocab_size - 100, (num_samples, answer_len))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx], self.targets[idx]


# =====================================================================
# 2. GROUP RELATIVE POLICY OPTIMIZATION (GRPO) TRAINER (Kimi K3 / R1)
# =====================================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Reinforcement Learning Engine.
    Samples group of G responses per prompt, normalizes rewards across the group,
    and optimizes policy without requiring a separate critic value network.
    """
    def __init__(self, model: nn.Module, group_size: int = 4, lr: float = 1e-5, clip_eps: float = 0.2):
        self.model = model
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def compute_rule_rewards(self, completions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Rule-Based Reward Function:
        1. Token Accuracy Reward (matching target tokens)
        2. Coherence & Diversity Reward
        """
        G, T = completions.shape
        rewards = torch.zeros(G, device=completions.device)
        for i in range(G):
            # Target matching accuracy
            target_slice = targets[:min(T, targets.shape[0])]
            comp_slice = completions[i, :target_slice.shape[0]]
            matches = (comp_slice == target_slice).float().mean()
            # Reward in range [0.0, 1.0]
            rewards[i] = matches.item() * 0.8 + 0.2 * (torch.unique(completions[i]).numel() / max(T, 1))
        return rewards

    def train_step_grpo(self, prompt: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Perform one GRPO policy optimization step for a single prompt."""
        self.model.train()
        device = prompt.device
        prompt_len = prompt.shape[0]
        gen_len = 24

        # Generate group of G responses
        group_completions = []
        with torch.no_grad():
            for _ in range(self.group_size):
                curr_ids = prompt.unsqueeze(0)  # [1, prompt_len]
                for _ in range(gen_len):
                    logits, _ = self.model(curr_ids)
                    next_token = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, dim=-1), num_samples=1)
                    curr_ids = torch.cat([curr_ids, next_token], dim=-1)
                group_completions.append(curr_ids[0, prompt_len:])

        completions_tensor = torch.stack(group_completions)  # [G, gen_len]

        # Compute rewards and normalize (Group Relative Advantage)
        raw_rewards = self.compute_rule_rewards(completions_tensor, target)
        mean_r = raw_rewards.mean()
        std_r = raw_rewards.std() + 1e-8
        advantages = (raw_rewards - mean_r) / std_r  # [G]

        # Policy Gradient Optimization
        total_loss = 0.0
        self.optimizer.zero_grad()

        for g in range(self.group_size):
            full_seq = torch.cat([prompt, completions_tensor[g]]).unsqueeze(0)  # [1, prompt_len + gen_len]
            logits, _ = self.model(full_seq)
            log_probs = F.log_softmax(logits[:, prompt_len-1:-1, :], dim=-1)
            target_tokens = completions_tensor[g].unsqueeze(0)
            token_log_probs = log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Policy objective: - Advantage * sum(log_p)
            policy_loss = -advantages[g] * token_log_probs.mean()
            total_loss += policy_loss

        loss = total_loss / self.group_size
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item()
        }


# =====================================================================
# 3. EXPERIMENT RUNNERS (EXP-001 MoE Baseline & EXP-002 GRPO RL Loop)
# =====================================================================

def run_experiment_exp002(device: str = "cpu", num_steps: int = 5) -> Dict[str, Any]:
    print("==========================================================================")
    print(" 🔬 EXPERIMENT ID: EXP-002 | AETHER MoE Architecture + GRPO RL Training")
    print("==========================================================================")

    # Initialize Sparse MoE Model (Kimi K3 Style)
    model = ModelV01(
        vocab_size=32000, d_model=256, n_heads=4, hidden_dim=512,
        num_layers=3, use_moe=True, num_experts=4, top_k=2
    )
    model.to(device)
    param_count = model.get_num_params()

    print(f"[*] Architecture: Sparse MoE + Shared Expert + RoPE + SwiGLU")
    print(f"[*] Total Parameters: {param_count:,} ({param_count / 1e6:.2f}M)")
    print(f"[*] Training Method: GRPO (Group Relative Policy Optimization)")
    print(f"[*] Target Device: {device}")

    dataset = ReasoningPromptDataset(vocab_size=32000, num_samples=num_steps)
    trainer = GRPOTrainer(model=model, group_size=4, lr=3e-4)

    start_time = time.time()
    step_metrics = []

    for step in range(num_steps):
        prompt, target = dataset[step]
        prompt, target = prompt.to(device), target.to(device)
        metrics = trainer.train_step_grpo(prompt, target)
        step_metrics.append(metrics)
        print(f" -> GRPO Step {step+1}/{num_steps} | Policy Loss: {metrics['loss']:.4f} | Reward: {metrics['mean_reward']:.3f}")

    elapsed = time.time() - start_time
    avg_reward = sum(m["mean_reward"] for m in step_metrics) / len(step_metrics)

    results = {
        "experiment_id": "EXP-002",
        "model_version": "AETHER-MoE-v0.2",
        "architecture": "Sparse MoE (4 experts, top-2) + Shared Expert",
        "parameters": param_count,
        "device": device,
        "training_framework": "GRPO (Group Relative Policy Optimization)",
        "steps": num_steps,
        "final_loss": round(step_metrics[-1]["loss"], 4),
        "avg_reward": round(avg_reward, 3),
        "elapsed_seconds": round(elapsed, 2),
        "status": "COMPLETED_CLEAN"
    }

    print("\n[SUCCESS] Experiment EXP-002 executed cleanly!")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment_exp002(device=device)
