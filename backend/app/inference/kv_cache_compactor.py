import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List

class HeavyHitterKVCompactor:
    """
    H2O (Heavy-Hitter Oracle) Dynamic KV-Cache Compactor.
    
    Mathematical Principle:
    In Transformer Attention: Attention(Q, K, V) = Softmax(Q K^T / sqrt(d)) V
    Only a sparse subset of tokens (Heavy Hitters) contribute to >85% of total attention mass.
    
    This module retains:
    1. Attention Sinks (Initial prompt tokens: first k_init)
    2. Local Sliding Window (Recent tokens: last k_recent)
    3. Heavy Hitters (Top-k cumulative attention score tokens: k_hh)
    
    Prunes the rest to reduce RAM/VRAM footprint by 60-80% during deep reasoning chains.
    """
    def __init__(
        self,
        k_init: int = 32,      # Initial attention sink tokens
        k_recent: int = 128,   # Sliding window recent tokens
        k_hh: int = 256,       # Heavy-hitter persistent tokens
        budget: int = 512      # Maximum allowed KV cache size
    ):
        self.k_init = k_init
        self.k_recent = k_recent
        self.k_hh = k_hh
        self.budget = budget

    def calculate_cumulative_attention(self, attn_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative importance score for every token position.
        attn_weights shape: (batch, num_heads, seq_len, seq_len)
        Returns: (batch, seq_len) importance scores.
        """
        # Sum attention received by token j from all queries i: score_j = sum_i(A_ij)
        scores = attn_weights.sum(dim=(1, 2))  # (batch, seq_len)
        return scores

    def compact_kv_cache(
        self,
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        attn_scores: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Compacts the tuple of (key, value) tensors across all transformer layers.
        Key/Value shape: (batch, num_heads, seq_len, head_dim)
        """
        if not past_key_values or len(past_key_values) == 0:
            return past_key_values

        first_k = past_key_values[0][0]
        seq_len = first_k.shape[2]

        if seq_len <= self.budget:
            return past_key_values

        batch_size = first_k.shape[0]
        device = first_k.device

        # 1. Attention sink indices [0 ... k_init-1]
        sink_indices = torch.arange(0, min(self.k_init, seq_len), device=device)

        # 2. Local recent window indices [seq_len - k_recent ... seq_len - 1]
        recent_start = max(self.k_init, seq_len - self.k_recent)
        recent_indices = torch.arange(recent_start, seq_len, device=device)

        # 3. Middle candidate tokens for Heavy Hitter selection
        middle_indices = torch.arange(self.k_init, recent_start, device=device)

        if len(middle_indices) == 0:
            return past_key_values

        if attn_scores is not None and attn_scores.shape[1] == seq_len:
            middle_scores = attn_scores[:, middle_indices]
            # Select top k_hh tokens
            k_select = min(self.k_hh, len(middle_indices))
            _, top_middle_idx = torch.topk(middle_scores, k=k_select, dim=-1)
            hh_indices = middle_indices[top_middle_idx[0]]
        else:
            # Uniform strided fallback if attention scores aren't available
            step = max(1, len(middle_indices) // max(1, self.k_hh))
            hh_indices = middle_indices[::step][:self.k_hh]

        # Combine, sort and deduplicate kept indices
        all_indices = torch.cat([sink_indices, hh_indices, recent_indices]).unique()
        all_indices, _ = torch.sort(all_indices)

        # Prune each layer's Key and Value tensors along sequence dimension
        compacted_layers = []
        for k_layer, v_layer in past_key_values:
            # (batch, num_heads, seq_len, head_dim) -> select along dim=2
            new_k = torch.index_select(k_layer, dim=2, index=all_indices)
            new_v = torch.index_select(v_layer, dim=2, index=all_indices)
            compacted_layers.append((new_k, new_v))

        return tuple(compacted_layers)

kv_compactor = HeavyHitterKVCompactor()
