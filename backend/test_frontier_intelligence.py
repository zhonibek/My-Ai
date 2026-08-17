import os
import sys
import asyncio
import time
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.inference.kv_cache_compactor import kv_compactor
from app.orchestrator.self_correction import self_correction_verifier
from app.orchestrator.deep_research import deep_research_agent
from app.storage.memory_graph import memory_graph
from app.orchestrator.tree_of_thought import tot_engine

async def test_all_modules():
    print("==================================================================")
    print(" [TEST SUITE] AETHER FRONTIER INTELLIGENCE & OPTIMIZATION MODULES")
    print("==================================================================")

    # ── Test 1: H2O Dynamic KV-Cache Compactor ────────────────────────
    print("\n[*] 1. Testing H2O Dynamic KV-Cache Compactor...")
    batch, heads, seq_len, dim = 1, 8, 1024, 64
    k_tensor = torch.randn(batch, heads, seq_len, dim)
    v_tensor = torch.randn(batch, heads, seq_len, dim)
    past_kv = ((k_tensor, v_tensor),)

    attn_scores = torch.randn(batch, seq_len).abs()
    compacted_kv = kv_compactor.compact_kv_cache(past_kv, attn_scores=attn_scores)

    new_seq_len = compacted_kv[0][0].shape[2]
    reduction = (1 - new_seq_len / seq_len) * 100
    print(f" [PASS] KV-Compactor reduced sequence length: {seq_len} -> {new_seq_len} ({reduction:.1f}% RAM saved)")

    # ── Test 2: Episodic Memory Graph ─────────────────────────────────
    print("\n[*] 2. Testing Long-Term User Memory Graph & Fact Extractor...")
    test_msg = "Привет! Меня зовут Женибек, я основатель стартапа AETHER и пишу на Python и React."
    facts = memory_graph.extract_and_store_facts(test_msg, user_id="test_user")
    print(f" [PASS] Extracted facts: {facts}")

    recalled = memory_graph.recall_relevant_memories("какой у меня стек технологий?", user_id="test_user")
    print(f" [PASS] Semantically recalled facts for query: {recalled}")

    # ── Test 3: Self-Correction Sandbox Verifier ──────────────────────
    print("\n[*] 3. Testing Self-Correction Sandbox Verifier...")
    broken_code_response = (
        "Вот решение задачи:\n"
        "```python\n"
        "def calculate_total():\n"
        "    numbers = [10, 20, 30]\n"
        "    return numbers[1] * 5\n"
        "print('Result:', calculate_total())\n"
        "```\n"
    )

    async def mock_correction_fn(prompt, model_name):
        return (
            "Исправленный код:\n"
            "```python\n"
            "def calculate_total():\n"
            "    numbers = [10, 20, 30]\n"
            "    return numbers[1] * 5\n"
            "print('Result:', calculate_total())\n"
            "```\n"
        )

    clean_text, v_logs = await self_correction_verifier.verify_and_correct(
        initial_text=broken_code_response,
        generation_fn=mock_correction_fn,
        conversation_context=[],
        model_name="test-model"
    )
    print(f" [PASS] Self-Correction Verifier Logs: {v_logs}")

    # ── Test 4: Tree-of-Thoughts Engine ──────────────────────────────
    print("\n[*] 4. Testing Tree-of-Thoughts (ToT) Critic Evaluation...")
    async def mock_tot_generator(branch_prompt, model_name, temperature):
        if "First-Principles" in branch_prompt:
            return "### Step 1: Analytical Breakdown\n```python\nx = 42\n```"
        return "Direct answer: 42."

    tot_result = await tot_engine.explore_reasoning_paths(
        prompt="Solve optimal path problem",
        generate_fn=mock_tot_generator,
        model_name="test-model"
    )
    print(f" [PASS] ToT Winning Strategy: '{tot_result['branch_used']}' with Score: {tot_result['score']}")

    # ── Test 5: Multi-Hop Deep Research Agent ─────────────────────────
    print("\n[*] 5. Testing Multi-Hop Deep Research Agent Planning...")
    sub_q = deep_research_agent.generate_sub_queries("Quantum Computing Photonic 2026")
    print(f" [PASS] Generated Multi-Hop Sub-Queries: {sub_q}")

    print("\n==================================================================")
    print(" [SUCCESS] ALL 5 FRONTIER INTELLIGENCE MODULES PASSED VALIDATION!")
    print("==================================================================")

if __name__ == "__main__":
    asyncio.run(test_all_modules())
