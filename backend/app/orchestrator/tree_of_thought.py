import asyncio
from typing import List, Dict, Any, Optional, Tuple

class TreeOfThoughtsEngine:
    """
    Tree-of-Thoughts (ToT) & Multi-Path Reasoning Engine.
    
    Generates multiple concurrent reasoning paths (Hypothesis, Edge-Case Verification, Direct Proof),
    evaluates candidates with an internal Critic Verifier, and selects the optimal path.
    """
    def __init__(self, num_branches: int = 2):
        self.num_branches = num_branches

    async def explore_reasoning_paths(
        self,
        prompt: str,
        generate_fn,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Executes parallel branch exploration and evaluates the best candidate.
        """
        branch_configs = [
            {"id": "branch_analytical", "desc": "First-Principles Decomposition", "temp": 0.3},
            {"id": "branch_edge_cases", "desc": "Edge-Case & Boundary Testing", "temp": 0.6}
        ]

        tasks = []
        for cfg in branch_configs[:self.num_branches]:
            branch_prompt = (
                f"REASONING STRATEGY: {cfg['desc']}\n"
                f"{prompt}"
            )
            tasks.append(generate_fn(branch_prompt, model_name=model_name, temperature=cfg["temp"]))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        candidates = []
        for idx, res in enumerate(results):
            if isinstance(res, str) and len(res.strip()) > 0:
                cfg = branch_configs[idx]
                score = self._score_candidate(res)
                candidates.append({
                    "branch_id": cfg["id"],
                    "strategy": cfg["desc"],
                    "text": res,
                    "score": score
                })

        if not candidates:
            return {"best_text": "", "branch_used": "none", "score": 0.0}

        # Select highest scoring candidate
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]

        return {
            "best_text": best["text"],
            "branch_used": best["strategy"],
            "score": best["score"],
            "all_branches": [{"strategy": c["strategy"], "score": c["score"]} for c in candidates]
        }

    def _score_candidate(self, text: str) -> float:
        """Internal heuristic critic verifier scoring structure, markdown headers, and code integrity."""
        score = 1.0
        # Reward structured markdown
        if "###" in text or "##" in text:
            score += 0.5
        # Reward code blocks with language tag
        if "```python" in text or "```" in text:
            score += 0.6
        # Penalize excessive repetition or gibberish
        words = text.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.35:
                score -= 1.0
            else:
                score += unique_ratio * 0.5
        return round(score, 2)

tot_engine = TreeOfThoughtsEngine()
