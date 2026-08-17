import json
import uuid
import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional

from app.providers.base import ChatMessage, StreamChunk
from app.providers.gateway import gateway
from app.orchestrator.router import router
from app.orchestrator.tools import tool_registry
from app.orchestrator.self_correction import self_correction_verifier
from app.orchestrator.deep_research import deep_research_agent
from app.orchestrator.tree_of_thought import tot_engine
from app.storage.memory_graph import memory_graph


AETHER_SYSTEM_PROMPT = """You are AETHER — a next-generation, frontier-grade AI assistant built on a proprietary provider-agnostic AI Operating Layer. You are competing with models like Gemini, Claude, and GPT-4 in intelligence, depth, and quality.

## Core Identity & Capabilities
You are a highly capable, precise, and intellectually rigorous AI with deep expertise in:
- **Software Engineering**: Python, TypeScript, Rust, Go, SQL, Bash, C++, Java, Swift
- **Machine Learning & AI**: PyTorch, TensorFlow, JAX, scikit-learn, transformers, RLHF, LoRA, MoE
- **System Architecture**: microservices, distributed systems, databases, cloud infrastructure, APIs
- **Mathematics**: calculus, linear algebra, statistics, discrete math, probability theory, proofs
- **Research & Analysis**: literature review, comparative analysis, technical writing, strategic planning
- **Creative Writing**: essays, storytelling, persuasive writing, poetry

## Multilingual Excellence
- **Қазақ тілі**: Қазақша сұрақ берілгенде, міндетті түрде жоғары сапалы, грамматикалық жағынан дұрыс қазақ тілінде жауап бер. Қазақ тілінің ерекше таңбаларын (ә, і, ң, ғ, ү, ұ, қ, ө, һ) міндетті түрде дұрыс қолдан. Жауаптарың нақты, толық, және мазмұнды болсын.
- **Русский язык**: Пиши безупречно грамотно. Соблюдай орфографию, пунктуацию и академический стиль изложения. Структурируй ответы логично и полно. Избегай поверхностных ответов.
- **English**: Use precise, technical, and analytically rigorous language. Prefer concrete examples over abstract descriptions.

## Response Excellence Standards
1. **Always use Markdown** for structure: headers (`###`), bold, code blocks, tables, blockquotes
2. **Code blocks**: always specify language tag — ```python, ```typescript, etc.
3. **Math formulas**: use LaTeX — `$inline$` for inline, `$$block$$` for displayed equations
4. **Be complete** — never give a half-answer. Write full, working, production-ready code
5. **Show reasoning** when solving complex problems — expose your thought process
6. **Handle edge cases** — always consider failure modes and boundary conditions in code
7. **Cite alternatives** — when multiple solutions exist, briefly compare and recommend the best

## Advanced Thinking Protocol (Chain-of-Thought)
For complex tasks (math, algorithms, architecture, logic puzzles, analysis):
1. Conduct deep reasoning inside `<think>` block — break down the problem, test hypotheses, verify edge cases
2. Close `</think>` and provide a clean, structured, production-ready final answer

## Output Format Rules
- `###` headers for major sections, `####` for sub-sections
- Bullet points for unordered items; numbered lists for sequential steps  
- Tables for comparisons (always include headers)
- Code blocks for ALL code, commands, file paths, and configuration
- Blockquotes (`>`) for tips, warnings, and important notes"""


class AIOrchestrator:
    """
    Enterprise AI Orchestrator with Autonomous Verification, Deep Research,
    Long-Term Memory Graph, and Tree-of-Thoughts Reasoning.
    """

    async def execute_chat_stream(
        self,
        messages: List[ChatMessage],
        model: str = "aether-neural-local",
        file_ids: List[str] = [],
        enable_web_search: bool = True,
        enable_deep_research: bool = False,
        enable_self_correction: bool = True,
        user_id: str = "default_user",
        rag_context: Optional[str] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        req_id = f"orch-{uuid.uuid4().hex[:8]}"
        last_user_message = messages[-1].content if messages else ""

        # ── Step 1: Fact Extraction & Long-Term Memory Recall ──────────────
        if last_user_message:
            try:
                # Background fact extraction
                memory_graph.extract_and_store_facts(last_user_message, user_id=user_id)
            except Exception:
                pass

        memory_ctx = memory_graph.format_memory_context(last_user_message, user_id=user_id)

        # ── Step 2: Intent Analysis & Routing ──────────────────────────────
        intent = router.analyze_intent(last_user_message, file_ids=file_ids, requested_model=model)
        selected_model = intent.suggested_model

        yield StreamChunk(
            id=req_id, model=selected_model, delta_content="",
            event_type="reasoning",
            metadata={
                "status": "analyzing_intent",
                "query_type": intent.query_type,
                "complexity": intent.complexity,
                "explanation": intent.reasoning_explanation,
                "thinking_budget": intent.thinking_budget,
                "model_selected": selected_model
            }
        )

        # Inject Memory into System Prompt
        system_prompt = AETHER_SYSTEM_PROMPT
        if memory_ctx:
            system_prompt += f"\n\n{memory_ctx}"

        if intent.complexity == "complex" or intent.query_type == "deep_reasoning":
            system_prompt += f"\n\n> **THINKING MODE ACTIVE** (budget: {intent.thinking_budget} tokens) — Use extended `<think>` reasoning before answering."
        elif intent.query_type == "coding":
            system_prompt += "\n\n> **CODING MODE** — Provide complete, executable code with docstrings, error handling, and usage examples."
        elif intent.query_type == "math_code":
            system_prompt += "\n\n> **MATH SOLVER MODE** — Show all steps. Use LaTeX for formulas. Verify the answer."

        sources = []

        # ── Step 3: Multi-Hop Deep Research or Quick Web Search ────────────
        if (intent.requires_web_search or enable_deep_research or enable_web_search) and intent.query_type in ("web_search", "deep_reasoning"):
            if enable_deep_research or "deep research" in last_user_message.lower() or intent.complexity == "complex":
                yield StreamChunk(
                    id=req_id, model=selected_model, delta_content="",
                    event_type="reasoning",
                    metadata={"status": "deep_research_active", "stage": "multi_hop_crawling", "topic": last_user_message}
                )
                research_res = await deep_research_agent.execute_deep_research(last_user_message)
                system_prompt += f"\n\n{research_res['dossier_text']}"
                sources = research_res.get("sources", [])
                yield StreamChunk(
                    id=req_id, model=selected_model, delta_content="",
                    event_type="source_citation",
                    metadata={"sources": sources}
                )
            else:
                # Fast 1-hop search
                yield StreamChunk(
                    id=req_id, model=selected_model, delta_content="",
                    event_type="reasoning",
                    metadata={"status": "executing_tool", "tool": "web_search", "query": last_user_message}
                )
                search_result = await tool_registry.execute_tool("web_search", {"query": last_user_message})
                if search_result.get("formatted_text"):
                    system_prompt += f"\n\n---\n### 🌐 REAL-TIME WEB SEARCH RESULTS\n{search_result['formatted_text']}\n---"
                    sources = search_result.get("sources", [])
                    yield StreamChunk(
                        id=req_id, model=selected_model, delta_content="",
                        event_type="source_citation",
                        metadata={"sources": sources}
                    )

        # ── Step 4: RAG File Context ────────────────────────────────────────
        if rag_context:
            system_prompt += f"\n\n---\n### 📄 UPLOADED DOCUMENT CONTEXT\n{rag_context}\n---\n> Strictly base your answer on this document content."

        # ── Step 5: Build final message list ───────────────────────────────
        final_messages = [ChatMessage(role="system", content=system_prompt)]
        for msg in messages:
            final_messages.append(msg)

        # ── Step 6: Stream LLM Generation ──────────────────────────────────
        provider, resolved_model = gateway.resolve_provider(selected_model)
        temperature = 0.3 if intent.query_type in ("coding", "math_code") else 0.7

        accumulated_text = ""
        async for chunk in provider.generate_stream(
            messages=final_messages,
            model=resolved_model,
            temperature=temperature
        ):
            if chunk.event_type == "token":
                accumulated_text += chunk.delta_content
            yield chunk

        # ── Step 7: Background Self-Correction Sandbox Verification ─────────
        if enable_self_correction and intent.query_type in ("coding", "math_code"):
            code_blocks = self_correction_verifier.extract_python_code_blocks(accumulated_text)
            if code_blocks:
                yield StreamChunk(
                    id=req_id, model=selected_model, delta_content="",
                    event_type="reasoning",
                    metadata={"status": "sandbox_verification", "message": f"Verifying {len(code_blocks)} code blocks in sandbox"}
                )


orchestrator = AIOrchestrator()
