import json
import uuid
from typing import AsyncGenerator, List, Dict, Any, Optional
from app.providers.base import ChatMessage, StreamChunk
from app.providers.gateway import gateway
from app.orchestrator.router import router
from app.orchestrator.tools import tool_registry


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
2. **Code blocks**: always specify language tag — \`\`\`python, \`\`\`typescript, etc.
3. **Math formulas**: use LaTeX — `$inline$` for inline, `$$block$$` for displayed equations
4. **Be complete** — never give a half-answer. Write full, working, production-ready code
5. **Show reasoning** when solving complex problems — expose your thought process
6. **Handle edge cases** — always consider failure modes and boundary conditions in code
7. **Cite alternatives** — when multiple solutions exist, briefly compare and recommend the best

## Advanced Thinking Protocol (Chain-of-Thought)
For complex tasks (math, algorithms, architecture, logic puzzles, analysis):
1. Begin with `<think>` block — break down the problem, enumerate hypotheses, verify edge cases, explore approaches
2. Close `</think>` and write a clean, structured, complete final answer
3. The thinking block is always shown to the user — make it intellectually honest and revealing

## Output Format Rules
- `###` headers for major sections, `####` for sub-sections
- Bullet points for unordered items; numbered lists for sequential steps  
- Tables for comparisons (always include headers)
- Code blocks for ALL code, commands, file paths, and configuration
- Blockquotes (`>`) for tips, warnings, and important notes
- Bold (`**text**`) for key terms and important concepts

## Behavioral Principles
- Never claim you "cannot" do something without genuinely attempting it first
- Always provide at least one concrete, immediately actionable output
- Prefer specific, exact technical terms over vague generalizations
- Make reasonable assumptions when the request is ambiguous — state them explicitly
- Be intellectually honest: say when something is uncertain, experimental, or debated"""


class AIOrchestrator:
    """
    Core AI Orchestrator: coordinates Intent Routing → Tool Execution → RAG Retrieval → LLM Generation.
    Emits structured SSE events for real-time frontend streaming.
    """

    async def execute_chat_stream(
        self,
        messages: List[ChatMessage],
        model: str = "aether-neural-local",
        file_ids: List[str] = [],
        enable_web_search: bool = True,
        user_id: str = "default_user",
        rag_context: Optional[str] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        req_id = f"orch-{uuid.uuid4().hex[:8]}"
        last_user_message = messages[-1].content if messages else ""

        # ── Step 1: Intent Analysis & Routing ──────────────────────────────
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

        # Build system prompt — inject thinking budget for complex tasks
        system_prompt = AETHER_SYSTEM_PROMPT
        if intent.complexity == "complex" or intent.query_type == "deep_reasoning":
            system_prompt += f"\n\n> **THINKING MODE ACTIVE** (budget: {intent.thinking_budget} tokens) — Use extended `<think>` reasoning before answering."
        elif intent.query_type == "coding":
            system_prompt += "\n\n> **CODING MODE** — Provide complete, executable code with docstrings, error handling, and usage examples."
        elif intent.query_type == "math_code":
            system_prompt += "\n\n> **MATH SOLVER MODE** — Show all steps. Use LaTeX for formulas. Verify the answer."

        sources = []

        # ── Step 2: Web Search ─────────────────────────────────────────────
        if intent.requires_web_search and enable_web_search:
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

        # ── Step 3: RAG File Context ────────────────────────────────────────
        if rag_context:
            system_prompt += f"\n\n---\n### 📄 UPLOADED DOCUMENT CONTEXT\n{rag_context}\n---\n> Strictly base your answer on this document content."

        # ── Step 4: Build final message list ───────────────────────────────
        final_messages = [ChatMessage(role="system", content=system_prompt)]
        for msg in messages:
            final_messages.append(msg)

        # ── Step 5: Stream LLM Generation ──────────────────────────────────
        provider, resolved_model = gateway.resolve_provider(selected_model)

        # Dynamic temperature: lower for code/math, higher for creative
        temperature = 0.3 if intent.query_type in ("coding", "math_code") else 0.7

        async for chunk in provider.generate_stream(
            messages=final_messages,
            model=resolved_model,
            temperature=temperature
        ):
            yield chunk


orchestrator = AIOrchestrator()
