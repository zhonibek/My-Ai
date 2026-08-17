import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class IntentAnalysis(BaseModel):
    query_type: str   # 'general', 'web_search', 'rag_file', 'math_code', 'deep_reasoning', 'coding'
    requires_web_search: bool
    requires_file_rag: bool
    suggested_model: str
    reasoning_explanation: str
    complexity: str   # 'simple', 'medium', 'complex'
    thinking_budget: int  # Estimated tokens for chain-of-thought


class ModelRouter:
    """
    Advanced Intent Classifier and Model Router.
    Analyzes user intent complexity, required tools, and language to select 
    the optimal model pipeline and reasoning budget.
    """

    # === Web search signal keywords ===
    WEB_KEYWORDS = {
        "ru": ["цена", "новости", "сегодня", "текущий", "найди", "поиск", "сравни", "курс", "погода", "кто такой", "когда выйдет"],
        "kz": ["баға", "жаңалық", "бүгін", "ағымдағы", "тап", "іздеу", "салыстыр"],
        "en": ["latest", "price", "news", "today", "current", "find", "search", "compare", "weather", "who is", "when will"]
    }
    WEB_PATTERNS = [r"https?://", r"\b202[4-9]\b", r"\b20[3-9]\d\b", r"\.com\b", r"\.org\b"]

    # === File / RAG signal keywords ===
    RAG_KEYWORDS = ["file", "document", "pdf", "docx", "uploaded", "report", "attachment",
                    "файл", "документ", "отчет", "приложение", "загружен", "файлда"]

    # === Math / Code signal keywords ===
    CODE_KEYWORDS = ["python", "javascript", "typescript", "rust", "go", "sql", "bash",
                     "script", "function", "algorithm", "class", "def ", "import ",
                     "код", "функция", "алгоритм", "скрипт", "программ"]

    MATH_KEYWORDS = ["calculate", "solve", "equation", "integral", "derivative", "matrix",
                     "probability", "statistics", "formula", "proof",
                     "вычисли", "реши", "уравнение", "интеграл", "матрица", "докажи",
                     "есептеу", "шешу", "теңдеу"]

    # === Deep reasoning signal keywords ===
    REASON_KEYWORDS = ["why", "explain", "analyze", "compare", "evaluate", "design",
                       "architecture", "strategy", "think", "reasoning", "step by step",
                       "почему", "объясни", "проанализируй", "сравни", "оцени", "спроектируй",
                       "шаг за шагом", "подробно", "разбери",
                       "неге", "түсіндір", "талда", "жобала"]

    @staticmethod
    def _detect_language(prompt: str) -> str:
        """Quick language detection based on script/character frequency."""
        cyrillic_count = len(re.findall(r'[а-яА-ЯёЁ]', prompt))
        kazakh_count = len(re.findall(r'[әіңғүұқөһӘІҢҒҮҰҚӨҺ]', prompt))
        if kazakh_count >= 2:
            return "kz"
        if cyrillic_count > len(prompt) * 0.2:
            return "ru"
        return "en"

    @classmethod
    def analyze_intent(
        cls,
        user_prompt: str,
        file_ids: List[str] = [],
        requested_model: str = "auto"
    ) -> IntentAnalysis:
        prompt_lower = user_prompt.lower()
        word_count = len(user_prompt.split())
        lang = cls._detect_language(user_prompt)

        # --- Feature detection ---
        # Web search need
        needs_search = (
            any(kw in prompt_lower for kws in cls.WEB_KEYWORDS.values() for kw in kws) or
            any(re.search(p, prompt_lower) for p in cls.WEB_PATTERNS)
        )

        # File/RAG need
        needs_rag = len(file_ids) > 0 or any(kw in prompt_lower for kw in cls.RAG_KEYWORDS)

        # Code need
        needs_code = any(kw in prompt_lower for kw in cls.CODE_KEYWORDS)

        # Math need
        needs_math = any(kw in prompt_lower for kw in cls.MATH_KEYWORDS)

        # Deep reasoning need
        needs_reasoning = any(kw in prompt_lower for kw in cls.REASON_KEYWORDS)

        # --- Complexity scoring ---
        complexity_score = 0
        if word_count > 50:
            complexity_score += 2
        elif word_count > 20:
            complexity_score += 1
        if needs_reasoning:
            complexity_score += 2
        if needs_code or needs_math:
            complexity_score += 1
        if needs_rag:
            complexity_score += 1

        if complexity_score >= 4:
            complexity = "complex"
            thinking_budget = 512
        elif complexity_score >= 2:
            complexity = "medium"
            thinking_budget = 256
        else:
            complexity = "simple"
            thinking_budget = 64

        # --- Query type classification ---
        if needs_search:
            query_type = "web_search"
            explanation = f"[{lang.upper()}] Real-time web data requested — activating search pipeline."
        elif needs_rag:
            query_type = "rag_file"
            explanation = f"[{lang.upper()}] Document/file context required — activating semantic RAG retrieval."
        elif needs_code:
            query_type = "coding"
            explanation = f"[{lang.upper()}] Code generation task detected — optimizing for programming output."
        elif needs_math:
            query_type = "math_code"
            explanation = f"[{lang.upper()}] Mathematical calculation/proof requested — activating step-by-step solver."
        elif needs_reasoning or complexity == "complex":
            query_type = "deep_reasoning"
            explanation = f"[{lang.upper()}] Complex reasoning task — activating chain-of-thought thinking mode."
        else:
            query_type = "general"
            explanation = f"[{lang.upper()}] Conversational query — standard response pipeline."

        # --- Model selection ---
        selected_model = requested_model if requested_model != "auto" else "aether-neural-local"

        return IntentAnalysis(
            query_type=query_type,
            requires_web_search=needs_search,
            requires_file_rag=needs_rag,
            suggested_model=selected_model,
            reasoning_explanation=explanation,
            complexity=complexity,
            thinking_budget=thinking_budget
        )


router = ModelRouter()
