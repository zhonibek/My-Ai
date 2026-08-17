import math
import sys
import io
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from app.providers.search_provider import TavilySearchProvider, SearchResult

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: pass

    @property
    @abstractmethod
    def description(self) -> str: pass

    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]: pass

    @abstractmethod
    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]: pass


class WebSearchTool(BaseTool):
    def __init__(self):
        self.search_provider = TavilySearchProvider()

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for up-to-date information, real-time prices, news, and technical specs."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The web search query string"}
            },
            "required": ["query"]
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        query = params.get("query", "")
        results: List[SearchResult] = await self.search_provider.search(query)
        
        formatted_context = ""
        sources = []
        for idx, res in enumerate(results, 1):
            formatted_context += f"[{idx}] {res.title}\nURL: {res.url}\nSnippet: {res.snippet}\n\n"
            sources.append({"id": idx, "title": res.title, "url": res.url, "domain": res.source_domain})

        return {
            "formatted_text": formatted_context,
            "sources": sources,
            "count": len(results)
        }


class CalculatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions safely (e.g. arithmetic, trigonometry, powers)."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression like '125 * 0.85 + math.sqrt(144)'"}
            },
            "required": ["expression"]
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        expr = params.get("expression", "")
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names["math"] = math
        try:
            result = eval(expr, {"__builtins__": {}}, allowed_names)
            return {"expression": expr, "result": result, "status": "success"}
        except Exception as e:
            return {"expression": expr, "error": str(e), "status": "error"}


class CodeExecutionTool(BaseTool):
    @property
    def name(self) -> str:
        return "code_execution"

    @property
    def description(self) -> str:
        return "Execute Python code safely in isolated environment and capture output."

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code snippet to execute"}
            },
            "required": ["code"]
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        code = params.get("code", "")
        if not code:
            return {"output": "No code provided.", "status": "error"}

        # Block malicious operating system destructive calls
        forbidden = ["os.system", "shutil.rmtree", "subprocess.Popen", "__import__('os')", "eval(", "exec("]
        for f in forbidden:
            if f in code:
                return {"output": f"Security Notice: Execution of '{f}' is restricted in safe sandbox mode.", "status": "error"}

        buffer = io.StringIO()
        sys_stdout = sys.stdout
        try:
            sys.stdout = buffer
            safe_globals = {
                "math": math,
                "json": __import__("json"),
                "re": __import__("re"),
                "datetime": __import__("datetime"),
                "print": print
            }
            exec(code, safe_globals)
            output = buffer.getvalue()
            return {"output": output.strip() or "Code executed cleanly with no stdout.", "status": "success"}
        except Exception as e:
            return {"output": f"Execution error: {str(e)}", "status": "error"}
        finally:
            sys.stdout = sys_stdout


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {
            "web_search": WebSearchTool(),
            "calculator": CalculatorTool(),
            "code_execution": CodeExecutionTool()
        }

    def get_openai_tool_definitions(self) -> List[Dict[str, Any]]:
        defs = []
        for tool in self.tools.values():
            defs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema
                }
            })
        return defs

    async def execute_tool(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found", "status": "error"}
        return await self.tools[name].execute(params)

tool_registry = ToolRegistry()
