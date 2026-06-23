"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import json
from typing import AsyncGenerator, List, Optional
import httpx
from backend.llm.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3") -> None:
        self._base = base_url.rstrip("/")
        self._model = model

    async def _post(self, endpoint: str, data: dict) -> dict:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self._base}/api/{endpoint}", json=data)
            return resp.json() if resp.status_code == 200 else {}

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        data = {"model": self._model, "prompt": prompt, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if system_prompt:
            data["system"] = system_prompt
        result = await self._post("generate", data)
        return result.get("response", "")

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> AsyncGenerator[str, None]:
        data = {"model": self._model, "prompt": prompt, "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if system_prompt:
            data["system"] = system_prompt
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", f"{self._base}/api/generate", json=data) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if chunk.get("response"):
                                yield chunk["response"]
                        except json.JSONDecodeError:
                            pass

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = await self._post("embeddings", {"model": model or "nomic-embed-text", "prompt": text})
            embeddings.append(result.get("embedding", [0.0]))
        return embeddings

    def get_model_name(self) -> str: return self._model

    def get_token_count(self, text: str) -> int: return len(text) // 4
