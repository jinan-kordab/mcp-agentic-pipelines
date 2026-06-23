"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

import asyncio
from typing import AsyncGenerator, List, Optional
import google.generativeai as genai
from backend.llm.base import LLMProvider


class GoogleProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro") -> None:
        genai.configure(api_key=api_key)
        self._model_name = model
        self._model = genai.GenerativeModel(self._model_name)

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        model = genai.GenerativeModel(self._model_name, system_instruction=system_prompt or None)
        resp = await asyncio.to_thread(model.generate_content, prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
        return resp.text or ""

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> AsyncGenerator[str, None]:
        model = genai.GenerativeModel(self._model_name, system_instruction=system_prompt or None)
        resp = await asyncio.to_thread(model.generate_content, prompt, stream=True,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
        for chunk in resp:
            if chunk.text:
                yield chunk.text

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        result = await asyncio.to_thread(genai.embed_content,
            model=model or "models/text-embedding-004", content=texts)
        return result.get("embedding", [[0.0]])

    def get_model_name(self) -> str: return self._model_name

    def get_token_count(self, text: str) -> int:
        return self._model.count_tokens(text).total_tokens
