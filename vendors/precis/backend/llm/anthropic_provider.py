"""© JINAN KORDAB — 2026 AI HYBRID AGENTIC RETRIEVAL-AUGMENTED GENERATION RAG PIPELINE - PERSONAL PROJECT"""

from typing import AsyncGenerator, List, Optional
from anthropic import AsyncAnthropic
from backend.llm.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022") -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None,
                       temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> str:
        resp = await self._client.messages.create(
            model=self._model, max_tokens=max_tokens, temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}], **kwargs)
        return resp.content[0].text if resp.content else ""

    async def generate_stream(self, prompt: str, system_prompt: Optional[str] = None,
                              temperature: float = 0.7, max_tokens: int = 4096, **kwargs) -> AsyncGenerator[str, None]:
        async with self._client.messages.stream(
            model=self._model, max_tokens=max_tokens, temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}], **kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        raise NotImplementedError("Anthropic does not provide a public embeddings API")

    def get_model_name(self) -> str: return self._model

    def get_token_count(self, text: str) -> int:
        try:
            return self._client.count_tokens(text)
        except Exception:
            return len(text) // 4
