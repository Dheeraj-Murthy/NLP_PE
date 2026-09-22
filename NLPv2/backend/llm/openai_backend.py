import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from llm.base import GenerationResult, LLMBackend

DEFAULT_OPENAI_MODEL = "gpt-4o"


class OpenAIBackend(LLMBackend):
    requires_external_ok = True

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_MODEL,
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
    ):
        load_dotenv()
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=key)
        self._model_id = model_id
        self.max_new_tokens = max_new_tokens

    def generate(
        self,
        context_block: str,
        user_query: str,
        system_prompt: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
    ) -> GenerationResult:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(
            {"role": m["role"], "content": m["content"]} for m in (conversation_history or [])
        )
        messages.append(
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion:\n{user_query}"}
        )

        resp = self.client.chat.completions.create(
            model=self._model_id,
            max_tokens=max_new_tokens or self.max_new_tokens,
            messages=messages,
        )

        choice = resp.choices[0]

        return GenerationResult(
            text=(choice.message.content or "").strip(),
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            model_id=self._model_id,
            model_version=resp.model,
            raw={"finish_reason": choice.finish_reason},
        )

    def count_tokens(self, text: str) -> int:
        # No local tokenizer bundled for OpenAI models; word-count heuristic
        # is good enough for context-budget estimation (not billing-accurate).
        return int(len(text.split()) * 1.3)

    @property
    def model_id(self) -> str:
        return self._model_id
