import os
from typing import Any, Dict, List, Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from llm.base import GenerationResult, LLMBackend

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"


class AnthropicBackend(LLMBackend):
    requires_external_ok = True

    def __init__(
        self,
        model_id: str = DEFAULT_ANTHROPIC_MODEL,
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
    ):
        # Called explicitly rather than relying on the transitive load via
        # citation_graph.py's import — this module can be imported/tested
        # standalone, so it shouldn't depend on that import-order accident.
        load_dotenv()
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.client = Anthropic(api_key=key)
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
        messages = [
            {"role": m["role"], "content": m["content"]} for m in (conversation_history or [])
        ]
        messages.append(
            {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion:\n{user_query}"}
        )

        resp = self.client.messages.create(
            model=self._model_id,
            max_tokens=max_new_tokens or self.max_new_tokens,
            system=system_prompt,
            messages=messages,
        )

        text = "".join(block.text for block in resp.content if block.type == "text")

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
            model_id=self._model_id,
            model_version=resp.model,
            raw={"stop_reason": resp.stop_reason},
        )

    def count_tokens(self, text: str) -> int:
        # No local tokenizer for Anthropic models; word-count heuristic is
        # good enough for context-budget estimation (not billing-accurate).
        return int(len(text.split()) * 1.3)

    @property
    def model_id(self) -> str:
        return self._model_id
