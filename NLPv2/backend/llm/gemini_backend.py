import concurrent.futures
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from llm.base import GenerationResult, LLMBackend

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"


class GeminiBackend(LLMBackend):
    requires_external_ok = True

    def __init__(
        self,
        model_id: str = DEFAULT_GEMINI_MODEL,
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
    ):
        load_dotenv()
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=key)
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
        # Gemini's `contents` API uses role "model" for prior assistant turns,
        # not "assistant" — everything else in our ChatSession history uses
        # the OpenAI/Anthropic-style "assistant", so translate it here.
        contents = []
        for m in conversation_history or []:
            role = "model" if m["role"] == "assistant" else m["role"]
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=f"Context:\n{context_block}\n\nQuestion:\n{user_query}")],
            )
        )

        # rag_pipeline's configured max_new_tokens (512 by default) is sized
        # for local Qwen inference, where bigger generations cost real VRAM.
        # "Thinking" Gemini models (e.g. the gemini-3-* family) spend part of
        # that same max_output_tokens budget on an internal reasoning pass
        # before the visible answer, so a 512-token cap can get eaten by
        # thinking and cut the answer off mid-sentence. Cloud API calls have
        # no VRAM constraint, so just floor the budget high enough to leave
        # room for both.
        requested_tokens = max_new_tokens or self.max_new_tokens
        resp = self.client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max(requested_tokens, 2048),
            ),
        )

        usage = resp.usage_metadata

        return GenerationResult(
            text=(resp.text or "").strip(),
            prompt_tokens=usage.prompt_token_count if usage else 0,
            completion_tokens=usage.candidates_token_count if usage else 0,
            model_id=self._model_id,
            model_version=getattr(resp, "model_version", None) or self._model_id,
        )

    def count_tokens(self, text: str) -> int:
        # No local tokenizer bundled for Gemini models; word-count heuristic
        # is good enough for context-budget estimation (not billing-accurate).
        return int(len(text.split()) * 1.3)

    @property
    def model_id(self) -> str:
        return self._model_id


PROBE_TIMEOUT_MS = 10_000  # per-call cap — the SDK's default is no timeout at
# all (an unreachable/hanging model would block its worker thread forever)

# Catalog entries that are "gemini-"-prefixed and generateContent-capable but
# are not plain text chat models — TTS/image-output/agentic-control variants
# that either can't serve our text RAG answers at all, or reject our call
# shape outright (e.g. TTS models 400 with "Multiturn chat is not enabled").
# Filtered by name first (cheaper, catches cases the live probe below can't
# always distinguish from a transient error) as well as by the probe itself.
UNSUITABLE_NAME_MARKERS = (
    "-tts", "-image", "computer-use", "embedding", "-aqa", "-live", "-audio",
)


def list_available_models(api_key: str) -> List[str]:
    """Models this key can actually use for a plain multi-turn text chat
    call — i.e. what GeminiBackend.generate() actually sends. Raises
    google.genai.errors.ClientError (e.g. API_KEY_INVALID) if the key itself
    is bad.

    ListModels is a static catalog, not a per-key/per-call-shape access list:
    - It still lists models like gemini-2.5-pro as generateContent-capable
      even when a given key gets "this model is no longer available to new
      users" on the real call (403/404).
    - It doesn't reflect per-model free-tier quota — some models (e.g.
      computer-use-preview, flash-image) report a 429 with an explicit
      "limit: 0" for the free tier, meaning they're not just rate-limited,
      they're categorically unusable on this plan.
    - It doesn't reflect call-shape support — TTS models 400 with "Multiturn
      chat is not enabled" because they're not conversational text models,
      regardless of key/quota.
    - It also isn't scoped to the "Gemini" product line — a key with broad
      access sees generateContent-capable entries from other Google model
      families sharing the same endpoint (e.g. deep-research-*,
      antigravity-*). rag_pipeline._resolve_backend only routes ids starting
      with "gemini-", so those are excluded by prefix below.

    So each surviving candidate is live-probed with the real call shape
    (generate_content, 1 output token) rather than the cheaper countTokens,
    since only that call hits the free-tier-quota and call-shape checks above.
    """
    client = genai.Client(
        api_key=api_key, http_options=types.HttpOptions(timeout=PROBE_TIMEOUT_MS)
    )
    candidates = sorted(
        {
            m.name.split("/")[-1]
            for m in client.models.list()
            if m.name
            and m.name.split("/")[-1].startswith("gemini-")
            and "generateContent" in (m.supported_actions or [])
            and not any(
                marker in m.name.split("/")[-1] for marker in UNSUITABLE_NAME_MARKERS
            )
        }
    )

    def _is_usable(name: str) -> bool:
        try:
            client.models.generate_content(
                model=name,
                contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
                config=types.GenerateContentConfig(max_output_tokens=1),
            )
            return True
        except genai_errors.ClientError as e:
            if e.code in (403, 404):
                return False  # key genuinely can't use this model
            if e.code == 400:
                return False  # model rejects this call shape (e.g. TTS-only)
            if e.code == 429 and "limit: 0" in (e.message or ""):
                return False  # zero free-tier quota — not just throttled
            # Anything else (e.g. a 429 that's just transient throttling with
            # a nonzero limit) means it's reachable — don't punish that.
            return True
        except Exception:
            # Timeout or other transient failure — inconclusive, not a
            # confirmed denial, so don't drop a model over it.
            return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        usable = list(pool.map(_is_usable, candidates))

    return [name for name, ok in zip(candidates, usable) if ok]
