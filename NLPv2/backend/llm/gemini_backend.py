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

        resp = self.client.models.generate_content(
            model=self._model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_new_tokens or self.max_new_tokens,
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


def list_available_models(api_key: str) -> List[str]:
    """Models this key can actually use. Raises google.genai.errors.ClientError
    (e.g. API_KEY_INVALID) if the key itself is bad.

    ListModels is a static catalog, not a per-key access list — it still lists
    models like gemini-2.5-pro as generateContent-capable even when a given
    key gets "this model is no longer available to new users" on the real
    call. So each catalog candidate is live-probed; only models that don't
    403/404 for this specific key are kept.

    The catalog also isn't scoped to the "Gemini" product line — a key with
    broad access sees generateContent-capable entries from other Google model
    families sharing the same endpoint (e.g. deep-research-*, antigravity-*).
    Those aren't routable: rag_pipeline._resolve_backend dispatches purely on
    an id starting with "gemini-" and raises "Unknown model" for anything
    else, so a key that could technically reach them would still fail here
    with an unrelated-looking error. Only "gemini-"-prefixed ids are kept.

    The probe uses countTokens rather than generateContent — it hits the same
    per-model access check (a restricted/deprecated model 404s there too) but
    does no generation, so it's cheaper, faster, and not subject to
    generation-quota rate limits, which matters when probing dozens of
    candidates per click.
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
        }
    )

    def _is_usable(name: str) -> bool:
        try:
            client.models.count_tokens(
                model=name,
                contents=[types.Content(role="user", parts=[types.Part(text="hi")])],
            )
            return True
        except genai_errors.ClientError as e:
            # 403/404 = this key genuinely can't use the model. Anything else
            # (e.g. 429 rate limit) means it's reachable, just throttled right
            # now — don't punish it for that.
            return e.code not in (403, 404)
        except Exception:
            # Timeout or other transient failure — inconclusive, not a
            # confirmed denial, so don't drop a model over it.
            return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        usable = list(pool.map(_is_usable, candidates))

    return [name for name, ok in zip(candidates, usable) if ok]
