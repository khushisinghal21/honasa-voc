"""Thin Claude wrapper: forced tool-use for structured output, validation retries, and token/cost tracking."""
from __future__ import annotations

import threading
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from voc.config import settings

T = TypeVar("T", bound=BaseModel)


class Usage:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens = {"fast": [0, 0], "smart": [0, 0]}  # [input, output]
        self.seconds = 0.0

    def add(self, tier: str, inp: int, out: int, secs: float) -> None:
        with self._lock:
            self.calls += 1
            self.tokens[tier][0] += inp
            self.tokens[tier][1] += out
            self.seconds += secs

    def cost_usd(self) -> float:
        f_in, f_out = self.tokens["fast"]
        s_in, s_out = self.tokens["smart"]
        return (
            f_in * settings.price_fast_in + f_out * settings.price_fast_out
            + s_in * settings.price_smart_in + s_out * settings.price_smart_out
        ) / 1_000_000

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "tokens": self.tokens,
            "llm_seconds": round(self.seconds, 1),
            "est_cost_usd": round(self.cost_usd(), 4),
        }


usage = Usage()


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Add it to .env, or run with VOC_LLM_BACKEND=baseline."
            )
        import anthropic

        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=4)

    def structured(
        self,
        schema: type[T],
        prompt: str,
        system: str = "",
        tier: str = "fast",
        max_tokens: int = 4096,
        attempts: int = 3,
    ) -> T:
        """Force the model to call a single tool whose input schema is `schema`, then validate it."""
        model = settings.fast_model if tier == "fast" else settings.smart_model
        tool = {
            "name": "emit",
            "description": f"Return the result as a {schema.__name__} object.",
            "input_schema": schema.model_json_schema(),
        }
        messages = [{"role": "user", "content": prompt}]
        last_err: Exception | None = None
        for _ in range(attempts):
            t0 = time.time()
            resp = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system or "You are a precise information-extraction system.",
                tools=[tool],
                tool_choice={"type": "tool", "name": "emit"},
                messages=messages,
            )
            usage.add(tier, resp.usage.input_tokens, resp.usage.output_tokens, time.time() - t0)
            block = next((b for b in resp.content if getattr(b, "type", "") == "tool_use"), None)
            if block is None:
                last_err = LLMError("model did not call the tool")
                continue
            try:
                return schema.model_validate(block.input)
            except ValidationError as e:
                last_err = e
                # Show the model its own mistake and ask again.
                messages = messages + [
                    {"role": "assistant", "content": resp.content},
                    {"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                        "content": f"Validation failed, fix and call emit again:\n{e}",
                    }]},
                ]
        raise LLMError(f"structured output failed after {attempts} attempts: {last_err}")


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm


def use_llm() -> bool:
    return settings.llm_backend == "anthropic"
