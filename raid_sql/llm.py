"""Chat LLM clients: OpenAI and Vertex Gemini, with retries and usage telemetry."""

from __future__ import annotations

import os
import random
import time
from typing import Any, Optional, Protocol

from config import Settings, get_settings
from raid_sql.metrics import StageMetrics, cost_usd, estimate_tokens


def _is_retryable(exc: BaseException) -> bool:
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    needles = (
        "429",
        "resource_exhausted",
        "resource exhausted",
        "rate",
        "quota",
        "unavailable",
        "timeout",
        "503",
        "500",
        "temporarily",
        "overloaded",
        # DNS / transient network (Errno 8 on macOS, etc.)
        "nodename nor servname",
        "errno 8",
        "failed to resolve",
        "name or service not known",
        "getaddrinfo",
        "connection reset",
        "connection refused",
        "connection aborted",
        "network is unreachable",
        "temporary failure in name resolution",
        "remote end closed",
        "broken pipe",
        "ssl",
        "timed out",
    )
    return any(n in msg for n in needles) or any(
        n in name for n in ("timeout", "unavailable", "ratelimit", "connectionerror", "oserror")
    )


def _is_network_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = (
        "nodename nor servname",
        "errno 8",
        "failed to resolve",
        "name or service not known",
        "getaddrinfo",
        "connection reset",
        "connection refused",
        "network is unreachable",
        "temporary failure in name resolution",
    )
    return any(n in msg for n in needles)


class ChatClient(Protocol):
    model: str

    def generate(
        self,
        prompt: str,
        *,
        stage: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        est_latency_ms: float = 1500.0,
    ) -> tuple[str, StageMetrics]: ...


def _build_gemini_client(settings: Settings):
    from google import genai

    if settings.gemini_api_key:
        return genai.Client(api_key=settings.gemini_api_key)

    if settings.google_application_credentials:
        cred_path = settings.google_application_credentials
        if os.path.exists(cred_path):
            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", cred_path)

    if not settings.google_cloud_project:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT (+ credentials) in .env"
        )

    return genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )


def _gemini_usage_tokens(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return 0, 0
    inp = int(
        getattr(usage, "prompt_token_count", None)
        or getattr(usage, "input_token_count", None)
        or 0
    )
    out = int(
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "output_token_count", None)
        or 0
    )
    return inp, out


def _gemini_response_text(response: Any) -> str:
    text = getattr(response, "text", None) or ""
    if text:
        return text
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    parts = getattr(candidates[0].content, "parts", None) or []
    return "".join(getattr(p, "text", "") or "" for p in parts)


class GeminiClient:
    """Vertex / Gemini chat client."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._client = _build_gemini_client(self.settings)
        self.model = self.settings.gemini_model

    def generate(
        self,
        prompt: str,
        *,
        stage: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        est_latency_ms: float = 1500.0,
    ) -> tuple[str, StageMetrics]:
        from google.genai import types

        temp = self.settings.temperature if temperature is None else temperature
        max_tok = (
            self.settings.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        in_rate, out_rate = self.settings.rates_for_model(self.model)
        est_in = estimate_tokens(prompt)
        est_out = max_tok
        metrics = StageMetrics(
            stage=stage,
            model=self.model,
            prompt_chars=len(prompt),
            est_input_tokens=est_in,
            est_output_tokens=est_out,
            est_cost_usd=cost_usd(est_in, est_out, in_rate, out_rate),
            est_latency_ms=est_latency_ms,
        )

        config_kwargs: dict[str, Any] = {
            "temperature": temp,
            "max_output_tokens": max_tok,
        }
        if stop_sequences:
            config_kwargs["stop_sequences"] = stop_sequences
        try:
            budget = 256 if "flash" in self.model.lower() else 512
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget
            )
        except Exception:  # noqa: BLE001
            pass

        last_exc: Optional[BaseException] = None
        attempts = max(1, self.settings.llm_max_retries)
        t0 = time.perf_counter()
        for attempt in range(attempts):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                text = _gemini_response_text(response)
                inp, out = _gemini_usage_tokens(response)
                if inp <= 0:
                    inp = est_in
                if out <= 0:
                    out = estimate_tokens(text)
                metrics.input_tokens = inp
                metrics.output_tokens = out
                metrics.cost_usd = cost_usd(inp, out, in_rate, out_rate)
                metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
                metrics.retries = attempt
                metrics.ok = True
                return text, metrics
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                metrics.retries = attempt + 1
                if attempt + 1 >= attempts or not _is_retryable(exc):
                    metrics.ok = False
                    metrics.error = str(exc)[:500]
                    metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
                    # Failed calls should not inflate spend / look "paid"
                    metrics.input_tokens = 0
                    metrics.output_tokens = 0
                    metrics.cost_usd = 0.0
                    raise
                if _is_network_error(exc):
                    try:
                        self._client = _build_gemini_client(self.settings)
                    except Exception:  # noqa: BLE001
                        pass
                sleep_s = self.settings.llm_retry_base_sec * (2**attempt) + random.uniform(
                    0, 1.0
                )
                if _is_network_error(exc):
                    sleep_s = max(sleep_s, 5.0)
                sleep_s = min(sleep_s, 90.0)
                print(
                    f"  [retry {attempt + 1}/{attempts} model={self.model} stage={stage}] "
                    f"{exc} — sleep {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)

        assert last_exc is not None
        raise last_exc


class OpenAIClient:
    """OpenAI Chat Completions client (e.g. gpt-4o)."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        from openai import OpenAI

        self.settings = settings or get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("Set OPENAI_API_KEY in .env")
        kwargs: dict[str, Any] = {"api_key": self.settings.openai_api_key}
        if self.settings.openai_base_url:
            kwargs["base_url"] = self.settings.openai_base_url
        self._client = OpenAI(**kwargs)
        self.model = self.settings.openai_model

    def generate(
        self,
        prompt: str,
        *,
        stage: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        est_latency_ms: float = 1500.0,
    ) -> tuple[str, StageMetrics]:
        temp = self.settings.temperature if temperature is None else temperature
        max_tok = (
            self.settings.max_output_tokens
            if max_output_tokens is None
            else max_output_tokens
        )
        in_rate, out_rate = self.settings.rates_for_model(self.model)
        est_in = estimate_tokens(prompt)
        est_out = max_tok
        metrics = StageMetrics(
            stage=stage,
            model=self.model,
            prompt_chars=len(prompt),
            est_input_tokens=est_in,
            est_output_tokens=est_out,
            est_cost_usd=cost_usd(est_in, est_out, in_rate, out_rate),
            est_latency_ms=est_latency_ms,
        )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": max_tok,
        }
        if stop_sequences:
            kwargs["stop"] = stop_sequences

        last_exc: Optional[BaseException] = None
        attempts = max(1, self.settings.llm_max_retries)
        t0 = time.perf_counter()
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0] if response.choices else None
                text = (choice.message.content if choice and choice.message else None) or ""
                usage = getattr(response, "usage", None)
                inp = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
                out = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
                if inp <= 0:
                    inp = est_in
                if out <= 0:
                    out = estimate_tokens(text)
                metrics.input_tokens = inp
                metrics.output_tokens = out
                metrics.cost_usd = cost_usd(inp, out, in_rate, out_rate)
                metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
                metrics.retries = attempt
                metrics.ok = True
                return text, metrics
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                metrics.retries = attempt + 1
                if attempt + 1 >= attempts or not _is_retryable(exc):
                    metrics.ok = False
                    metrics.error = str(exc)[:500]
                    metrics.latency_ms = (time.perf_counter() - t0) * 1000.0
                    metrics.input_tokens = 0
                    metrics.output_tokens = 0
                    metrics.cost_usd = 0.0
                    raise
                if _is_network_error(exc):
                    try:
                        from openai import OpenAI

                        kwargs_c: dict[str, Any] = {"api_key": self.settings.openai_api_key}
                        if self.settings.openai_base_url:
                            kwargs_c["base_url"] = self.settings.openai_base_url
                        self._client = OpenAI(**kwargs_c)
                    except Exception:  # noqa: BLE001
                        pass
                sleep_s = self.settings.llm_retry_base_sec * (2**attempt) + random.uniform(
                    0, 1.0
                )
                if _is_network_error(exc):
                    sleep_s = max(sleep_s, 5.0)
                sleep_s = min(sleep_s, 90.0)
                print(
                    f"  [retry {attempt + 1}/{attempts} model={self.model} stage={stage}] "
                    f"{exc} — sleep {sleep_s:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_s)

        assert last_exc is not None
        raise last_exc


def create_chat_client(settings: Optional[Settings] = None) -> ChatClient:
    """Factory: OpenAI or Gemini based on CHAT_PROVIDER."""
    settings = settings or get_settings()
    provider = (settings.chat_provider or "openai").lower().strip()
    if provider in {"openai", "gpt", "gpt-4o"}:
        return OpenAIClient(settings)
    if provider in {"gemini", "vertex", "google"}:
        return GeminiClient(settings)
    raise ValueError(f"Unknown CHAT_PROVIDER={settings.chat_provider!r}")
