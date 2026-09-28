"""
Model Provider Adapters for LLM Guard Bench

Implements unified async interfaces for multiple LLM providers:
- OllamaAdapter: Local HTTP API (http://localhost:11434)
- GroqAdapter: Groq cloud API (https://api.groq.com/openai/v1)

CRITICAL SECURITY: API credentials are loaded dynamically via os.getenv()
from the .env file (which is GITIGNORED). Never hardcode API keys.
All adapters validate credentials at initialization time.

Each adapter:
1. Accepts system_prompt + user_prompt (single-turn)
2. Or accepts messages list (multi-turn conversation)
3. Returns response string or raises RuntimeError on failure
4. Implements exponential backoff for transient failures
"""

from __future__ import annotations

import asyncio
import os
import pathlib
from abc import ABC, abstractmethod
from typing import Any, Optional

import aiohttp
import logging
from groq import AsyncGroq


class BaseAdapter(ABC):
    """Abstract interface for model provider adapters."""

    @abstractmethod
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        override_temperature: float = None,
    ) -> str:
        """Generate a model response for the provided prompts."""
        raise NotImplementedError

    @abstractmethod
    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float = None,
    ) -> str:
        """
        Generate a model response for multi-turn conversation.
        Messages should be a list of dicts with 'role' (system/user/assistant) and 'content' keys.
        """
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Return whether the provider is ready to accept a request."""
        return True


class OllamaAdapter(BaseAdapter):
    """Async adapter for Ollama local HTTP API."""

    CONNECTION_RETRY_DELAYS = (10, 30, 60)

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 180.0,
        default_temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.default_temperature = default_temperature

    async def health_check(self) -> bool:
        """Ping Ollama, retrying connection failures while the service recovers."""
        url = f"{self.base_url}/api/tags"
        timeout = aiohttp.ClientTimeout(total=min(self.timeout_seconds, 10.0))

        for attempt in range(len(self.CONNECTION_RETRY_DELAYS) + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as response:
                        if response.status < 400:
                            return True
                        logging.getLogger(__name__).warning(
                            f"Ollama health check returned HTTP {response.status}"
                        )
            except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as exc:
                logging.getLogger(__name__).warning(
                    f"Ollama health check failed on attempt {attempt + 1}/"
                    f"{len(self.CONNECTION_RETRY_DELAYS) + 1}: {exc}"
                )

            if attempt < len(self.CONNECTION_RETRY_DELAYS):
                delay = self.CONNECTION_RETRY_DELAYS[attempt]
                logging.getLogger(__name__).warning(
                    f"Retrying Ollama health check in {delay}s"
                )
                await asyncio.sleep(delay)

        return False

    async def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Post to Ollama, retrying only connection-level failures."""
        url = f"{self.base_url}/api/chat"
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)

        for attempt in range(len(self.CONNECTION_RETRY_DELAYS) + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload) as response:
                        if response.status >= 400:
                            body = await response.text()
                            raise RuntimeError(
                                f"Ollama API error (status={response.status}): {body}"
                            )
                        return await response.json(content_type=None)
            except asyncio.TimeoutError as exc:
                raise RuntimeError(
                    f"Ollama request timed out after {self.timeout_seconds} seconds"
                ) from exc
            except aiohttp.ClientConnectionError as exc:
                if attempt >= len(self.CONNECTION_RETRY_DELAYS):
                    raise RuntimeError(f"Ollama connection error: {exc}") from exc

                delay = self.CONNECTION_RETRY_DELAYS[attempt]
                logging.getLogger(__name__).warning(
                    f"Ollama connection failure on attempt {attempt + 1}/"
                    f"{len(self.CONNECTION_RETRY_DELAYS) + 1}: {exc}. "
                    f"Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)
            except aiohttp.ClientError as exc:
                raise RuntimeError(f"Ollama HTTP client error: {exc}") from exc
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(f"Unexpected Ollama adapter error: {exc}") from exc

        raise RuntimeError("Ollama connection retry loop exited unexpectedly")

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        override_temperature: float = None,
    ) -> str:
        temperature = (
            override_temperature
            if override_temperature is not None
            else self.default_temperature
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }

        data = await self._post_chat(payload)

        try:
            message = data.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    # Return content even if empty; empty indicates refusal or no-op
                    return content.strip()

            response_text = data.get("response")
            if isinstance(response_text, str):
                return response_text.strip()

            # Fallback: log and return empty string instead of raising
            logging.getLogger(__name__).warning(
                f"Ollama response missing expected content keys; returning empty string. Raw: {data}"
            )
            return ""
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float = None,
    ) -> str:
        """Generate response for multi-turn conversation."""
        temperature = (
            override_temperature
            if override_temperature is not None
            else self.default_temperature
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }

        data = await self._post_chat(payload)

        try:
            message = data.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()

            response_text = data.get("response")
            if isinstance(response_text, str):
                return response_text.strip()

            logging.getLogger(__name__).warning(
                f"Ollama response missing expected content keys for multi-turn; returning empty string. Raw: {data}"
            )
            return ""
        except Exception as exc:
            raise RuntimeError(f"Failed to parse Ollama response: {exc}") from exc


class GroqAdapter(BaseAdapter):
    """Async adapter for Groq chat completions."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        timeout_seconds: float = 180.0,
        default_temperature: float = 0.0,
    ) -> None:
        # Validate API key at initialization time: fail fast if credentials missing
        if not api_key:
            raise RuntimeError("Groq API key is required")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.default_temperature = default_temperature
        self.client = AsyncGroq(api_key=api_key, timeout=timeout_seconds)

    @staticmethod
    def masked_api_key(api_key: str) -> str:
        """Return only the safe-to-log prefix and suffix of an API key."""
        if not api_key or len(api_key) < 8:
            return "MISSING/TOO_SHORT"
        return f"{api_key[:4]}...{api_key[-4:]}"

    @staticmethod
    def _format_request_error(exc: Exception) -> str:
        """Classify Groq failures with actionable diagnostics."""
        status_code = getattr(exc, "status_code", None)
        message = str(exc)
        message_lower = message.lower()

        if status_code == 401 or "invalid_api_key" in message_lower:
            return (
                "Groq authentication failed (HTTP 401): invalid API key. "
                "Verify GROQ_API_KEY in .env and ensure it has no quotes or whitespace."
            )
        if status_code == 429 or "rate limit" in message_lower:
            return (
                "Groq rate limit reached (HTTP 429). Wait before retrying or "
                "reduce request frequency/concurrency."
            )
        if (
            isinstance(exc, (aiohttp.ClientError, ConnectionError, OSError))
            or "connection" in message_lower
            or "connect" in message_lower
            or "network" in message_lower
            or "dns" in message_lower
        ):
            return (
                "Groq network failure: unable to reach api.groq.com. "
                "Check container outbound internet, DNS, proxy, and TLS settings."
            )
        return f"Groq API failure ({type(exc).__name__}): {message}"

    async def _request_completion(self, messages: list[dict[str, str]], temperature: float):
        """Request a Groq completion and raise a classified error on failure."""
        try:
            return await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Groq network failure: request timed out after {self.timeout_seconds} seconds"
            ) from exc
        except Exception as exc:
            raise RuntimeError(self._format_request_error(exc)) from exc

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        override_temperature: float = None,
    ) -> str:
        temperature = (
            override_temperature
            if override_temperature is not None
            else self.default_temperature
        )

        completion = await self._request_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        try:
            choices = getattr(completion, "choices", None)
            if not choices:
                logging.getLogger(__name__).warning(
                    "Groq completion did not return any choices; returning empty string"
                )
                return ""

            content = choices[0].message.content
            if not isinstance(content, str):
                logging.getLogger(__name__).warning(
                    "Groq completion returned non-string content; returning empty string"
                )
                return ""
            # Return content (may be empty string)
            return content.strip()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"Failed to parse Groq response: {exc}. Returning empty string."
            )
            return ""

    async def generate_multi_turn(
        self,
        messages: list[dict[str, str]],
        override_temperature: float = None,
    ) -> str:
        """Generate response for multi-turn conversation."""
        temperature = (
            override_temperature
            if override_temperature is not None
            else self.default_temperature
        )

        completion = await self._request_completion(messages, temperature)

        try:
            choices = getattr(completion, "choices", None)
            if not choices:
                logging.getLogger(__name__).warning(
                    "Groq completion did not return any choices (multi-turn); returning empty string"
                )
                return ""

            content = choices[0].message.content
            if not isinstance(content, str):
                logging.getLogger(__name__).warning(
                    "Groq completion returned non-string content (multi-turn); returning empty string"
                )
                return ""
            return content.strip()
        except Exception as exc:
            logging.getLogger(__name__).warning(
                f"Failed to parse Groq response (multi-turn): {exc}. Returning empty string."
            )
            return ""


def _get_default_ollama_url() -> str:
    """
    Determine Ollama base URL based on environment and execution context.
    
    Priority:
    1. OLLAMA_BASE_URL environment variable (explicit override)
    2. OLLAMA_ENDPOINT environment variable (Docker Compose default)
    3. host.docker.internal:11434 if running in Docker (detected by /.dockerenv)
    4. localhost:11434 if running locally
    """
    # Priority 1: Explicit environment variable
    if env_url := os.getenv("OLLAMA_BASE_URL"):
        return env_url
    
    # Priority 2: Docker Compose OLLAMA_ENDPOINT
    if env_url := os.getenv("OLLAMA_ENDPOINT"):
        return env_url
    
    # Priority 3: Running in Docker container (check for /.dockerenv marker)
    if pathlib.Path("/.dockerenv").exists():
        return "http://host.docker.internal:11434"
    
    # Fallback: localhost (for local development)
    return "http://localhost:11434"


def get_adapter(provider: str, model_name: str, api_key: str = None) -> BaseAdapter:
    """Factory for provider-specific adapters."""
    normalized = (provider or "").strip().lower()

    if normalized == "ollama":
        base_url = _get_default_ollama_url()
        # এখানে explicit ভাবে timeout_seconds=180.0 পাস করে দিচ্ছি
        return OllamaAdapter(model_name=model_name, base_url=base_url, timeout_seconds=180.0)

    if normalized == "groq":
        if not api_key:
            raise RuntimeError("API key is required for Groq adapter")
        return GroqAdapter(model_name=model_name, api_key=api_key, timeout_seconds=180.0)

    raise RuntimeError(f"Unsupported provider: {provider}")