"""Tests for classifying permanent provider request errors."""

from __future__ import annotations

import aiohttp
import pytest

from llm_guard_bench.providers import adapters
from llm_guard_bench.providers.adapters import GroqAdapter, OllamaAdapter


class FakeContent:
    """Provide an HTTP response body in bounded chunks."""

    def __init__(self, body: bytes) -> None:
        self.body = body

    async def iter_chunked(self, chunk_size: int):
        for index in range(0, len(self.body), chunk_size):
            yield self.body[index : index + chunk_size]


class FakeResponse:
    """Response context manager with a chunked body."""

    def __init__(self, status: int, body: bytes = b"provider error") -> None:
        self.status = status
        self.content = FakeContent(body)

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None


def _install_ollama_response(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    class ClientSession:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> ClientSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> None:
            return None

        def post(
            self,
            _url: str,
            *,
            json: dict[str, object],
        ) -> FakeResponse:
            return FakeResponse(status)

    monkeypatch.setattr(adapters.aiohttp, "ClientSession", ClientSession)


class StatusError(Exception):
    """Exception carrying a provider status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider returned HTTP {status_code}")
        self.status_code = status_code


class FakeGroqClient:
    """Fake Groq client whose create call always raises."""

    def __init__(self, error: Exception) -> None:
        self.chat = FakeChat(error)


class FakeChat:
    def __init__(self, error: Exception) -> None:
        self.completions = FakeCompletions(error)


class FakeCompletions:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def create(self, **_kwargs: object) -> object:
        raise self.error


def _groq_adapter(error: Exception) -> GroqAdapter:
    adapter = GroqAdapter("test-model", api_key="test-key")
    adapter.client = FakeGroqClient(error)
    return adapter


@pytest.mark.parametrize("status", [400, 401, 403, 404])
@pytest.mark.parametrize("method_name", ["_post_chat", "generate"])
async def test_ollama_client_errors_are_permanent(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    method_name: str,
) -> None:
    _install_ollama_response(monkeypatch, status)
    adapter = OllamaAdapter("test-model", retry_delays=())

    with pytest.raises(adapters.PermanentProviderError, match=f"status={status}"):
        if method_name == "_post_chat":
            await adapter._post_chat({})
        else:
            await adapter.generate("system", "user")


@pytest.mark.parametrize("status", [408, 429, 500, 503])
async def test_ollama_transient_http_errors_are_not_permanent(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    _install_ollama_response(monkeypatch, status)
    adapter = OllamaAdapter("test-model", retry_delays=())

    with pytest.raises(RuntimeError, match=f"status={status}") as exc_info:
        await adapter._post_chat({})

    assert not isinstance(exc_info.value, adapters.PermanentProviderError)


async def test_ollama_connection_error_still_uses_retry_ladder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    class FailingRequest:
        async def __aenter__(self) -> object:
            raise aiohttp.ClientConnectionError("connection refused")

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> None:
            return None

    class ClientSession:
        def __init__(self, **_kwargs: object) -> None:
            nonlocal attempts
            attempts += 1

        async def __aenter__(self) -> ClientSession:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> None:
            return None

        def post(
            self,
            _url: str,
            *,
            json: dict[str, object],
        ) -> FailingRequest:
            return FailingRequest()

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(adapters.aiohttp, "ClientSession", ClientSession)
    monkeypatch.setattr(adapters.asyncio, "sleep", record_sleep)
    adapter = OllamaAdapter("test-model", retry_delays=(0.25, 0.5))

    with pytest.raises(RuntimeError, match="connection error") as exc_info:
        await adapter._post_chat({})

    assert not isinstance(exc_info.value, adapters.PermanentProviderError)
    assert attempts == 3
    assert sleeps == [0.25, 0.5]


@pytest.mark.parametrize("status", [400, 401, 403])
@pytest.mark.parametrize("method_name", ["generate", "generate_multi_turn"])
async def test_groq_client_errors_are_permanent(
    status: int,
    method_name: str,
) -> None:
    adapter = _groq_adapter(StatusError(status))

    with pytest.raises(adapters.PermanentProviderError) as exc_info:
        if method_name == "generate":
            await adapter.generate("system", "user")
        else:
            await adapter.generate_multi_turn([{"role": "user", "content": "user"}])

    if status == 401:
        assert "HTTP 401" in str(exc_info.value)


@pytest.mark.parametrize("status", [429, 500])
@pytest.mark.parametrize("method_name", ["generate", "generate_multi_turn"])
async def test_groq_transient_http_errors_are_not_permanent(
    status: int,
    method_name: str,
) -> None:
    adapter = _groq_adapter(StatusError(status))

    with pytest.raises(RuntimeError) as exc_info:
        if method_name == "generate":
            await adapter.generate("system", "user")
        else:
            await adapter.generate_multi_turn([{"role": "user", "content": "user"}])

    assert not isinstance(exc_info.value, adapters.PermanentProviderError)


async def test_groq_error_without_status_code_is_not_permanent() -> None:
    adapter = _groq_adapter(RuntimeError("provider failure"))

    with pytest.raises(RuntimeError) as exc_info:
        await adapter.generate("system", "user")

    assert not isinstance(exc_info.value, adapters.PermanentProviderError)
