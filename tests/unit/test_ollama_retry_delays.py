"""Tests for configurable Ollama connection retry delays."""

import math

import aiohttp
import pytest

from llm_guard_bench.providers import adapters
from llm_guard_bench.providers.adapters import OllamaAdapter


class FailingRequestContext:
    """Request context manager raising the configured connection error."""

    def __init__(self, error: aiohttp.ClientConnectionError) -> None:
        self.error = error

    async def __aenter__(self) -> object:
        raise self.error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None


def _install_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
    error: aiohttp.ClientConnectionError,
) -> list[None]:
    attempts: list[None] = []

    class ClientSession:
        def __init__(self, **_kwargs: object) -> None:
            attempts.append(None)

        async def __aenter__(self) -> object:
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc_value: BaseException | None,
            traceback: object,
        ) -> None:
            return None

        def get(self, _url: str) -> FailingRequestContext:
            return FailingRequestContext(error)

        def post(
            self,
            _url: str,
            *,
            json: dict[str, object],
        ) -> FailingRequestContext:
            return FailingRequestContext(error)

    monkeypatch.setattr(adapters.aiohttp, "ClientSession", ClientSession)
    return attempts


def _record_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> list[float]:
    sleeps: list[float] = []

    async def record_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(adapters.asyncio, "sleep", record_sleep)
    return sleeps


async def test_default_delays_retry_health_check_without_network_or_sleep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _install_connection_failure(
        monkeypatch,
        aiohttp.ClientConnectionError("connection refused"),
    )
    sleeps = _record_sleeps(monkeypatch)
    adapter = OllamaAdapter("test-model")

    assert await adapter.health_check() is False
    assert attempts == [None, None, None, None]
    assert sleeps == [10, 30, 60]


async def test_empty_retry_delays_make_one_health_check_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _install_connection_failure(
        monkeypatch,
        aiohttp.ClientConnectionError("connection refused"),
    )
    sleeps = _record_sleeps(monkeypatch)
    adapter = OllamaAdapter("test-model", retry_delays=())

    assert await adapter.health_check() is False
    assert attempts == [None]
    assert sleeps == []


async def test_custom_retry_delays_control_attempt_count_and_sleep_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _install_connection_failure(
        monkeypatch,
        aiohttp.ClientConnectionError("connection refused"),
    )
    sleeps = _record_sleeps(monkeypatch)
    adapter = OllamaAdapter("test-model", retry_delays=(0.5, 1.0))

    assert await adapter.health_check() is False
    assert attempts == [None, None, None]
    assert sleeps == [0.5, 1.0]


@pytest.mark.parametrize("delay", [-0.1, math.nan, math.inf, -math.inf])
def test_invalid_retry_delay_is_rejected_at_construction(delay: float) -> None:
    with pytest.raises(ValueError):
        OllamaAdapter("test-model", retry_delays=(delay,))


async def test_generate_uses_configured_delay_for_connection_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = _install_connection_failure(
        monkeypatch,
        aiohttp.ClientConnectionError("connection refused"),
    )
    sleeps = _record_sleeps(monkeypatch)
    adapter = OllamaAdapter("test-model", retry_delays=(0.5,))

    with pytest.raises(RuntimeError, match="connection error"):
        await adapter.generate("system", "user")

    assert attempts == [None, None]
    assert sleeps == [0.5]
