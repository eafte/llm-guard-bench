"""Tests for bounded Ollama HTTP response handling."""

from __future__ import annotations

import pytest

from llm_guard_bench.providers import adapters
from llm_guard_bench.providers.adapters import OllamaAdapter


class FakeContent:
    """Async chunk source that tracks how many response bytes were consumed."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.consumed = 0

    async def iter_chunked(self, _size: int):
        for chunk in self.chunks:
            self.consumed += len(chunk)
            yield chunk


class FakeResponse:
    """HTTP response fake with chunked content and no text or JSON helpers."""

    def __init__(self, status: int, body: bytes, chunk_size: int = 50) -> None:
        self.status = status
        self.content = FakeContent(
            [body[index : index + chunk_size] for index in range(0, len(body), chunk_size)]
        )

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        return None


def _install_response(
    monkeypatch: pytest.MonkeyPatch,
    response: FakeResponse,
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
            return response

    monkeypatch.setattr(adapters.aiohttp, "ClientSession", ClientSession)


async def test_small_success_body_is_parsed_from_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = b'{"message": {"content": "hi"}}'
    response = FakeResponse(status=200, body=body)
    _install_response(monkeypatch, response)
    adapter = OllamaAdapter("test-model")

    result = await adapter._post_chat({})

    assert result == {"message": {"content": "hi"}}
    assert response.content.consumed == len(body)


async def test_success_body_over_limit_stops_after_one_extra_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(status=200, body=b"x" * 500, chunk_size=50)
    _install_response(monkeypatch, response)
    adapter = OllamaAdapter("test-model", max_response_bytes=100)

    with pytest.raises(RuntimeError, match="response too large"):
        await adapter._post_chat({})

    assert response.content.consumed <= 150


async def test_large_error_body_is_bounded_and_preserves_http_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(status=500, body=b"e" * 50_000, chunk_size=50)
    _install_response(monkeypatch, response)
    adapter = OllamaAdapter("test-model", max_response_bytes=100)

    with pytest.raises(RuntimeError, match="status=500") as exc_info:
        await adapter._post_chat({})

    assert len(str(exc_info.value)) <= 2_500
    assert response.content.consumed <= 150


async def test_invalid_success_json_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeResponse(status=200, body=b"not-json")
    _install_response(monkeypatch, response)
    adapter = OllamaAdapter("test-model")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        await adapter._post_chat({})


def test_default_response_limit_is_eight_megabytes() -> None:
    adapter = OllamaAdapter("test-model")

    assert adapter.max_response_bytes == 8 * 1024 * 1024


@pytest.mark.parametrize("invalid_limit", [0, -1, True, False, 1.0, "100"])
def test_invalid_response_limit_is_rejected(invalid_limit: object) -> None:
    with pytest.raises(ValueError):
        OllamaAdapter("test-model", max_response_bytes=invalid_limit)
