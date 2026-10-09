"""Tests for safe CLI output with narrow console encodings."""

import asyncio
import io
import sys
from collections.abc import Coroutine
from typing import Any

import pytest

from llm_guard_bench import cli


def _cp1252_stream() -> tuple[io.BytesIO, io.TextIOWrapper]:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", newline="\n")
    return buffer, stream


def test_configure_console_streams_replaces_unencodable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout_buffer, stdout = _cp1252_stream()
    stderr_buffer, stderr = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    with pytest.raises(UnicodeEncodeError):
        stdout.write("✓")

    cli.configure_console_streams()

    print("✓ ✗ →")
    sys.stderr.write("✓")
    sys.stdout.flush()
    sys.stderr.flush()
    assert stdout_buffer.getvalue() == b"? ? ?\n"
    assert stderr_buffer.getvalue() == b"?"


def test_configure_console_streams_tolerates_streams_without_reconfigure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    cli.configure_console_streams()


def test_configure_console_streams_preserves_utf8_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = io.BytesIO()
    stream = io.TextIOWrapper(buffer, encoding="utf-8", errors="strict")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    cli.configure_console_streams()
    print("✓", end="")
    sys.stdout.flush()

    assert buffer.getvalue() == "✓".encode()
    assert stream.encoding.lower().replace("-", "") == "utf8"


def test_main_configures_console_before_running_async_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def configure_console_streams() -> None:
        events.append("configure")

    async def async_main() -> None:
        events.append("async_main")

    real_run = asyncio.run

    def run(coroutine: Coroutine[Any, Any, None]) -> None:
        events.append("asyncio.run")
        try:
            assert events[:1] == ["configure"]
            real_run(coroutine)
        finally:
            coroutine.close()

    monkeypatch.setattr(cli, "configure_console_streams", configure_console_streams, raising=False)
    monkeypatch.setattr(cli, "async_main", async_main)
    monkeypatch.setattr(cli.asyncio, "run", run)

    cli.main()

    assert events == ["configure", "asyncio.run", "async_main"]
