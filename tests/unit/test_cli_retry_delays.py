"""CLI tests for configuring Ollama retry delays."""

import sys
from types import SimpleNamespace

import pytest

from llm_guard_bench import cli
from llm_guard_bench.providers.adapters import GroqAdapter, OllamaAdapter


def test_get_adapter_passes_retry_delays_to_ollama() -> None:
    configured = cli.get_adapter("ollama", "m", retry_delays=(1.0, 2.0))
    defaulted = cli.get_adapter("ollama", "m")

    assert isinstance(configured, OllamaAdapter)
    assert configured.retry_delays == (1.0, 2.0)
    assert isinstance(defaulted, OllamaAdapter)
    assert defaulted.retry_delays == (10, 30, 60)


def test_get_adapter_ignores_retry_delays_for_groq() -> None:
    adapter = cli.get_adapter("groq", "m", api_key="k", retry_delays=(1.0,))

    assert isinstance(adapter, GroqAdapter)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], (10.0, 30.0, 60.0)),
        (["--retry-delays", "1,2.5"], (1.0, 2.5)),
        (["--retry-delays", ""], ()),
    ],
)
def test_parse_arguments_retry_delay_values(
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected: tuple[float, ...],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-guard-bench", "--target", "t", *extra_args],
    )

    assert cli.parse_arguments().retry_delays == expected


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "abc", "1,,2"])
def test_parse_arguments_rejects_invalid_retry_delay_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    value: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["llm-guard-bench", "--target", "t", "--retry-delays", value],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.parse_arguments()

    assert exc_info.value.code == 2
    assert "argument --retry-delays" in capsys.readouterr().err


async def test_orchestrator_passes_retry_delays_to_both_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record_adapter(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(cli, "get_adapter", record_adapter)
    monkeypatch.setenv("TARGET_PROVIDER", "ollama")
    monkeypatch.setenv("JUDGE_PROVIDER", "ollama")
    orchestrator = cli.LLMGuardBenchOrchestrator(
        target="target",
        judge="judge",
        concurrency=1,
        retry_delays=(0.5,),
    )

    await orchestrator.initialize_adapters()

    assert len(calls) == 2
    assert [kwargs["retry_delays"] for _, kwargs in calls] == [(0.5,), (0.5,)]


async def test_async_main_passes_retry_delays_to_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "parse_arguments",
        lambda: SimpleNamespace(
            target="t",
            judge="j",
            concurrency=1,
            categories=None,
            auto_flush=False,
            attacks_file=None,
            evaluation_delay=0.25,
            retry_delays=(0.5,),
        ),
    )
    monkeypatch.setattr(cli, "validate_configuration", lambda: None)

    class FakeOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            captured_kwargs.append(kwargs)

        async def orchestrate(self) -> None:
            return None

    monkeypatch.setattr(cli, "LLMGuardBenchOrchestrator", FakeOrchestrator)

    await cli.async_main()

    assert captured_kwargs[0]["retry_delays"] == (0.5,)
