"""Startup validation tests for the CLI entry points."""

from types import SimpleNamespace

import pytest

from llm_guard_bench import cli


def _install_startup_stubs(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    *,
    validation_error: ValueError | None = None,
) -> None:
    monkeypatch.setattr(
        cli,
        "parse_arguments",
        lambda: SimpleNamespace(
            target="target-model",
            judge="judge-model",
            concurrency=1,
            categories=None,
            auto_flush=False,
            attacks_file=None,
        ),
    )

    class FakeOrchestrator:
        def __init__(self, **kwargs: object) -> None:
            events.append("construct")

        async def orchestrate(self) -> None:
            events.append("orchestrate")

    def validate_configuration_spy() -> None:
        events.append("validate")
        if validation_error is not None:
            raise validation_error

    monkeypatch.setattr(cli, "LLMGuardBenchOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(cli, "validate_configuration", validate_configuration_spy, raising=False)


async def test_configuration_is_validated_once_before_anything_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_startup_stubs(monkeypatch, events)

    await cli.async_main()

    assert events == ["validate", "construct", "orchestrate"]


async def test_invalid_configuration_stops_before_orchestrator_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    _install_startup_stubs(
        monkeypatch,
        events,
        validation_error=ValueError("bad config"),
    )

    with pytest.raises(SystemExit) as exc_info:
        await cli.async_main()

    assert exc_info.value.code == 2
    assert "construct" not in events


def test_main_exits_2_and_reports_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []
    _install_startup_stubs(
        monkeypatch,
        events,
        validation_error=ValueError("bad config"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "bad config" in captured.err
