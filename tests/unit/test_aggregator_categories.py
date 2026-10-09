"""Tests for loading configured attack categories from package resources."""

from __future__ import annotations

import io
import json
import logging
from importlib.resources import files

import pytest

from llm_guard_bench.reporting import aggregator as aggregator_module
from llm_guard_bench.reporting.aggregator import ResultsAggregator


class _FakeResource:
    def __init__(self, content: str | None, *, missing: bool = False) -> None:
        self.content = content
        self.missing = missing

    def joinpath(self, *_parts: str) -> _FakeResource:
        return self

    def __truediv__(self, _part: str) -> _FakeResource:
        return self

    def read_text(self, *, encoding: str) -> str:
        if self.missing:
            raise FileNotFoundError("missing prompts resource")
        return self.content or ""

    def open(self, *_args: object, **_kwargs: object) -> io.StringIO:
        if self.missing:
            raise FileNotFoundError("missing prompts resource")
        return io.StringIO(self.content or "")


def test_load_configured_categories_reads_packaged_prompts_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    resource_path = files("llm_guard_bench").joinpath("resources", "prompts.json")
    prompts = json.loads(resource_path.read_text(encoding="utf-8"))
    expected = sorted(
        {attack["category"] for attack in prompts["attacks"] if attack.get("category")}
    )
    aggregator = ResultsAggregator(object())
    caplog.set_level(logging.WARNING)

    categories = aggregator._load_configured_categories()

    assert categories
    assert categories == expected
    assert categories == sorted(set(categories))
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]


def test_load_configured_categories_warns_once_when_resource_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resource = _FakeResource(None, missing=True)
    monkeypatch.setattr(aggregator_module, "files", lambda _package: resource, raising=False)
    aggregator = ResultsAggregator(object())
    caplog.set_level(logging.WARNING)

    assert aggregator._load_configured_categories() == []
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "missing prompts resource" in warnings[0].getMessage()


def test_load_configured_categories_warns_once_for_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    resource = _FakeResource("{invalid JSON")
    monkeypatch.setattr(aggregator_module, "files", lambda _package: resource, raising=False)
    aggregator = ResultsAggregator(object())
    caplog.set_level(logging.WARNING)

    assert aggregator._load_configured_categories() == []
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Expecting" in warnings[0].getMessage()
    assert "No such file" not in warnings[0].getMessage()
