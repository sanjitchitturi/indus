"""Tests for MLflow tracking utilities."""

from __future__ import annotations

from indus.tracking.mlflow_logger import _flatten_config


class TestFlattenConfig:
    def test_flat_dict(self) -> None:
        result = _flatten_config({"a": 1, "b": "hello"})
        assert result == {"a": 1, "b": "hello"}

    def test_nested_dict(self) -> None:
        result = _flatten_config({"model": {"name": "mistral", "size": 7}})
        assert result == {"model.name": "mistral", "model.size": 7}

    def test_deeply_nested(self) -> None:
        result = _flatten_config({"a": {"b": {"c": 42}}})
        assert result == {"a.b.c": 42}

    def test_list_values(self) -> None:
        result = _flatten_config({"targets": ["q_proj", "v_proj"]})
        assert result == {"targets": "['q_proj', 'v_proj']"}

    def test_empty_dict(self) -> None:
        result = _flatten_config({})
        assert result == {}
