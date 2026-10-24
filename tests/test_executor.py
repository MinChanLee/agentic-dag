"""Tests for the DAG executor."""

from __future__ import annotations

import pytest

from agentic.models import NodeSpec
from agentic.runtime.executor import Executor


# -----------------------------------------------------------------------
# Basic execution
# -----------------------------------------------------------------------

class TestSimpleExecution:
    @pytest.mark.asyncio
    async def test_all_outputs_produced(
        self, simple_specs: list[NodeSpec], sample_inputs: dict[str, str]
    ) -> None:
        executor = Executor()
        result = await executor.run_specs(simple_specs, sample_inputs)

        assert "A" in result
        assert "B" in result
        assert "C" in result
        # Each output is validated against GenericOutput (``result: str``)
        for key in ("A", "B", "C"):
            assert "result" in result[key]


# -----------------------------------------------------------------------
# Steps: C unchanged when A gains internal steps
# -----------------------------------------------------------------------

class TestStepsTransparency:
    @pytest.mark.asyncio
    async def test_c_spec_unchanged_with_stepped_a(
        self,
        simple_specs: list[NodeSpec],
        stepped_specs: list[NodeSpec],
        sample_inputs: dict[str, str],
    ) -> None:
        """C node spec is identical in both pipelines – execution must
        succeed in both cases without modifying C."""
        # Extract C spec from both – they must be the same object values
        c_simple = next(s for s in simple_specs if s.id == "C")
        c_stepped = next(s for s in stepped_specs if s.id == "C")
        assert c_simple.deps == c_stepped.deps
        assert c_simple.operator == c_stepped.operator

        executor = Executor()

        result_simple = await executor.run_specs(simple_specs, sample_inputs)
        result_stepped = await executor.run_specs(stepped_specs, sample_inputs)

        assert "C" in result_simple
        assert "C" in result_stepped


# -----------------------------------------------------------------------
# Missing deps
# -----------------------------------------------------------------------

class TestMissingDeps:
    @pytest.mark.asyncio
    async def test_missing_input_raises(
        self, simple_specs: list[NodeSpec]
    ) -> None:
        executor = Executor()
        with pytest.raises((ValueError, RuntimeError)):
            await executor.run_specs(simple_specs, {"input1": "a"})
