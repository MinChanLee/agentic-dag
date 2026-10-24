"""Tests for the public programmatic API and file input handling."""

from __future__ import annotations

import pytest

from agentic import Executor, FileInput, PipelineResult
from agentic.models import InputNode, NodeSpec


# -----------------------------------------------------------------------
# Programmatic API – dict input
# -----------------------------------------------------------------------


class TestProgrammaticAPI:
    @pytest.mark.asyncio
    async def test_run_returns_pipeline_result(
        self, simple_specs: list[NodeSpec], sample_inputs: dict[str, str]
    ) -> None:
        executor = Executor()
        result = await executor.run_specs(simple_specs, sample_inputs)

        assert isinstance(result, PipelineResult)
        assert "A" in result
        assert "B" in result
        assert "C" in result

    @pytest.mark.asyncio
    async def test_dict_like_access(
        self, simple_specs: list[NodeSpec], sample_inputs: dict[str, str]
    ) -> None:
        """PipelineResult supports result[key] for backward compatibility."""
        executor = Executor()
        result = await executor.run_specs(simple_specs, sample_inputs)

        assert "result" in result["A"]
        assert list(result.keys())
        assert list(result.items())

    @pytest.mark.asyncio
    async def test_outputs_only_excludes_inputs(
        self, simple_specs: list[NodeSpec], sample_inputs: dict[str, str]
    ) -> None:
        executor = Executor()
        result = await executor.run_specs(simple_specs, sample_inputs)

        outputs = result.outputs_only()
        assert "input1" not in outputs
        assert "input2" not in outputs
        assert "input3" not in outputs
        assert "A" in outputs
        assert "B" in outputs
        assert "C" in outputs


# -----------------------------------------------------------------------
# Input node tracking
# -----------------------------------------------------------------------


class TestInputNodeTracking:
    @pytest.mark.asyncio
    async def test_text_inputs_tracked(
        self, simple_specs: list[NodeSpec], sample_inputs: dict[str, str]
    ) -> None:
        executor = Executor()
        result = await executor.run_specs(simple_specs, sample_inputs)

        assert "input1" in result.input_nodes
        assert result.input_nodes["input1"].kind == "text"
        assert result.input_nodes["input1"].temp_path is None

    @pytest.mark.asyncio
    async def test_all_inputs_tracked(
        self, simple_specs: list[NodeSpec], sample_inputs: dict[str, str]
    ) -> None:
        executor = Executor()
        result = await executor.run_specs(simple_specs, sample_inputs)

        for key in sample_inputs:
            assert key in result.input_nodes
            assert isinstance(result.input_nodes[key], InputNode)


# -----------------------------------------------------------------------
# FileInput handling
# -----------------------------------------------------------------------


class TestFileInput:
    @pytest.mark.asyncio
    async def test_file_input_bytes(
        self, simple_specs: list[NodeSpec]
    ) -> None:
        """FileInput with data=bytes should be materialized to temp file."""
        inputs = {
            "input1": FileInput(filename="test.txt", data=b"file content here"),
            "input2": "data for B",
            "input3": "data for C",
        }
        executor = Executor()
        result = await executor.run_specs(simple_specs, inputs)

        # input1 should have been resolved to a string path in context
        assert isinstance(result.context["input1"], str)
        assert result.input_nodes["input1"].kind == "file"
        assert result.input_nodes["input1"].temp_path is not None

    @pytest.mark.asyncio
    async def test_file_input_path(
        self, simple_specs: list[NodeSpec], tmp_path
    ) -> None:
        """FileInput with path= should copy file to temp."""
        src_file = tmp_path / "source.csv"
        src_file.write_text("col1,col2\na,b", encoding="utf-8")

        inputs = {
            "input1": FileInput(filename="source.csv", path=str(src_file)),
            "input2": "data for B",
            "input3": "data for C",
        }
        executor = Executor()
        result = await executor.run_specs(simple_specs, inputs)

        assert result.input_nodes["input1"].kind == "file"
        assert result.input_nodes["input1"].temp_path is not None

    @pytest.mark.asyncio
    async def test_cleanup_after_run(
        self, simple_specs: list[NodeSpec]
    ) -> None:
        """Temp directory should be cleaned up after run completes."""
        inputs = {
            "input1": FileInput(filename="test.txt", data=b"data"),
            "input2": "data for B",
            "input3": "data for C",
        }
        executor = Executor()
        await executor.run_specs(simple_specs, inputs)

        # After run, temp dir should be cleaned up
        assert executor._temp_dir is None

    @pytest.mark.asyncio
    async def test_cleanup_on_exception(
        self, simple_specs: list[NodeSpec]
    ) -> None:
        """Temp files should be cleaned up even if execution fails."""
        inputs = {
            "input1": FileInput(filename="test.txt", data=b"data"),
            # Missing input2 and input3 -> should raise
        }
        executor = Executor()
        with pytest.raises((ValueError, RuntimeError)):
            await executor.run_specs(simple_specs, inputs)

        assert executor._temp_dir is None

    def test_file_input_requires_path_or_data(self) -> None:
        """FileInput with neither path nor data should raise."""
        with pytest.raises(ValueError, match="either 'path' or 'data'"):
            FileInput(filename="test.txt")
