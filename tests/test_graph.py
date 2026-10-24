"""Tests for DAG construction, topo sort, and layer computation."""

from __future__ import annotations

import pytest

from agentic.models import NodeSpec, NodeStep
from agentic.runtime.graph import (
    compute_layers,
    flatten_nodes,
    infer_input_keys,
    build_dot,
)


INPUT_KEYS = {"input1", "input2", "input3"}


# -----------------------------------------------------------------------
# flatten_nodes
# -----------------------------------------------------------------------

class TestFlattenNodes:
    def test_simple_nodes_pass_through(self, simple_specs: list[NodeSpec]) -> None:
        flat = flatten_nodes(simple_specs)
        assert len(flat) == 3
        ids = {n.id for n in flat}
        assert ids == {"A", "B", "C"}

    def test_composite_node_expanded(self, stepped_specs: list[NodeSpec]) -> None:
        flat = flatten_nodes(stepped_specs)
        assert len(flat) == 4  # A_prime, A_final, B, C
        ids = {n.id for n in flat}
        assert ids == {"A_prime", "A_final", "B", "C"}

    def test_duplicate_produces_raises(self) -> None:
        specs = [
            NodeSpec(id="X", produces="dup", deps=["input1"],
                     operator="GenericPromptOperator", prompt_template="p.md"),
            NodeSpec(id="Y", produces="dup", deps=["input2"],
                     operator="GenericPromptOperator", prompt_template="p.md"),
        ]
        with pytest.raises(ValueError, match="Duplicate produces"):
            flatten_nodes(specs)


# -----------------------------------------------------------------------
# compute_layers
# -----------------------------------------------------------------------

class TestComputeLayers:
    def test_simple_two_layers(self, simple_specs: list[NodeSpec]) -> None:
        flat = flatten_nodes(simple_specs)
        layers = compute_layers(flat, INPUT_KEYS)

        assert len(layers) == 2
        # Layer 0: A and B (independent)
        assert set(layers[0]) == {"A", "B"}
        # Layer 1: C (depends on A)
        assert layers[1] == ["C"]

    def test_stepped_three_layers(self, stepped_specs: list[NodeSpec]) -> None:
        flat = flatten_nodes(stepped_specs)
        layers = compute_layers(flat, INPUT_KEYS)

        assert len(layers) == 3
        # Layer 0: A_prime and B (independent)
        assert set(layers[0]) == {"A_prime", "B"}
        # Layer 1: A_final
        assert layers[1] == ["A_final"]
        # Layer 2: C
        assert layers[2] == ["C"]

    def test_missing_dep_raises(self) -> None:
        specs = [
            NodeSpec(id="X", produces="X", deps=["nonexistent"],
                     operator="GenericPromptOperator", prompt_template="p.md"),
        ]
        flat = flatten_nodes(specs)
        with pytest.raises(ValueError, match="nonexistent"):
            compute_layers(flat, set())

    def test_cycle_detection(self) -> None:
        """Two nodes depending on each other -> cycle."""
        specs = [
            NodeSpec(id="X", produces="x_out", deps=["y_out"],
                     operator="GenericPromptOperator", prompt_template="p.md"),
            NodeSpec(id="Y", produces="y_out", deps=["x_out"],
                     operator="GenericPromptOperator", prompt_template="p.md"),
        ]
        flat = flatten_nodes(specs)
        with pytest.raises(ValueError, match="Cycle"):
            compute_layers(flat, set())


# -----------------------------------------------------------------------
# infer_input_keys
# -----------------------------------------------------------------------

class TestInferInputKeys:
    def test_simple(self, simple_specs: list[NodeSpec]) -> None:
        flat = flatten_nodes(simple_specs)
        keys = infer_input_keys(flat)
        assert keys == {"input1", "input2", "input3"}

    def test_stepped(self, stepped_specs: list[NodeSpec]) -> None:
        flat = flatten_nodes(stepped_specs)
        keys = infer_input_keys(flat)
        assert keys == {"input1", "input2", "input3"}


# -----------------------------------------------------------------------
# build_dot
# -----------------------------------------------------------------------

class TestBuildDot:
    def test_dot_contains_nodes(self, simple_specs: list[NodeSpec]) -> None:
        flat = flatten_nodes(simple_specs)
        dot = build_dot(flat, INPUT_KEYS)
        assert "digraph pipeline" in dot
        assert '"A"' in dot
        assert '"B"' in dot
        assert '"C"' in dot
