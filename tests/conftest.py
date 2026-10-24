"""Shared fixtures for tests."""

from __future__ import annotations

import pytest

from agentic.models import NodeSpec, NodeStep


# ---------------------------------------------------------------------------
# Simple pipeline specs (A, B, C – no steps)
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_specs() -> list[NodeSpec]:
    return [
        NodeSpec(
            id="A",
            produces="A",
            deps=["input1"],
            operator="GenericPromptOperator",
            prompt_template="tests/fixtures/generic.md",
            prompt_version="1.0",
            output_schema="GenericOutput",
        ),
        NodeSpec(
            id="B",
            produces="B",
            deps=["input2"],
            operator="GenericPromptOperator",
            prompt_template="tests/fixtures/generic.md",
            prompt_version="1.0",
            output_schema="GenericOutput",
        ),
        NodeSpec(
            id="C",
            produces="C",
            deps=["A", "input3"],
            operator="GenericPromptOperator",
            prompt_template="tests/fixtures/generic.md",
            prompt_version="1.0",
            output_schema="GenericOutput",
        ),
    ]


# ---------------------------------------------------------------------------
# Pipeline specs with A having internal steps
# ---------------------------------------------------------------------------

@pytest.fixture()
def stepped_specs() -> list[NodeSpec]:
    return [
        NodeSpec(
            id="A",
            produces="A",
            deps=["input1"],
            steps=[
                NodeStep(
                    id="A_prime",
                    produces="A_prime",
                    deps=["input1"],
                    operator="GenericPromptOperator",
                    prompt_template="tests/fixtures/generic.md",
                    prompt_version="1.0",
                    output_schema="GenericOutput",
                ),
                NodeStep(
                    id="A_final",
                    produces="A",
                    deps=["A_prime", "input1"],
                    operator="GenericPromptOperator",
                    prompt_template="tests/fixtures/generic.md",
                    prompt_version="1.0",
                    output_schema="GenericOutput",
                ),
            ],
        ),
        NodeSpec(
            id="B",
            produces="B",
            deps=["input2"],
            operator="GenericPromptOperator",
            prompt_template="tests/fixtures/generic.md",
            prompt_version="1.0",
            output_schema="GenericOutput",
        ),
        NodeSpec(
            id="C",
            produces="C",
            deps=["A", "input3"],
            operator="GenericPromptOperator",
            prompt_template="tests/fixtures/generic.md",
            prompt_version="1.0",
            output_schema="GenericOutput",
        ),
    ]


@pytest.fixture()
def sample_inputs() -> dict[str, str]:
    return {
        "input1": "data for A",
        "input2": "data for B",
        "input3": "data for C",
    }
