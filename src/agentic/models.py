"""Core data models for node specifications and pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, model_validator


class NodeStep(BaseModel):
    """A single step within a composite node."""

    id: str
    produces: str
    deps: list[str] = []
    operator: str
    prompt_template: str | None = None
    prompt_version: str = "1.0"
    output_schema: str | None = None
    context_scope: Literal["deps_only", "all_previous"] = "deps_only"
    context_allowlist: list[str] | None = None
    context_denylist: list[str] | None = None
    context_max_keys: int | None = None


class NodeSpec(BaseModel):
    """Specification for a DAG node.

    * ``kind="input"``: external input node — no operator or steps needed.
    * ``kind="process"`` (default): either ``operator`` (simple) or ``steps``
      (composite) must be set.
    """

    id: str
    produces: str
    kind: Literal["input", "process"] = "process"
    deps: list[str] = []
    operator: str | None = None
    prompt_template: str | None = None
    prompt_version: str = "1.0"
    output_schema: str | None = None
    context_scope: Literal["deps_only", "all_previous"] = "deps_only"
    context_allowlist: list[str] | None = None
    context_denylist: list[str] | None = None
    context_max_keys: int | None = None
    steps: list[NodeStep] | None = None
    description: str = ""

    # Metadata – populated at load time
    source_file: str | None = None

    @model_validator(mode="after")
    def _validate_operator_or_steps(self) -> NodeSpec:
        if self.kind == "input":
            return self
        if not self.steps and not self.operator:
            raise ValueError(
                f"Node '{self.id}': must have either 'operator' or 'steps'"
            )
        if self.steps:
            step_produces = {s.produces for s in self.steps}
            if self.produces not in step_produces:
                raise ValueError(
                    f"Node '{self.id}' has steps but none produces '{self.produces}'"
                )
        return self


class FlatNode(BaseModel):
    """A flattened node (after step expansion) ready for DAG scheduling."""

    id: str
    produces: str
    deps: list[str] = []
    operator: str
    prompt_template: str | None = None
    prompt_version: str = "1.0"
    output_schema: str | None = None
    context_scope: Literal["deps_only", "all_previous"] = "deps_only"
    context_allowlist: list[str] | None = None
    context_denylist: list[str] | None = None
    context_max_keys: int | None = None
    parent_node_id: str | None = None


class PipelineSpec(BaseModel):
    """Pipeline definition – references node YAML files."""

    name: str
    description: str = ""
    node_dir: str = "src/agentic/nodes"
    config_dir: str | None = "config"
    nodes: list[str]


# ---------------------------------------------------------------------------
# Input handling & result types
# ---------------------------------------------------------------------------


@dataclass
class FileInput:
    """Wraps file data for an input node.

    Provide either ``path`` (local file) or ``data`` (raw bytes).
    The executor copies/writes the content to a temp directory so that
    downstream nodes can read it via a stable path.
    """

    filename: str
    path: str | None = None
    data: bytes | None = None
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        if self.path is None and self.data is None:
            raise ValueError("FileInput requires either 'path' or 'data'")


class InputNode(BaseModel):
    """First-class representation of a pipeline input."""

    key: str
    kind: Literal["text", "file"] = "text"
    description: str = ""
    temp_path: str | None = None


class PipelineResult(BaseModel):
    """Result of a pipeline execution."""

    context: dict[str, Any] = field(default_factory=dict)
    input_nodes: dict[str, InputNode] = field(default_factory=dict)
    llm_logs: dict[str, list[dict[str, str]]] = field(default_factory=dict)

    def outputs_only(self) -> dict[str, Any]:
        """Return only computation-node outputs (exclude raw inputs)."""
        return {k: v for k, v in self.context.items() if k not in self.input_nodes}

    # dict-like access for backward compatibility
    def __getitem__(self, key: str) -> Any:
        return self.context[key]

    def __contains__(self, key: object) -> bool:
        return key in self.context

    def items(self):  # noqa: ANN201
        return self.context.items()

    def keys(self):  # noqa: ANN201
        return self.context.keys()

    def to_markdown(self, output_path: str | Path | None = None) -> str:
        """Render pipeline results as a markdown report.

        If *output_path* is given the report is also written to that file.
        """
        lines: list[str] = ["# Pipeline Result Report", ""]

        for key, value in self.outputs_only().items():
            lines.append(f"## {key}")
            lines.append("")

            if isinstance(value, dict):
                for k, v in value.items():
                    lines.append(f"### {k}")
                    lines.append("")
                    lines.append(str(v) if v else "*(empty)*")
                    lines.append("")
            elif isinstance(value, list):
                for item in value:
                    lines.append(
                        f"- {json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else item}"
                    )
                lines.append("")
            else:
                lines.append(str(value))
                lines.append("")

            lines.append("---")
            lines.append("")

        report = "\n".join(lines)

        if output_path is not None:
            Path(output_path).write_text(report, encoding="utf-8")

        return report

    def to_txt(self, output_path: str | Path | None = None) -> str:
        """Render ALL pipeline results (including inputs) as a simple plain text report."""
        lines: list[str] = [
            "========================================",
            "PIPELINE EXECUTION RESULT (PLAIN TEXT)",
            "========================================",
            "",
        ]

        # input_nodes first
        if self.input_nodes:
            lines.append("--- [INPUT NODES] ---")
            for key in sorted(self.input_nodes.keys()):
                val = self.context.get(key)
                lines.append(f"[{key}]")
                lines.append(str(val))
                lines.append("")

        # Then processing nodes
        outputs = self.outputs_only()
        if outputs:
            lines.append("--- [PROCESS/OUTPUT NODES] ---")
            for key in sorted(outputs.keys()):
                val = outputs[key]
                lines.append(f"[{key}]")
                if isinstance(val, (dict, list)):
                    lines.append(json.dumps(val, ensure_ascii=False, indent=2))
                else:
                    lines.append(str(val))
                lines.append("")

        report = "\n".join(lines)

        if output_path is not None:
            Path(output_path).write_text(report, encoding="utf-8")

        return report
