"""CLI entry-point – ``agentic`` command."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import typer
import yaml
from rich.console import Console

load_dotenv()
from rich.table import Table

from .models import FileInput, NodeSpec, PipelineSpec
from .runtime.executor import Executor, load_all_node_specs, load_pipeline
from .runtime.graph import build_dot, compute_layers, flatten_nodes, infer_input_keys

app = typer.Typer(
    name="agentic",
    help="Agentic DAG Execution Framework",
    no_args_is_help=True,
)
node_app = typer.Typer(help="Node inspection commands")
app.add_typer(node_app, name="node")

console = Console()


# -----------------------------------------------------------------------
# agentic run
# -----------------------------------------------------------------------


def _default_llm() -> str:
    """GEMINI_API_KEY가 .env에 있으면 gemini, 없으면 mock."""
    return "gemini" if os.environ.get("GEMINI_API_KEY") else "mock"


def _build_llm_client(llm: str, api_key: str | None):
    """Resolve an LLM client by name."""
    if llm == "mock":
        from .llm.mock import MockLLMClient

        return MockLLMClient()
    if llm == "gemini":
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print(
                "[red]GEMINI_API_KEY not found. Set it in .env or pass --api-key[/red]"
            )
            raise typer.Exit(1)
        from .llm.gemini import GeminiClient

        return GeminiClient(api_key=api_key)
    if llm == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            console.print(
                "[red]OPENAI_API_KEY not found. Set it in .env or pass --api-key[/red]"
            )
            raise typer.Exit(1)
        from .llm.openai import OpenAIClient

        return OpenAIClient(api_key=api_key)
    console.print(f"[red]Unknown LLM: {llm}. Available: mock, gemini, openai[/red]")
    raise typer.Exit(1)


def _parse_inputs(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert ``{"_file": "path"}`` entries to :class:`FileInput` objects."""
    parsed: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict) and "_file" in value:
            parsed[key] = FileInput(
                filename=Path(value["_file"]).name,
                path=value["_file"],
            )
        else:
            parsed[key] = value
    return parsed


@app.command()
def run(
    pipeline: str = typer.Argument(..., help="Path to pipeline YAML"),
    inputs: str = typer.Option(..., "--inputs", help="Path to inputs JSON"),
    llm: str = typer.Option(
        "", "--llm", help="LLM backend: mock | gemini | openai (auto-detect from .env)"
    ),
    api_key: str = typer.Option(None, "--api-key", help="API key for LLM service"),
) -> None:
    """Execute a pipeline end-to-end."""
    raw_data = json.loads(Path(inputs).read_text(encoding="utf-8"))
    inputs_data = _parse_inputs(raw_data)

    resolved_llm = llm if llm else _default_llm()
    llm_client = _build_llm_client(resolved_llm, api_key)
    executor = Executor(llm_client=llm_client)
    result = asyncio.run(executor.run(pipeline, inputs_data))

    # Print only node outputs (skip raw inputs)
    console.print("\n[bold green]Pipeline completed.[/bold green]\n")
    for key, value in result.outputs_only().items():
        console.print(f"  [cyan]{key}[/cyan]: {json.dumps(value, ensure_ascii=False)}")
    console.print()


# -----------------------------------------------------------------------
# agentic node show <ID>
# -----------------------------------------------------------------------


@node_app.command("show")
def node_show(
    node_id: str = typer.Argument(..., help="Node ID to inspect"),
    node_dir: str = typer.Option(
        "src/agentic/nodes", "--node-dir", help="Directory containing node YAML files"
    ),
) -> None:
    """Show spec details for a node."""
    dir_path = Path(node_dir)
    if not dir_path.is_dir():
        console.print(f"[red]Node directory not found: {dir_path}[/red]")
        raise typer.Exit(1)

    for yaml_file in sorted(dir_path.rglob("*.yaml")):
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        if data.get("id") == node_id:
            spec = NodeSpec(**data, source_file=str(yaml_file))
            _print_node_spec(spec, yaml_file)
            return

    console.print(f"[red]Node '{node_id}' not found in {dir_path}[/red]")
    raise typer.Exit(1)


def _print_node_spec(spec: NodeSpec, path: Path) -> None:
    table = Table(title=f"Node: {spec.id}", show_header=False, show_lines=True)
    table.add_column("Field", style="bold cyan", width=18)
    table.add_column("Value")

    table.add_row("File", str(path))
    table.add_row("Produces", spec.produces)
    table.add_row("Deps", ", ".join(spec.deps) if spec.deps else "(none)")
    table.add_row("Operator", spec.operator or "(composite – see steps)")
    table.add_row("Prompt Template", spec.prompt_template or "(via steps)")
    table.add_row("Prompt Version", spec.prompt_version)
    table.add_row("Output Schema", spec.output_schema or "(none)")
    table.add_row("Context Scope", spec.context_scope)

    if spec.steps:
        step_lines = "\n".join(
            f"  {i+1}. {s.id} -> {s.produces}  (deps: {s.deps})"
            for i, s in enumerate(spec.steps)
        )
        table.add_row("Steps", step_lines)

    console.print(table)


# -----------------------------------------------------------------------
# agentic layers
# -----------------------------------------------------------------------


@app.command()
def layers(
    pipeline: str = typer.Argument(..., help="Path to pipeline YAML"),
    inputs: str = typer.Option(
        None,
        "--inputs",
        help="Path to inputs JSON (optional – keys inferred if absent)",
    ),
) -> None:
    """Display DAG execution layers."""
    pipe = load_pipeline(pipeline)
    specs = load_all_node_specs(pipe)
    flat = flatten_nodes(specs)

    if inputs:
        input_keys = set(json.loads(Path(inputs).read_text(encoding="utf-8")).keys())
    else:
        input_keys = infer_input_keys(flat)

    layer_list = compute_layers(flat, input_keys)

    console.print(f"\n[bold]Pipeline:[/bold] {pipe.name}")
    console.print(f"[bold]Input keys:[/bold] {sorted(input_keys)}\n")

    table = Table(title="Execution Layers")
    table.add_column("Layer", style="bold", justify="center")
    table.add_column("Nodes")
    table.add_column("Type", justify="center")

    # Layer 0: inputs (data, not executed)
    table.add_row("0", ", ".join(sorted(input_keys)), "input")

    for idx, layer in enumerate(layer_list, start=1):
        parallel = "parallel" if len(layer) > 1 else "serial"
        table.add_row(str(idx), ", ".join(layer), parallel)

    console.print(table)
    console.print()


# -----------------------------------------------------------------------
# agentic graph
# -----------------------------------------------------------------------


@app.command()
def graph(
    pipeline: str = typer.Argument(..., help="Path to pipeline YAML"),
    format: str = typer.Option("dot", "--format", help="Output format (dot)"),
    inputs: str = typer.Option(None, "--inputs", help="Path to inputs JSON (optional)"),
) -> None:
    """Output the DAG in DOT format."""
    pipe = load_pipeline(pipeline)
    specs = load_all_node_specs(pipe)
    flat = flatten_nodes(specs)

    if inputs:
        input_keys = set(json.loads(Path(inputs).read_text(encoding="utf-8")).keys())
    else:
        input_keys = infer_input_keys(flat)

    if format != "dot":
        console.print(
            f"[red]Unsupported format: {format}. Only 'dot' is supported.[/red]"
        )
        raise typer.Exit(1)

    from .runtime.graph import extract_input_specs

    dot = build_dot(flat, input_keys, extract_input_specs(specs))
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(dot)


# -----------------------------------------------------------------------
# Entry-point guard
# -----------------------------------------------------------------------

if __name__ == "__main__":
    app()
