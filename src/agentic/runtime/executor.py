"""DAG executor – the main runtime engine.

Loads a pipeline, builds the DAG, resolves layers, and executes nodes
layer-by-layer with ``asyncio.gather`` for intra-layer parallelism.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from ..llm.base import BaseLLMClient
from ..llm.mock import MockLLMClient
from ..models import (
    FileInput,
    FlatNode,
    InputNode,
    NodeSpec,
    PipelineResult,
    PipelineSpec,
)
from ..schemas import get_schema
from .graph import compute_layers, extract_input_specs, flatten_nodes
from .registry import get_operator


def _safe_json(obj: Any) -> str:
    """json.dumps with fallback for non-serializable types (numpy int64 etc.)."""
    def _default(o: Any) -> Any:
        if hasattr(o, 'item'):  # numpy scalar
            return o.item()
        return str(o)
    return json.dumps(obj, ensure_ascii=False, default=_default)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_pipeline(pipeline_path: str | Path) -> PipelineSpec:
    path = Path(pipeline_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return PipelineSpec(**data)


def load_node_spec(node_name: str, node_dir: str | Path) -> NodeSpec:
    path = Path(node_dir) / f"{node_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Node spec not found: {path}. "
            f"Ensure '{node_name}.yaml' exists in '{node_dir}'"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["source_file"] = str(path)
    return NodeSpec(**data)


def load_all_node_specs(pipeline: PipelineSpec) -> list[NodeSpec]:
    return [load_node_spec(name, pipeline.node_dir) for name in pipeline.nodes]


def load_config(config_dir: str | Path | None) -> dict[str, Any]:
    """Load all JSON files from config_dir into a dict.

    Each file's stem becomes the key, its parsed content becomes the value.
    e.g. ``config/settings.json`` → ``{"settings": {"output_language": "English"}}``
    """
    if config_dir is None:
        return {}
    path = Path(config_dir)
    if not path.is_dir():
        return {}
    result: dict[str, Any] = {}
    for f in sorted(path.glob("*.json")):
        result[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return result


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def _resolve_context(
    node: FlatNode,
    context: dict[str, Any],
    completed_keys: set[str],
) -> dict[str, Any]:
    """Build the input dict passed to prompt rendering."""
    if node.context_scope == "deps_only":
        return {k: context[k] for k in node.deps if k in context}

    # all_previous
    result = {k: context[k] for k in completed_keys if k in context}

    if node.context_allowlist is not None:
        result = {k: v for k, v in result.items() if k in node.context_allowlist}
    if node.context_denylist is not None:
        result = {k: v for k, v in result.items() if k not in node.context_denylist}
    if node.context_max_keys is not None:
        ordered = list(node.deps) + [k for k in result if k not in node.deps]
        result = {k: result[k] for k in ordered[: node.context_max_keys] if k in result}

    return result


def _render_prompt(
    template_path: str | None,
    node_context: dict[str, Any],
    node: FlatNode,
    config: dict[str, Any] | None = None,
) -> str:
    if template_path is None:
        return ""
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    # Use FileSystemLoader so {% include %} works inside templates.
    search_dirs = [str(path.parent)]
    prompts_dir = path.parent  # e.g. prompts/
    if str(prompts_dir) not in search_dirs:
        search_dirs.append(str(prompts_dir))
    env = Environment(
        loader=FileSystemLoader(search_dirs, encoding="utf-8"),
        keep_trailing_newline=True,
    )
    template = env.get_template(path.name)
    # Config values are always available in templates (global constants).
    # node_context values take priority on collision.
    render_vars: dict[str, Any] = {}
    if config:
        render_vars.update(config)
    render_vars.update(node_context)
    return template.render(
        node_id=node.id,
        produces=node.produces,
        inputs=node_context,
        **render_vars,
    )


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class Executor:
    """Async DAG executor with layer-parallel scheduling."""

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
    ) -> None:
        self.llm_client = llm_client or MockLLMClient()
        self._temp_dir: Path | None = None
        self._config: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        pipeline_path: str | Path,
        inputs: dict[str, Any],
    ) -> PipelineResult:
        """Load and execute a pipeline."""
        pipeline = load_pipeline(pipeline_path)
        node_specs = load_all_node_specs(pipeline)
        config = load_config(pipeline.config_dir)
        return await self.run_specs(node_specs, inputs, config=config)

    @staticmethod
    def _print_inputs(
        raw_inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> None:
        """Print pipeline inputs and config summary."""
        sep = "=" * 60
        print(sep)
        print(" PIPELINE INPUTS")
        print(sep)
        for key, value in raw_inputs.items():
            if isinstance(value, FileInput):
                print(f"  [{key}] (file) {value.filename}")
            elif isinstance(value, dict):
                preview = _safe_json(value)
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                print(f"  [{key}] (dict) {preview}")
            elif isinstance(value, list):
                print(f"  [{key}] (list) {len(value)} items")
            elif isinstance(value, str):
                if len(value) > 200:
                    print(f"  [{key}] {value[:200]}...")
                else:
                    print(f"  [{key}] {value}")
            else:
                print(f"  [{key}] {value}")
        if config:
            print(f"\n CONFIG ({len(config)} files)")
            for key, value in config.items():
                if isinstance(value, dict):
                    preview = _safe_json(value)
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    print(f"  [{key}] {preview}")
                elif isinstance(value, list):
                    print(f"  [{key}] (list) {len(value)} items")
                else:
                    print(f"  [{key}] {value}")
        print(sep + "\n")

    @staticmethod
    def _print_outputs(context: dict[str, Any], terminal_keys: set[str]) -> None:
        """Print only terminal pipeline outputs (not consumed by other nodes)."""
        output = ""
        sep = "=" * 60
        output += "\n" + sep + "\n"
        output += " PIPELINE OUTPUTS\n"
        output += sep + "\n"
        for key, value in context.items():
            if key not in terminal_keys:
                continue
            if isinstance(value, dict):
                output += f"  [{key}] {_safe_json(value)}\n"
            elif isinstance(value, list):
                output += f"  [{key}] (list, {len(value)} items) {_safe_json(value)}\n"
            elif isinstance(value, str):
                output += f"  [{key}] {value}\n"
            else:
                output += f"  [{key}] {value}\n"
        output += sep
        print({"output": output})

    async def run_specs(
        self,
        node_specs: list[NodeSpec],
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """Execute an already-loaded list of node specs."""
        try:
            self._config = config or {}
            self._print_inputs(inputs, self._config)

            input_specs = extract_input_specs(node_specs)
            context_inputs, input_nodes = self._prepare_inputs(inputs, input_specs)

            # Inject config into context (inputs take priority on collision)
            for k, v in self._config.items():
                if k not in context_inputs:
                    context_inputs[k] = v

            flat_nodes = flatten_nodes(node_specs)
            node_map: dict[str, FlatNode] = {n.id: n for n in flat_nodes}
            input_keys = set(context_inputs.keys())

            layers = compute_layers(flat_nodes, input_keys)

            context: dict[str, Any] = dict(context_inputs)
            completed_keys: set[str] = set(context_inputs.keys())

            for layer in layers:
                tasks = [
                    self._execute_node(node_map[nid], context, completed_keys)
                    for nid in layer
                ]
                results = await asyncio.gather(*tasks)

                for nid, result in zip(layer, results):
                    node = node_map[nid]
                    context[node.produces] = result
                    completed_keys.add(node.produces)

            # Terminal outputs: produces keys not consumed as deps by any node
            all_deps: set[str] = set()
            for n in flat_nodes:
                all_deps.update(n.deps)
            config_keys = set(self._config.keys())
            terminal_keys = {n.produces for n in flat_nodes} - all_deps - input_keys - config_keys
            self._print_outputs(context, terminal_keys)

            return PipelineResult(
                context=context,
                input_nodes=input_nodes,
            )
        finally:
            self._cleanup()

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self,
        raw_inputs: dict[str, Any],
        input_specs: list[NodeSpec] | None = None,
    ) -> tuple[dict[str, Any], dict[str, InputNode]]:
        """Process raw inputs: materialize FileInput to temp files."""
        context_inputs: dict[str, Any] = {}
        input_nodes: dict[str, InputNode] = {}

        input_desc = (
            {s.produces: s.description for s in input_specs} if input_specs else {}
        )

        for key, value in raw_inputs.items():
            desc = input_desc.get(key, "")
            if isinstance(value, FileInput):
                if self._temp_dir is None:
                    self._temp_dir = Path(tempfile.mkdtemp(prefix="agentic_"))

                file_path = self._temp_dir / value.filename

                if value.data is not None:
                    file_path.write_bytes(value.data)
                elif value.path is not None:
                    shutil.copy2(value.path, file_path)

                context_inputs[key] = str(file_path)
                input_nodes[key] = InputNode(
                    key=key,
                    kind="file",
                    temp_path=str(file_path),
                    description=desc,
                )
            else:
                context_inputs[key] = value
                input_nodes[key] = InputNode(key=key, kind="text", description=desc)

        return context_inputs, input_nodes

    def _cleanup(self) -> None:
        """Remove temp directory if it was created."""
        if self._temp_dir is not None and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _execute_node(
        self,
        node: FlatNode,
        context: dict[str, Any],
        completed_keys: set[str],
    ) -> Any:
        # 1. Verify all deps are in context
        missing = [d for d in node.deps if d not in context]
        if missing:
            raise RuntimeError(
                f"Node '{node.id}' is missing deps: {missing}. "
                f"Available keys: {sorted(context.keys())}"
            )

        # 2. Build node-specific input context
        node_context = _resolve_context(node, context, completed_keys)
        node_context["_node_id"] = node.id

        # 3. Instantiate operator
        operator = get_operator(node.operator, self.llm_client)

        # 4. Render prompt (config always available in templates)
        rendered = _render_prompt(node.prompt_template, node_context, node, self._config)

        # 5. Execute
        schema = get_schema(node.output_schema)
        result = await operator.execute(rendered, schema, context=node_context)

        return result
