"""DAG construction, topological sort, and layer computation."""

from __future__ import annotations

from ..models import FlatNode, NodeSpec


# ---------------------------------------------------------------------------
# Step expansion
# ---------------------------------------------------------------------------


def flatten_nodes(specs: list[NodeSpec]) -> list[FlatNode]:
    """Expand composite nodes (those with *steps*) into flat nodes.

    Simple nodes pass through unchanged.  Composite nodes are replaced by
    their internal steps – the last step whose ``produces`` matches the
    parent's ``produces`` becomes the effective output for downstream nodes.
    Input nodes (``kind="input"``) are skipped – they are not executed.
    """
    flat: list[FlatNode] = []

    for spec in specs:
        if spec.kind == "input":
            continue
        if spec.steps:
            for step in spec.steps:
                flat.append(
                    FlatNode(
                        id=step.id,
                        produces=step.produces,
                        deps=list(step.deps),
                        operator=step.operator,
                        prompt_template=step.prompt_template,
                        prompt_version=step.prompt_version,
                        output_schema=step.output_schema,
                        context_scope=step.context_scope,
                        context_allowlist=step.context_allowlist,
                        context_denylist=step.context_denylist,
                        context_max_keys=step.context_max_keys,
                        parent_node_id=spec.id,
                    )
                )
        else:
            assert spec.operator is not None
            flat.append(
                FlatNode(
                    id=spec.id,
                    produces=spec.produces,
                    deps=list(spec.deps),
                    operator=spec.operator,
                    prompt_template=spec.prompt_template,
                    prompt_version=spec.prompt_version,
                    output_schema=spec.output_schema,
                    context_scope=spec.context_scope,
                    context_allowlist=spec.context_allowlist,
                    context_denylist=spec.context_denylist,
                    context_max_keys=spec.context_max_keys,
                )
            )

    # Validate: no duplicate produces keys
    seen: dict[str, str] = {}
    for node in flat:
        if node.produces in seen:
            raise ValueError(
                f"Duplicate produces key '{node.produces}': "
                f"both '{seen[node.produces]}' and '{node.id}' produce it"
            )
        seen[node.produces] = node.id

    return flat


# ---------------------------------------------------------------------------
# Topological sort with layer calculation (Kahn's algorithm)
# ---------------------------------------------------------------------------


def infer_input_keys(flat_nodes: list[FlatNode]) -> set[str]:
    """Derive input keys: deps that no node produces."""
    all_deps: set[str] = set()
    all_produces: set[str] = set()
    for n in flat_nodes:
        all_deps.update(n.deps)
        all_produces.add(n.produces)
    return all_deps - all_produces


def compute_layers(
    flat_nodes: list[FlatNode],
    input_keys: set[str],
) -> list[list[str]]:
    """Return layers of node IDs.  Nodes in the same layer are independent
    and can execute in parallel.

    Raises ``ValueError`` on missing deps or cycles.
    """
    produces_to_node: dict[str, str] = {n.produces: n.id for n in flat_nodes}
    in_degree: dict[str, int] = {n.id: 0 for n in flat_nodes}
    forward: dict[str, list[str]] = {n.id: [] for n in flat_nodes}

    for n in flat_nodes:
        for dep_key in n.deps:
            if dep_key in produces_to_node:
                producer_id = produces_to_node[dep_key]
                forward[producer_id].append(n.id)
                in_degree[n.id] += 1
            elif dep_key not in input_keys:
                raise ValueError(
                    f"Node '{n.id}' depends on '{dep_key}' which is neither "
                    f"an input key nor produced by any node.\n"
                    f"  Available inputs:   {sorted(input_keys)}\n"
                    f"  Available produces: {sorted(produces_to_node.keys())}"
                )

    layers: list[list[str]] = []
    queue = sorted(nid for nid, deg in in_degree.items() if deg == 0)

    while queue:
        layers.append(queue)
        next_queue: list[str] = []
        for nid in queue:
            for dep_id in forward[nid]:
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    next_queue.append(dep_id)
        queue = sorted(next_queue)

    total = sum(len(layer) for layer in layers)
    if total != len(flat_nodes):
        executed = {nid for layer in layers for nid in layer}
        remaining = sorted(n.id for n in flat_nodes if n.id not in executed)
        raise ValueError(f"Cycle detected in DAG. Stuck nodes: {remaining}")

    return layers


# ---------------------------------------------------------------------------
# DOT graph output
# ---------------------------------------------------------------------------


def extract_input_specs(specs: list[NodeSpec]) -> list[NodeSpec]:
    """Return only input-kind node specs."""
    return [s for s in specs if s.kind == "input"]


def build_dot(
    flat_nodes: list[FlatNode],
    input_keys: set[str],
    input_specs: list[NodeSpec] | None = None,
) -> str:
    """Generate a Graphviz DOT representation of the DAG."""
    lines = [
        "digraph pipeline {",
        "  rankdir=LR;",
        '  node [fontname="Malgun Gothic"];',
        '  edge [fontname="Malgun Gothic"];',
        '  graph [fontname="Malgun Gothic"];',
        "",
    ]

    # Build description lookup from input specs
    input_desc: dict[str, str] = {}
    if input_specs:
        for s in input_specs:
            input_desc[s.produces] = s.description

    # Input nodes as ellipses
    for key in sorted(input_keys):
        desc = input_desc.get(key, "")
        label = f"{key}\\n{desc}" if desc else key
        lines.append(
            f'  "{key}" [shape=ellipse, style=filled, '
            f'fillcolor=lightblue, label="{label}"];'
        )
    lines.append("")

    # Processing nodes as boxes
    for n in flat_nodes:
        prompt_name = n.prompt_template.rsplit("/", 1)[-1] if n.prompt_template else ""
        label = f"{n.id}\\n[{n.operator}]"
        if prompt_name:
            label += f"\\n{prompt_name}"

        color = "lightyellow" if n.parent_node_id is None else "lightgreen"
        lines.append(
            f'  "{n.id}" [shape=box, style=filled, '
            f'fillcolor={color}, label="{label}"];'
        )
    lines.append("")

    # Edges
    produces_to_node = {n.produces: n.id for n in flat_nodes}
    for n in flat_nodes:
        for dep in n.deps:
            source = produces_to_node.get(dep, dep)  # input key or node
            lines.append(f'  "{source}" -> "{n.id}";')

    lines.append("}")
    return "\n".join(lines)
