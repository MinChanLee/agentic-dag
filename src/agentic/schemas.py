"""Output schema registry.

Register pydantic models by name so that node specs can reference them as
strings (e.g. ``output_schema: "GenericOutput"``).
"""

from __future__ import annotations

from pydantic import BaseModel


class GenericOutput(BaseModel):
    """Default output schema – suitable for most generic prompt nodes."""

    result: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "GenericOutput": GenericOutput,
}


def get_schema(name: str | None) -> type[BaseModel] | None:
    """Look up a schema class by name. Returns *None* when *name* is None."""
    if name is None:
        return None
    schema = SCHEMA_REGISTRY.get(name)
    if schema is None:
        available = ", ".join(sorted(SCHEMA_REGISTRY.keys()))
        raise ValueError(f"Unknown output schema '{name}'. Available: [{available}]")
    return schema


def register_schema(name: str, schema: type[BaseModel]) -> None:
    """Register a custom output schema for use in node specs."""
    SCHEMA_REGISTRY[name] = schema
