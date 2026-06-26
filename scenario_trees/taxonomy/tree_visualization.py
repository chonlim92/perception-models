"""
Visualization utilities for the Functional Scenario Tree.

Supports rendering as:
  - ASCII text tree
  - HTML with collapsible sections
  - Graphviz DOT format
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .scenario_tree import ScenarioTreeNode


def render_text(tree: "ScenarioTreeNode", max_depth: Optional[int] = None) -> str:
    """
    Render the scenario tree as an ASCII text tree with indentation and box-drawing chars.

    Args:
        tree: The root ScenarioTreeNode to render.
        max_depth: Maximum depth to render (None = unlimited). Root is depth 0.

    Returns:
        Multi-line string representation of the tree.
    """
    lines: list[str] = []
    _render_text_recursive(tree, lines, prefix="", is_last=True, depth=0, max_depth=max_depth)
    return "\n".join(lines)


def _render_text_recursive(
    node: "ScenarioTreeNode",
    lines: list[str],
    prefix: str,
    is_last: bool,
    depth: int,
    max_depth: Optional[int],
) -> None:
    """Recursively build text lines for the tree."""
    if max_depth is not None and depth > max_depth:
        return

    # Connector characters
    if depth == 0:
        connector = ""
        child_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        child_prefix = "    " if is_last else "│   "

    # Format the node line
    layer_tag = f"[L{node.layer}]" if node.layer > 0 else "[ROOT]"
    label = f"{node.name} {layer_tag}"
    if node.id and node.id != "ROOT":
        label = f"({node.id}) {label}"

    lines.append(f"{prefix}{connector}{label}")

    # Recurse into children
    new_prefix = prefix + child_prefix
    for i, child in enumerate(node.children):
        is_child_last = i == len(node.children) - 1
        _render_text_recursive(child, lines, new_prefix, is_child_last, depth + 1, max_depth)


def render_html(tree: "ScenarioTreeNode") -> str:
    """
    Render the scenario tree as an HTML document with collapsible ul/li sections.

    Uses <details>/<summary> elements for native collapsibility without JavaScript.

    Args:
        tree: The root ScenarioTreeNode to render.

    Returns:
        Complete HTML string.
    """
    html_parts: list[str] = []

    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Functional Scenario Tree</title>
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 2rem;
    background: #f8f9fa;
    color: #212529;
  }
  h1 { color: #1a1a2e; margin-bottom: 0.5rem; }
  ul {
    list-style: none;
    padding-left: 1.5rem;
    margin: 0.2rem 0;
  }
  li { margin: 0.2rem 0; }
  details { margin: 0.2rem 0; }
  summary {
    cursor: pointer;
    padding: 0.3rem 0.5rem;
    border-radius: 4px;
    transition: background 0.2s;
  }
  summary:hover { background: #e9ecef; }
  .node-id {
    font-family: monospace;
    color: #6c757d;
    font-size: 0.85em;
    margin-right: 0.5rem;
  }
  .node-name { font-weight: 500; }
  .layer-badge {
    display: inline-block;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.75em;
    font-weight: 600;
    margin-left: 0.5rem;
    color: white;
  }
  .layer-0 { background: #495057; }
  .layer-1 { background: #d63384; }
  .layer-2 { background: #fd7e14; }
  .layer-3 { background: #ffc107; color: #212529; }
  .layer-4 { background: #198754; }
  .layer-5 { background: #0dcaf0; color: #212529; }
  .layer-6 { background: #6f42c1; }
  .description {
    color: #6c757d;
    font-size: 0.85em;
    font-style: italic;
    margin-left: 0.5rem;
  }
  .leaf-node {
    padding: 0.3rem 0.5rem;
  }
</style>
</head>
<body>
<h1>Functional Scenario Tree</h1>
<p>PEGASUS/ASAM-based 6-layer taxonomy for autonomous driving scenarios</p>
""")

    html_parts.append("<ul>")
    _render_html_node(tree, html_parts)
    html_parts.append("</ul>")

    html_parts.append("""
</body>
</html>""")

    return "\n".join(html_parts)


def _render_html_node(node: "ScenarioTreeNode", parts: list[str]) -> None:
    """Recursively render a node as HTML."""
    layer_class = f"layer-{node.layer}"
    node_id_span = f'<span class="node-id">{node.id}</span>' if node.id != "ROOT" else ""
    badge = f'<span class="layer-badge {layer_class}">L{node.layer}</span>' if node.layer > 0 else ""
    desc_span = f'<span class="description">- {node.description}</span>' if node.description else ""

    if node.children:
        parts.append("<li>")
        parts.append(f"<details open>")
        parts.append(
            f'<summary>{node_id_span}<span class="node-name">{node.name}</span>{badge}{desc_span}</summary>'
        )
        parts.append("<ul>")
        for child in node.children:
            _render_html_node(child, parts)
        parts.append("</ul>")
        parts.append("</details>")
        parts.append("</li>")
    else:
        parts.append(
            f'<li class="leaf-node">{node_id_span}<span class="node-name">{node.name}</span>{badge}{desc_span}</li>'
        )


def render_graphviz(tree: "ScenarioTreeNode") -> str:
    """
    Generate a Graphviz DOT format string for the scenario tree.

    Args:
        tree: The root ScenarioTreeNode to render.

    Returns:
        DOT language string that can be rendered with graphviz tools.
    """
    lines: list[str] = []
    lines.append("digraph ScenarioTree {")
    lines.append("    rankdir=TB;")
    lines.append("    node [shape=box, style=filled, fontname=\"Helvetica\", fontsize=10];")
    lines.append("    edge [arrowsize=0.7];")
    lines.append("")

    # Layer colors
    layer_colors = {
        0: "#495057",
        1: "#f8d7da",
        2: "#ffe5d0",
        3: "#fff3cd",
        4: "#d1e7dd",
        5: "#cff4fc",
        6: "#e2d9f3",
    }

    _render_graphviz_node(tree, lines, layer_colors)

    lines.append("")
    lines.append("    // Layer legend")
    lines.append("    subgraph cluster_legend {")
    lines.append('        label="Layers";')
    lines.append("        style=dashed;")
    lines.append('        legend_l1 [label="L1: Road Topology" fillcolor="#f8d7da"];')
    lines.append('        legend_l2 [label="L2: Traffic Infrastructure" fillcolor="#ffe5d0"];')
    lines.append('        legend_l3 [label="L3: Temporary Modifications" fillcolor="#fff3cd"];')
    lines.append('        legend_l4 [label="L4: Dynamic Objects" fillcolor="#d1e7dd"];')
    lines.append('        legend_l5 [label="L5: Environment" fillcolor="#cff4fc"];')
    lines.append('        legend_l6 [label="L6: Digital Information" fillcolor="#e2d9f3"];')
    lines.append("        legend_l1 -> legend_l2 -> legend_l3 -> legend_l4 -> legend_l5 -> legend_l6 [style=invis];")
    lines.append("    }")

    lines.append("}")
    return "\n".join(lines)


def _render_graphviz_node(
    node: "ScenarioTreeNode",
    lines: list[str],
    layer_colors: dict[int, str],
) -> None:
    """Recursively generate DOT statements for a node and its children."""
    # Sanitize ID for DOT (replace dots and special chars)
    dot_id = _to_dot_id(node.id)
    color = layer_colors.get(node.layer, "#ffffff")
    font_color = "#ffffff" if node.layer == 0 else "#212529"

    # Node declaration
    label = f"{node.id}\\n{node.name}"
    lines.append(
        f'    {dot_id} [label="{label}" fillcolor="{color}" fontcolor="{font_color}"];'
    )

    # Edges to children
    for child in node.children:
        child_dot_id = _to_dot_id(child.id)
        lines.append(f"    {dot_id} -> {child_dot_id};")

    # Recurse
    for child in node.children:
        _render_graphviz_node(child, lines, layer_colors)


def _to_dot_id(node_id: str) -> str:
    """Convert a node ID to a valid Graphviz identifier."""
    return node_id.replace(".", "_").replace("+", "plus").replace("-", "_")
