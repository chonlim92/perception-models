"""
Functional Scenario Trees - Taxonomy Module

A 6-layer hierarchical scenario classification system based on the PEGASUS/ASAM
taxonomy for autonomous driving data management.

Layers:
    1. Road Topology - road types, intersections, lanes, geometry
    2. Traffic Infrastructure - lights, signs, markings, barriers
    3. Temporary Modifications - construction, closures, events
    4. Dynamic Objects - vehicles, pedestrians, behaviors
    5. Environment - weather, lighting, road surface
    6. Digital Information - sensor degradation, map accuracy, V2X
"""

from .scenario_tree import (
    ScenarioTreeNode,
    build_default_tree,
    get_leaf_nodes,
    get_node_by_id,
    get_nodes_by_layer,
)
from .scenario_schema import (
    ScenarioAnnotation,
    ScenarioQuery,
    ScenarioTag,
    ScenarioTreeModel,
    ScenarioTreeNodeModel,
)
from .tree_visualization import (
    render_graphviz,
    render_html,
    render_text,
)

__all__ = [
    # Tree structure
    "ScenarioTreeNode",
    "build_default_tree",
    "get_node_by_id",
    "get_nodes_by_layer",
    "get_leaf_nodes",
    # Schema models
    "ScenarioTag",
    "ScenarioAnnotation",
    "ScenarioQuery",
    "ScenarioTreeModel",
    "ScenarioTreeNodeModel",
    # Visualization
    "render_text",
    "render_html",
    "render_graphviz",
]
